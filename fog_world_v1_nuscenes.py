"""
核心要点:
v1 = 在 v0 (单帧 6 路 GT 约束拼接) 基础上, 沿关键帧"开车前进", 把每个时刻的重建
累积进同一个米制全局世界 -> 战争迷雾式逐步揭示 -> 渲染自车视角前进视频。
版本:v1

相对 v0 的两处新增:
1. 多时刻: 沿 sample['next'] 走 T 个关键帧, 每帧仍做一次 v0 式的 6 路重建。
2. 米制对齐 + 累积: 每个时刻把该帧高斯钉进同一个 GT 米制全局系 ——
   朝向取 GT(6 路相机朝向平均, 不受相机共面影响), 尺度取地面 RANSAC + 已知相机离地高度,
   平移把相机质心对齐到 GT 全局; 再做跨时刻体素去重后累积。
   渲染时只画"累积到当前时刻"的高斯 -> 随车前进, 世界逐步显现(迷雾散开)。

为什么不用"预测相机中心"估尺度/旋转(v1 初版的坑):
6 路相机几乎共面、基线仅 ~2m, 用它们的中心外推到 ~30m 场景, 尺度会逐帧乱跳(实测 5.3~13.5),
导致各帧云大小差 2.5 倍叠在一起 -> 穿模。故改为: 旋转取 GT, 尺度取地面(上千点拟合, 稳)。

视角: 默认抬高第三人称跟拍(相机在 2.5D 壳外俯看, 不会被埋); --first_person 才是车内视角。

坐标系约定 (同 v0):
- nuScenes 相机系 = OpenCV(x右 y下 z前); 车身系 = x前 y左 z上; 全局系 z 朝上。
- prior 用"每个时刻自身 ego(CAM_FRONT) 参考系"下的 c2w (和 v0 一致, 数值小、好条件);
- Umeyama 的目标用"GT 全局系"下的相机中心 (带真实米制)。

CLI:
# GT 强约束 + 自车前进
python fog_world_v1_nuscenes.py --scene_index 0 --use_gt \
    --num_timesteps 12 --output_video outputs/v1_ego_forward.mp4 \
    --output_ply outputs/v1_fog_world.ply
"""
import os
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision.transforms import functional as TF

from diffsynth import save_video
from diffsynth.models import ModelManager
from diffsynth.utils.auxiliary import center_crop, homo_matrix_inverse
from diffsynth.auxiliary_models.worldmirror.utils.save_utils import save_gs_ply

# 六个视角
RING_ORDER = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
              'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
REF_CAM = 'CAM_FRONT'  # 以 CAM_FRONT 的车身系为参考


# 几何变换
def pose_to_matrix(translation, rotation):
    # nuScenes (平移[x,y,z], 四元数[w,x,y,z]) -> 4x4;
    w, x, y, z = rotation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = translation
    return T

def adjust_K_for_center_crop(K, src_wh, dst_wh):
    # 内参 K 随 center_crop(先等比放大覆盖, 再中心裁剪)同步修正。
    W, H = src_wh
    tw, th = dst_wh
    s = max(tw / W, th / H)
    ow, oh = int(W * s), int(H * s)
    left = (ow - tw) // 2
    top = (oh - th) // 2
    K2 = K.astype(np.float64).copy()
    K2[0, 0] *= s
    K2[1, 1] *= s
    K2[0, 2] = K[0, 2] * s - left
    K2[1, 2] = K[1, 2] * s - top
    return K2


def normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v) + eps)


def look_at_opencv(eye, target, up):
    # OpenCV(x右 y下 z前) look-at, 返回 c2w 4x4。
    z = normalize(target - eye)
    x = normalize(np.cross(z, up))
    y = np.cross(z, x)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = x, y, z
    c2w[:3, 3] = eye
    return c2w


# 米制对齐 + 高斯搬运
def rotation_align(a, b):
    # 把单位向量 a 旋到 b 的旋转矩阵 (Rodrigues)
    a, b = normalize(a), normalize(b)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3, dtype=np.float32) if c > 0 else -np.eye(3, dtype=np.float32)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return (np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))).astype(np.float32)


def average_rotation(Rs):
    # 多个旋转的 L2 均值: 对矩阵之和做 SVD 投影回 SO(3)
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R.astype(np.float32)


def level_and_scale(pts, R_rig, cam_center, known_h, thresh_frac=0.02, n_iter=400, min_pts=200):
    # 在按 GT 朝向旋正后的点云里拟合地面, 用"已知相机离地高度"定米制尺度(复用 v4)
    # 返回 (尺度 s, 残余整平旋转 R_lev); 拟合不可靠时返回 (None, I)
    p1 = pts @ R_rig.T                                   # 旋到全局朝向(网络尺度)
    below = p1[:, 2] < cam_center[2]
    cand = p1[below] if int(below.sum()) >= min_pts else p1
    if cand.shape[0] < min_pts:
        return None, np.eye(3, dtype=np.float32)
    thresh = thresh_frac * np.median(np.linalg.norm(cand - cand.mean(0), axis=1))
    up = np.array([0.0, 0.0, 1.0]); best_inl, best = 0, None
    for _ in range(n_iter):                              # RANSAC 找内点最多的近水平面
        tri = cand[np.random.choice(cand.shape[0], 3, replace=False)]
        nrm = np.cross(tri[1] - tri[0], tri[2] - tri[0]); ln = np.linalg.norm(nrm)
        if ln < 1e-8:
            continue
        nrm = nrm / ln
        if abs(nrm @ up) < 0.85:                         # 只收近水平面(排除墙/车头)
            continue
        d = -nrm @ tri[0]
        inl = int((np.abs(cand @ nrm + d) < thresh).sum())
        if inl > best_inl:
            best_inl, best = inl, (nrm, d)
    if best is None or best_inl < min_pts:
        return None, np.eye(3, dtype=np.float32)
    nrm, d = best
    if nrm[2] < 0:
        nrm, d = -nrm, -d
    dist = abs(nrm @ cam_center + d)                      # 相机到地面垂直距离(网络尺度)
    if dist < 1e-6:
        return None, np.eye(3, dtype=np.float32)
    return float(known_h / dist), rotation_align(nrm, up)


def transform_gaussians_(g, s, R_np, t_np, device):
    # 就地把相似变换 (s,R,t) 作用到一组高斯: 位置/尺度/朝向/速度一起变
    R = torch.tensor(R_np, dtype=torch.float32, device=device)
    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    g.means = s * (g.means.float() @ R.T) + t
    g.scales = g.scales.float() * s
    q = g.rotations.detach().cpu().numpy()                 # 四元数存为 [w,x,y,z]
    q_xyzw = q[:, [1, 2, 3, 0]]
    new_xyzw = (Rotation.from_matrix(R_np) * Rotation.from_quat(q_xyzw)).as_quat()
    new_wxyz = np.concatenate([new_xyzw[:, 3:4], new_xyzw[:, :3]], axis=1)
    g.rotations = torch.tensor(new_wxyz, dtype=torch.float32, device=device)
    for attr in ("forward_vel", "backward_vel"):           # 速度只旋转+缩放, 不平移
        v = getattr(g, attr, None)
        if v is not None:
            setattr(g, attr, s * (v.float() @ R.T))
    return g


def apply_mask_(g, keep):
    # 就地按布尔掩码保留高斯的每个 per-point 张量属性
    n = g.means.shape[0]
    for k, v in list(g.__dict__.items()):
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n:
            setattr(g, k, v[keep])
    return g


def cull_points_(g, center_xy, radius, z_min, z_max):
    # 就地剔除: XY 超出半径 或 Z 超出 [z_min,z_max] 的点
    # (地面已钉 z=0, 借此砍掉地下垃圾 / 过高的远端卷曲)
    n = g.means.shape[0]
    if n == 0:
        return g
    m = g.means.float()
    keep = (m[:, 2] >= z_min) & (m[:, 2] <= z_max)
    if radius > 0:
        c = torch.tensor(center_xy, dtype=torch.float32, device=m.device)
        keep &= torch.linalg.norm(m[:, :2] - c, dim=1) <= radius
    return g if bool(keep.all()) else apply_mask_(g, keep)


def voxel_keys(means, voxel_size):
    # 把坐标量化成体素整数哈希(用于跨时刻去重: 同一体素只留一份, 消叠层 smear)
    v = torch.floor(means.float() / voxel_size).to(torch.int64) + (1 << 19)
    return (v[:, 0] << 40) + (v[:, 1] << 20) + v[:, 2]


# ply 导出
def dump_gaussians_ply(gaussians, output_path, max_points):
    """把累积世界导出标准 3DGS .ply, 丢进 SuperSplat 检查。"""
    gaussians = [g for g in gaussians if g.means.shape[0] > 0]
    total = sum(g.means.shape[0] for g in gaussians)
    if total == 0:
        print("No Gauss for output")
        return
    keep = min(1.0, max_points / total)
    ms, sc, ro, dc, op = [], [], [], [], []
    for g in gaussians:
        n = g.means.shape[0]
        idx = torch.randperm(n, device=g.means.device)[:max(1, int(n * keep))] if keep < 1.0 else slice(None)
        ms.append(g.means[idx].float())
        sc.append(g.scales[idx].float())
        ro.append(g.rotations[idx].float())
        dc.append(g.harmonics[idx, 0, :].float())
        op.append(g.opacities[idx].float())
    means, scales, rot = torch.cat(ms), torch.cat(sc), torch.cat(ro)
    rgbs = torch.cat(dc)
    opac01 = torch.cat(op)
    # 清理 NaN/Inf(个别时刻 Umeyama 退化会产生), 否则查看器可能整片不显示
    finite = torch.isfinite(means).all(1) & torch.isfinite(scales).all(1)
    means, scales, rot, rgbs, opac01 = means[finite], scales[finite], rot[finite], rgbs[finite], opac01[finite]
    # 关键修复: 高斯在 nuScenes 全局系下离原点可达数百~上千米, 查看器默认相机在原点会"看不到";
    # 平移到自身中位数中心(只影响 ply 观看, 不影响视频渲染)
    center = means.median(0).values
    means = means - center
    lo, hi = means.min(0).values.tolist(), means.max(0).values.tolist()
    print(f"    PLY 已重定心(原中心≈{np.round(center.cpu().numpy(), 1)} m), 包围盒 "
          f"x[{lo[0]:.0f},{hi[0]:.0f}] y[{lo[1]:.0f},{hi[1]:.0f}] z[{lo[2]:.0f},{hi[2]:.0f}] m")
    opac = torch.logit(opac01.clamp(1e-4, 1 - 1e-4))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_gs_ply(Path(output_path), means, scales, rot, rgbs, opac)
    print(f"已保存 PLY（{means.shape[0]:,} / {total:,} 个高斯）-> {output_path}")


# 单个时刻: 取 6 路真值
def load_timestep(nusc, sample, dataroot, res):
    # 取一个 sample 的 6 路: 图像 + 修正后 K + c2w_ref(prior用) + c2w_global(Umeyama用)
    # 另返回该时刻自车在全局系的位置/朝向(渲染前进轨迹用)
    sd_ref = nusc.get("sample_data", sample["data"][REF_CAM])
    ego_ref = nusc.get("ego_pose", sd_ref["ego_pose_token"])
    T_global_egoref = pose_to_matrix(ego_ref["translation"], ego_ref["rotation"])
    T_ref_global = np.linalg.inv(T_global_egoref)          # 全局 -> 本时刻参考车身系

    images, K_list, c2w_ref_list, c2w_glb_list = [], [], [], []
    for cam in RING_ORDER:
        sd = nusc.get("sample_data", sample["data"][cam])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])

        pil = Image.open(os.path.join(dataroot, sd["filename"])).convert("RGB")
        src_wh = pil.size                                  # (1600, 900)
        images.append(center_crop(pil, res))
        K_list.append(adjust_K_for_center_crop(np.array(cs["camera_intrinsic"], np.float64), src_wh, res))

        T_ego_cam = pose_to_matrix(cs["translation"], cs["rotation"])
        T_global_ego = pose_to_matrix(ep["translation"], ep["rotation"])
        c2w_global = T_global_ego @ T_ego_cam              # 相机 -> 全局(米制, Umeyama 目标)
        c2w_ref_list.append(T_ref_global @ c2w_global)     # 相机 -> 本时刻参考系(prior, 数值小)
        c2w_glb_list.append(c2w_global)

    ego_pos = T_global_egoref[:3, 3].astype(np.float32)            # 自车全局位置
    ego_head = normalize(T_global_egoref[:3, 0].astype(np.float32))  # 自车前向(ego +x)在全局
    return images, K_list, c2w_ref_list, c2w_glb_list, ego_pos, ego_head


# main
@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda"
    torch_dtype = torch.bfloat16
    res = (args.width, args.height)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)       # 全局系 z 朝上

    # Phase 0: 准备 nuScenes, 收集 T 个连续关键帧的 sample
    print("Phase 0: 加载 nuScenes 并收集关键帧序列")
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as e:
        raise ImportError("python -m pip install nuscenes-devkit (用当前解释器自己的 pip)") from e
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    scene = nusc.scene[args.scene_index]
    token = scene["first_sample_token"]
    for _ in range(args.start_offset):                     # 跳到起始关键帧
        s = nusc.get("sample", token)
        if not s["next"]:
            break
        token = s["next"]
    samples = []
    for _ in range(args.num_timesteps):                    # 按 time_stride 取 T 帧
        cur = nusc.get("sample", token)
        samples.append(cur)
        for _ in range(args.time_stride):                  # 按 stride 向后前进
            if cur["next"]:
                cur = nusc.get("sample", cur["next"])
        token = cur["token"]
        if not cur["next"]:                                # 片段到头, 提前结束
            break
    T = len(samples)
    print(f"    scene[{args.scene_index}]={scene['name']}, 取到 {T} 个关键帧")

    # Phase 1: 加载重建器(只加载一次)
    print("Phase 1: 加载重建器")
    if not os.path.isfile(args.reconstructor_path):
        raise FileNotFoundError(f"找不到重建器权重: {args.reconstructor_path}")
    mm = ModelManager()
    mm.load_model(args.reconstructor_path, device=device, torch_dtype=torch_dtype)
    reconstructor = mm.fetch_model("reconstructor")
    S = len(RING_ORDER)

    # Phase 2: 逐时刻重建 -> Umeyama 对齐到米制全局系 -> 累积
    print("Phase 2: 逐时刻重建并累积进米制全局世界")
    accum_per_step = []          # accum_per_step[t] = 第 t 时刻搬好的高斯 list
    ego_track = []               # 每个时刻的 (全局位置, 前向)
    K_ref = None                 # 渲染用内参(取第一帧网络输出)
    last_good_s = None           # 地面拟合失败时回退用的上一帧尺度
    occupied = torch.empty(0, dtype=torch.int64, device=device)   # 已占据体素(跨时刻去重)
    for t, sample in enumerate(samples):
        images, K_list, c2w_ref_list, c2w_glb_list, ego_pos, ego_head = \
            load_timestep(nusc, sample, args.dataroot, res)
        ego_track.append((ego_pos, ego_head))

        views = {
            "img": torch.stack([TF.to_tensor(im)[None] for im in images], dim=1).to(device),
            "is_target": torch.zeros((1, S), dtype=torch.bool, device=device),
            "is_static": torch.ones((1, S), dtype=torch.bool, device=device),
            "timestamp": torch.zeros((1, S), dtype=torch.int64, device=device),
        }
        cond_flags = [0, 0, 0]
        if args.use_gt:                                    # 注入 GT 内外参(prior 用本时刻参考系 c2w)
            views["camera_poses"] = torch.tensor(np.stack(c2w_ref_list), dtype=torch.float32, device=device)[None]
            views["camera_intrs"] = torch.tensor(np.stack(K_list), dtype=torch.float32, device=device)[None]
            cond_flags = [0, args.use_gt_intrinsics, args.use_gt_extrinsics]

        with torch.amp.autocast("cuda", dtype=torch_dtype):
            pred = reconstructor(views, cond_flags=cond_flags, is_inference=True, use_motion=False)
        if K_ref is None:
            K_ref = pred["rendered_intrinsics"][0, 0].float()

        # === 摆放: 朝向取 GT, 尺度取地面, 平移对齐 GT 全局(不再用预测中心估尺度) ===
        pred_c2w = pred["rendered_extrinsics"][0].float().cpu().numpy()        # [6,4,4] 网络系
        gt_c2w = np.stack(c2w_glb_list)                                        # [6,4,4] GT 全局
        net_ctr, gt_ctr = pred_c2w[:, :3, 3], gt_c2w[:, :3, 3]                 # 各自相机中心
        # 旋转: 每路 R_glb·R_net^T 的平均(用满 6 个朝向 -> 不受相机共面影响, 稳)
        Rs = [gt_c2w[i, :3, :3] @ pred_c2w[i, :3, :3].T for i in range(S)]
        R = average_rotation(Rs)
        maxdev = max(float(np.degrees(np.arccos(np.clip((np.trace(R.T @ Ri) - 1) / 2, -1, 1)))) for Ri in Rs)
        # 尺度+残余整平: 地面 RANSAC, 已知相机离地高度 = GT 相机平均高度
        sub = np.concatenate([g.means.float().detach().cpu().numpy()
                              for g in pred["splats"][0] if g.means.shape[0] > 0], axis=0)
        if sub.shape[0] > 60000:
            sub = sub[np.random.choice(sub.shape[0], 60000, replace=False)]
        s, R_lev = level_and_scale(sub, R, R @ net_ctr.mean(0), float(gt_ctr[:, 2].mean()),
                                   thresh_frac=args.ground_thresh_frac)
        if s is None or s <= 0:                       # 地面拟合失败 -> 沿用上一帧尺度
            s = last_good_s if last_good_s else 1.0
        else:
            last_good_s = s
        R_total = (R_lev @ R).astype(np.float32)
        tt = (gt_ctr.mean(0) - s * (R_total @ net_ctr.mean(0))).astype(np.float32)  # 相机质心对齐 GT

        # 搬运 -> 剔除(XY 半径 & Z 范围) -> 跨时刻体素去重(只留新体素 = 迷雾揭示, 消叠层)
        step_g = []
        for g in pred["splats"][0]:
            if g.means.shape[0] == 0:
                continue
            transform_gaussians_(g, s, R_total, tt, device)
            cull_points_(g, (float(ego_pos[0]), float(ego_pos[1])), args.max_range, args.z_min, args.z_max)
            if args.voxel_size > 0 and g.means.shape[0] > 0:
                keys = voxel_keys(g.means, args.voxel_size)
                new = ~torch.isin(keys, occupied) if occupied.numel() else torch.ones_like(keys, dtype=torch.bool)
                apply_mask_(g, new)
                occupied = torch.cat([occupied, keys[new]])
            if g.means.shape[0] > 0:
                step_g.append(g)
        accum_per_step.append(step_g)
        print(f"    t={t:02d} scale={s:.3f} 旋转一致性<{maxdev:.1f}° 本帧 {sum(x.means.shape[0] for x in step_g):,} 点")
        del pred
        torch.cuda.empty_cache()

    # 可选: 导出"全程累积"的世界 PLY
    all_g = [g for step in accum_per_step for g in step]
    if args.output_ply:
        dump_gaussians_ply(all_g, args.output_ply, args.ply_max_points)

    # Phase 3: 自车视角前进渲染 -> 只画"累积到当前时刻"的高斯 = 迷雾随前进散开
    if args.no_video:
        print("完成(跳过视频)。")
        return
    print("Phase 3: 渲染自车视角前进视频")
    rasterizer = reconstructor.gs_renderer.rasterizer
    ts = torch.zeros(1, device=device)                     # 静态查询时刻

    # 前进朝向: motion=相邻时刻真实位移(与高斯所在全局系严格一致, 推荐, 修问题1);
    #          ego=车身姿态轴(ego +x)。位移过小(近乎静止)时退回车身朝向。
    positions = [p for (p, _) in ego_track]
    orients = [o for (_, o) in ego_track]
    def heading_at(t):
        if args.heading_source == "ego":
            return orients[t]
        if t + 1 < T:
            d = positions[t + 1] - positions[t]
        elif T > 1:
            d = positions[t] - positions[t - 1]
        else:
            return orients[t]
        return normalize(d) if np.linalg.norm(d) > 0.3 else orients[t]
    headings = [heading_at(t) for t in range(T)]

    frames = []
    accum = []                                             # 运行时累积(到当前时刻为止)
    for t in range(T):
        accum.extend(accum_per_step[t])                    # 揭示: 加入本时刻新高斯
        if len(accum) == 0:
            continue
        pos_t, head_t = positions[t], headings[t]
        pos_n, head_n = positions[min(t + 1, T - 1)], headings[min(t + 1, T - 1)]  # 下一时刻, 插值前进
        # 默认抬高第三人称跟拍(相机在壳外俯看 -> 不会钻进 2.5D 壳里被埋);
        # --first_person 改回车内视角(注意: v2 扩散补洞前会有空洞)
        cam_h, back, look_h = (1.6, 0.0, 1.0) if args.first_person \
            else (args.cam_height, args.back_off, args.look_height)
        for k in range(args.frames_per_step):              # 帧间插值, 让"前进"顺滑
            a = k / args.frames_per_step
            pos = pos_t * (1 - a) + pos_n * a
            head = normalize(head_t * (1 - a) + head_n * a)
            eye = pos + up * cam_h - head * back            # 后退 + 抬高
            target = pos + head * args.look_ahead + up * look_h   # 看向前下方
            c2w = look_at_opencv(eye, target, up)
            w2c = homo_matrix_inverse(torch.tensor(c2w, device=device))[None]
            rgb, _, _ = rasterizer.forward(
                [list(accum)], render_viewmats=[w2c], render_Ks=[K_ref[None]],
                render_timestamps=[ts], sh_degree=0, width=args.width, height=args.height)
            frames.append((rgb[0, 0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy())

    os.makedirs(os.path.dirname(args.output_video) or ".", exist_ok=True)
    save_video([np.asarray(f) for f in frames], args.output_video, fps=args.fps)
    print(f"已保存自车前进视频（{len(frames)} 帧）-> {args.output_video}")
    print("完成。")


def parse_args():
    p = argparse.ArgumentParser(description="v1: 沿关键帧累积米制高斯世界 + 自车视角前进")
    # 数据
    p.add_argument("--dataroot", default="/mnt/ssd1/jzl/datasets/nuscenes")
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--scene_index", type=int, default=0)
    p.add_argument("--start_offset", type=int, default=0, help="起始关键帧偏移")
    p.add_argument("--num_timesteps", type=int, default=20, help="累积多少个关键帧(开多远; scene 约 40 帧)")
    p.add_argument("--time_stride", type=int, default=1, help="关键帧间隔(>1 跳着取, 走更快)")
    # 模型
    p.add_argument("--reconstructor_path", default="/mnt/ssd1/wzq_models/NeoVerse/reconstructor.ckpt")
    p.add_argument("--width", type=int, default=560)
    p.add_argument("--height", type=int, default=336)
    # GT 约束(同 v0)
    p.add_argument("--use_gt", action="store_true", help="注入 GT 内外参 prior(提升单帧内连续性)")
    p.add_argument("--use_gt_extrinsics", type=int, default=1)
    p.add_argument("--use_gt_intrinsics", type=int, default=1)
    # 累积控制
    p.add_argument("--max_range", type=float, default=25.0, help="每帧只累积自车 XY 半径内的点(米, 0=不裁)")
    p.add_argument("--z_min", type=float, default=-0.5, help="剔除低于此高度的点(地面已钉 z=0, 砍地下垃圾)")
    p.add_argument("--z_max", type=float, default=12.0, help="剔除高于此高度的点(砍远端卷曲/天空)")
    p.add_argument("--voxel_size", type=float, default=0.15, help="跨时刻体素去重边长[m](0=关闭)")
    p.add_argument("--ground_thresh_frac", type=float, default=0.02, help="地面 RANSAC 内点阈值(占场景尺度比例)")
    # 输出: PLY
    p.add_argument("--output_ply", default="outputs/v1_fog_world.ply")
    p.add_argument("--ply_max_points", type=int, default=3_000_000)
    # 输出: 自车前进视频
    p.add_argument("--output_video", default="outputs/v1_ego_forward.mp4")
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--heading_source", choices=["motion", "ego"], default="motion",
                   help="前进朝向来源: motion=真实位移(推荐); ego=车身姿态轴(用于对比排查)")
    p.add_argument("--frames_per_step", type=int, default=10, help="相邻关键帧之间插几帧(越大越慢越顺滑)")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--first_person", action="store_true",
                   help="车内第一人称视角(默认是抬高第三人称跟拍, 避免钻进壳被埋)")
    p.add_argument("--cam_height", type=float, default=4.0, help="跟拍相机离地高度[m]")
    p.add_argument("--back_off", type=float, default=8.0, help="跟拍相机在自车后方距离[m]")
    p.add_argument("--look_ahead", type=float, default=10.0, help="注视点在前方距离[m]")
    p.add_argument("--look_height", type=float, default=1.0, help="注视点离地高度[m]")
    return p.parse_args()


if __name__ == "__main__":
    main()
