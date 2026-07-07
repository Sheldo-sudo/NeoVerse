"""
核心要点:
nuScenes 数据集的相机内外参校准 VGGT 的预测
通过真实的相机内外参实现对六个视角的拼接
版本:v0

NeoVerse:
VGGT + DPT 预测头, 也是唯一区别于 VGGT 的地方
复用点: WorldMirror.forward 的 prior 注入

实验：
对比实验, 对于同一组 nuScenes 数据, 对照 VGGT 预测 和 实参
两次都导出 .ply, 丢进 https://superspl.at/editor 肉眼对比缝隙；
同时各自渲染一段环绕(turntable)视频, 直观看 6 路是否连成一片。

坐标系约定：
- nuScenes 相机系 = OpenCV 光学系(x 右, y 下, z 前)，与 WorldMirror 一致，无需翻轴。
- nuScenes 车身(ego)系 = x 前, y 左, z 上。
- calibrated_sensor 给的是 T_ego←cam(相机在车身系下的位姿)= 直接就是 c2w(世界取车身)。
- 一个 sample 的 6 路相机时间戳略有差异(各自的 ego_pose)，所以严谨地把每路先映射到
  全局系再统一搬到"参考车身系"(取 CAM_FRONT 的 ego_pose 作参考)。

CLI参数约定:
# VGGT/NeoVerse 预测(基线)
python fog_world_v0_nuscenes.py --scene_index 0 \
        --output_ply outputs/v0_baseline.ply --output_video outputs/v0_baseline.mp4
# GT 强约束
python fog_world_v0_nuscenes.py --scene_index 0 --use_gt \
        --output_ply outputs/v0_gtcam.ply --output_video outputs/v0_gtcam.mp4
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


# ============================ 几何变换 ============================
def pose_to_matrix(translation, rotation):
    """nuScenes 的 (平移[x,y,z], 四元数[w,x,y,z]) -> 4x4 齐次矩阵。

    注意: nuScenes 存的四元数顺序是 [w,x,y,z]，而 scipy.from_quat 要 [x,y,z,w]。
    """
    w, x, y, z = rotation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()   # scipy 用 xyzw
    T[:3, 3] = translation
    return T


def adjust_K_for_center_crop(K, src_wh, dst_wh):
    """把内参 K 按 center_crop(先等比放大覆盖, 再中心裁剪)同步修正。

    必须和 diffsynth.utils.auxiliary.center_crop 的像素变换严格一致，否则喂进网络的
    GT 内参与实际图像不对齐，约束就成了噪声。fx/fy/cx/cy 都要先乘放大倍数 s，cx/cy 再减裁剪偏移。
    """
    W, H = src_wh
    tw, th = dst_wh
    s = max(tw / W, th / H)               # 等比放大倍数(覆盖目标框)
    ow, oh = int(W * s), int(H * s)       # 放大后尺寸(与 center_crop 的 int() 对齐)
    left = (ow - tw) // 2                  # 中心裁剪的左/上偏移
    top = (oh - th) // 2
    K2 = K.astype(np.float64).copy()
    K2[0, 0] *= s                          # fx
    K2[1, 1] *= s                          # fy
    K2[0, 2] = K[0, 2] * s - left          # cx (先缩放再减偏移)
    K2[1, 2] = K[1, 2] * s - top           # cy
    return K2


def normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v) + eps)


def look_at_opencv(eye, target, up):
    """OpenCV 约定(x 右, y 下, z 前)的 look-at，返回 c2w 4x4。"""
    z = normalize(target - eye)
    x = normalize(np.cross(z, up))
    y = np.cross(z, x)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = x, y, z
    c2w[:3, 3] = eye
    return c2w


# ============================ ply 导出 ============================
def dump_gaussians_ply(gaussians, output_path, max_points):
    """将一组 Gaussians 导出为标准 3DGS .ply，丢进 SuperSplat 检查连续性。

    取每个高斯自身基准时刻的快照；下采样到 max_points 控文件大小。
    复用 save_gs_ply：它内部把 scales->log、opacity[0,1]->logit、harmonics DC->f_dc。
    """
    gaussians = [g for g in gaussians if g.means.shape[0] > 0]
    total = sum(g.means.shape[0] for g in gaussians)
    if total == 0:
        print("No Gauss for output")
        return
    keep = min(1.0, max_points / total)
    ms, sc, ro, dc, op = [], [], [], [], []
    for g in gaussians:
        n = g.means.shape[0]
        if keep < 1.0:
            idx = torch.randperm(n, device=g.means.device)[:max(1, int(n * keep))]
        else:
            idx = slice(None)
        ms.append(g.means[idx].float())
        sc.append(g.scales[idx].float())
        ro.append(g.rotations[idx].float())
        dc.append(g.harmonics[idx, 0, :].float())
        op.append(g.opacities[idx].float())
    means, scales, rot = torch.cat(ms), torch.cat(sc), torch.cat(ro)
    rgbs = torch.cat(dc)
    opac = torch.logit(torch.cat(op).clamp(1e-4, 1 - 1e-4))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_gs_ply(Path(output_path), means, scales, rot, rgbs, opac)
    print(f"已保存 PLY（{means.shape[0]:,} / {total:,} 个高斯）-> {output_path}")


# ====================== 视频渲染(环绕 turntable) ======================
def render_turntable(reconstructor, gaussians, K_ref, args):
    """绕场景中心转一圈渲染，输出一段视频，直观看 6 路是否连成一片。

    关键点：use_gt 注入 prior 后高斯落在网络归一化系(非米制)，所以相机的位置/尺度
    不能写死，必须从高斯点云自身统计出来：
      - 中心 = 点云中位数；
      - 用 PCA 取方差最小的方向当作"上"(地面法线)，另两个主轴张成环绕平面；
      - 半径/高度按点云尺度(分位距)成比例。
    这样无论归一化系还是米制系、无论 baseline 还是 use_gt 都能渲出合理的环绕。
    """
    rasterizer = reconstructor.gs_renderer.rasterizer
    device = gaussians[0].means.device

    # 1) 点云统计：中心 + 尺度 + 主轴(PCA)
    pts = torch.cat([g.means.float() for g in gaussians], dim=0)        # [Ntot,3]
    center = pts.median(dim=0).values                                  # 鲁棒中心
    lo = torch.quantile(pts, 0.1, dim=0)
    hi = torch.quantile(pts, 0.9, dim=0)
    extent = float(torch.linalg.norm(hi - lo))                         # 场景大致尺度
    X = pts - pts.mean(dim=0, keepdim=True)
    cov = (X.T @ X) / max(1, X.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov.double())                     # 升序特征值
    up = evecs[:, 0].float().cpu().numpy()                             # 最小方差方向≈地面法线
    e1 = evecs[:, 2].float().cpu().numpy()                             # 最大方差方向(平面内)
    e2 = evecs[:, 1].float().cpu().numpy()                             # 次大方差方向(平面内)
    if args.flip_up:
        up = -up
    c = center.cpu().numpy()
    radius = extent * args.orbit_radius_scale
    height = extent * args.orbit_height_scale

    # 2) 逐帧绕圈渲染
    N = args.num_frames
    ts = torch.zeros(1, device=device)            # 静态场景：查询时刻取 0 即可
    frames = []
    for i in range(N):
        ang = 2.0 * np.pi * i / N
        eye = c + radius * (np.cos(ang) * e1 + np.sin(ang) * e2) + height * up
        c2w = look_at_opencv(eye.astype(np.float32), c.astype(np.float32), up.astype(np.float32))
        w2c = homo_matrix_inverse(torch.tensor(c2w, device=device))[None]   # [1,4,4]
        rgb, _, _ = rasterizer.forward(
            [list(gaussians)], render_viewmats=[w2c], render_Ks=[K_ref[None]],
            render_timestamps=[ts], sh_degree=0, width=args.width, height=args.height)
        frames.append((rgb[0, 0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy())

    os.makedirs(os.path.dirname(args.output_video) or ".", exist_ok=True)
    save_video([np.asarray(f) for f in frames], args.output_video, fps=args.fps)
    print(f"已保存环绕视频（{N} 帧）-> {args.output_video}")


# ============================ main ============================
@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda"
    torch_dtype = torch.bfloat16
    res = (args.width, args.height)

    # Phase 0：从 nuScenes 取一帧(一个 sample)的 6 路真图 + 真值内外参
    # 读图 -> center_crop 到网络分辨率
    # 取每路 GT 内参 K（并同步修正到裁剪后的分辨率）
    # 取每路 GT 外参 c2w（统一到 CAM_FRONT 的参考车身系）
    print("Phase 0:加载nuScenes真值")
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as e:
        raise ImportError(
            "当前 Python 环境没装 nuScenes devkit。请用『当前解释器自己的 pip』安装："
            "python -m pip install nuscenes-devkit"
        ) from e

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    scene = nusc.scene[args.scene_index]
    sample_token = scene["first_sample_token"]
    for _ in range(args.sample_offset):     # 沿时间向后跳若干帧
        sample = nusc.get("sample", sample_token)
        if not sample["next"]:
            break
        sample_token = sample["next"]
    sample = nusc.get("sample", sample_token)
    print(f"    scene[{args.scene_index}] = {scene['name']}, sample = {sample_token}")

    # 参考车身系：用 CAM_FRONT 这一路 sample_data 的 ego_pose
    sd_ref = nusc.get("sample_data", sample["data"][REF_CAM])
    ego_ref = nusc.get("ego_pose", sd_ref["ego_pose_token"])
    T_global_egoref = pose_to_matrix(ego_ref["translation"], ego_ref["rotation"])
    T_ref_global = np.linalg.inv(T_global_egoref)    # 全局 -> 参考车身系

    images, K_list, c2w_list = [], [], []
    for cam in RING_ORDER:
        sd = nusc.get("sample_data", sample["data"][cam])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])   # 相机->车身
        ep = nusc.get("ego_pose", sd["ego_pose_token"])                     # 车身->全局(该路自己的时刻)

        # 图像：原图 1600x900 -> center_crop 到 (W,H) ---
        img_path = os.path.join(args.dataroot, sd["filename"])
        pil = Image.open(img_path).convert("RGB")
        src_wh = pil.size                       # (1600, 900)
        images.append(center_crop(pil, res))

        # 内参：随 center_crop 同步修正 ---
        K_full = np.array(cs["camera_intrinsic"], dtype=np.float64)
        K_list.append(adjust_K_for_center_crop(K_full, src_wh, res))

        # 外参：相机 -> 全局 -> 参考车身系，得到 c2w（OpenCV，世界=参考车身系）---
        T_ego_cam = pose_to_matrix(cs["translation"], cs["rotation"])
        T_global_ego = pose_to_matrix(ep["translation"], ep["rotation"])
        c2w_ref = T_ref_global @ T_global_ego @ T_ego_cam
        c2w_list.append(c2w_ref)
        # 自检：CAM_FRONT 的 c2w 平移应≈出厂安装位(约 x前1.7, z上1.5)
        print(f"    {cam:16s} c2w 平移 = {np.round(c2w_ref[:3, 3], 3)}")

    # Phase 0b：加载语义地图，仅验证可用性，真正的地面/路面约束放到 v1
    if args.load_map:
        try:
            from nuscenes.map_expansion.map_api import NuScenesMap
            location = nusc.get("log", scene["log_token"])["location"]
            nmap = NuScenesMap(dataroot=args.dataroot, map_name=location)
            ego_xy = T_global_egoref[:2, 3]
            on_road = nmap.layers_on_point(ego_xy[0], ego_xy[1]).get("drivable_area", "")
            print(f"    地图 location={location}，自车{'在' if on_road else '不在'}可行驶区域 "
                  f"(v0 暂不融合地图，留给 v1 做地面 z=0 约束)")
        except Exception as e:    # 地图扩展未安装/缺文件时不阻断主流程
            print(f"    地图加载跳过：{e}")

    # Phase 1：加载 NeoVerse 重建器(WorldMirror)，构造 views 做一次静态重建
    # 关键开关 use_gt：决定是否把 GT 内外参当 prior 喂进去
    print("Phase 1: 加载重建器并前馈")
    if not os.path.isfile(args.reconstructor_path):
        raise FileNotFoundError(f"找不到重建器权重: {args.reconstructor_path}")
    mm = ModelManager()
    mm.load_model(args.reconstructor_path, device=device, torch_dtype=torch_dtype)
    reconstructor = mm.fetch_model("reconstructor")     # 就是一个 WorldMirror 实例

    S = len(RING_ORDER)
    # 6 路当作"同一静态场景的多视角"：is_static=True, 时间戳全 0, 都是源视角(非 target)
    views = {
        "img": torch.stack([TF.to_tensor(im)[None] for im in images], dim=1).to(device),  # [1,6,3,H,W]
        "is_target": torch.zeros((1, S), dtype=torch.bool, device=device),
        "is_static": torch.ones((1, S), dtype=torch.bool, device=device),
        "timestamp": torch.zeros((1, S), dtype=torch.int64, device=device),
    }

    # cond_flags = [depth, rays(内参), camera(外参)]；按开关决定注入哪些 GT
    cond_flags = [0, 0, 0]
    if args.use_gt:
        # 外参 prior：c2w(OpenCV)，世界取参考车身系（6 路共享同一世界系 => 强制相互对齐）
        views["camera_poses"] = torch.tensor(
            np.stack(c2w_list), dtype=torch.float32, device=device)[None]      # [1,6,4,4]
        # 内参 prior：修正到裁剪分辨率后的 K
        views["camera_intrs"] = torch.tensor(
            np.stack(K_list), dtype=torch.float32, device=device)[None]        # [1,6,3,3]
        cond_flags = [0, args.use_gt_intrinsics, args.use_gt_extrinsics]
        print(f"    使用 GT 约束: cond_flags={cond_flags} "
              f"(rays/内参={bool(cond_flags[1])}, camera/外参={bool(cond_flags[2])})")
    else:
        print("    基线模式: 不注入任何 GT，纯网络预测(复现不连续)")

    with torch.amp.autocast("cuda", dtype=torch_dtype):
        pred = reconstructor(views, cond_flags=cond_flags,
                             is_inference=True, use_motion=False)

    # Phase 2：导出融合后的高斯 .ply，丢进 SuperSplat 肉眼对比"连续性"
    # 6 路在 splats[0] 里（每路一组高斯），直接全部导出
    # 对比 baseline.ply 与 gtcam.ply：缝隙/飘开是否缓解
    print("Phase 2: 导出 PLY")
    gaussians = list(pred["splats"][0])
    n_groups = len(gaussians)
    n_pts = sum(g.means.shape[0] for g in gaussians)
    print(f"    得到 {n_groups} 组高斯，共 {n_pts:,} 个点")
    dump_gaussians_ply(gaussians, args.output_ply, args.ply_max_points)

    # Phase 3：渲染环绕(turntable)视频，直观看连续性（不需要 SuperSplat）
    if not args.no_video:
        print("Phase 3: 渲染环绕视频")
        gaussians_nonempty = [g for g in gaussians if g.means.shape[0] > 0]
        if len(gaussians_nonempty) == 0:
            print("    无高斯可渲染，跳过视频")
        else:
            # 渲染用内参：直接复用网络这次预测出的内参(对应当前网络分辨率)
            K_ref = pred["rendered_intrinsics"][0, 0].float()
            render_turntable(reconstructor, gaussians_nonempty, K_ref, args)

    print("完成。")


def parse_args():
    p = argparse.ArgumentParser(
        description="v0: 用 nuScenes 真值内外参强约束 NeoVerse 重建器，测试是否缓解环视不连续")
    # 数据
    p.add_argument("--dataroot", default="/mnt/ssd1/jzl/datasets/nuscenes",
                   help="nuScenes 数据集根目录")
    p.add_argument("--version", default="v1.0-trainval",
                   help="nuScenes 版本 (v1.0-trainval / v1.0-mini)")
    p.add_argument("--scene_index", type=int, default=0, help="第几个 scene")
    p.add_argument("--sample_offset", type=int, default=0,
                   help="从 scene 首帧向后跳多少个关键帧(取哪一时刻的环视)")
    p.add_argument("--load_map", action="store_true",
                   help="顺带加载语义地图验证可用性(v0 不融合, 仅打印)")
    # 模型
    p.add_argument("--reconstructor_path",
                   default="/mnt/ssd1/wzq_models/NeoVerse/reconstructor.ckpt")
    # 分辨率(需能被 patch_size=14 整除：560/14=40, 336/14=24)
    p.add_argument("--width", type=int, default=560)
    p.add_argument("--height", type=int, default=336)
    # GT 约束开关
    p.add_argument("--use_gt", action="store_true",
                   help="开启 -> 把 GT 相机内外参当 prior 喂网络；关闭 -> 纯预测基线")
    p.add_argument("--use_gt_extrinsics", type=int, default=1,
                   help="use_gt 下是否注入外参(c2w)，1/0")
    p.add_argument("--use_gt_intrinsics", type=int, default=1,
                   help="use_gt 下是否注入内参(K)，1/0")
    # 输出：PLY
    p.add_argument("--output_ply", default="outputs/v0_surround.ply")
    p.add_argument("--ply_max_points", type=int, default=2_000_000)
    # 输出：视频(环绕)
    p.add_argument("--output_video", default="outputs/v0_surround.mp4")
    p.add_argument("--no_video", action="store_true", help="跳过视频渲染，只出 PLY")
    p.add_argument("--num_frames", type=int, default=120, help="环绕视频帧数")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--orbit_radius_scale", type=float, default=1.2,
                   help="环绕半径 = 点云尺度 * 该系数")
    p.add_argument("--orbit_height_scale", type=float, default=0.3,
                   help="相机抬高 = 点云尺度 * 该系数(俯视程度)")
    p.add_argument("--flip_up", action="store_true",
                   help="若环绕视角上下颠倒，翻转 PCA 估计的'上'方向")
    return p.parse_args()


if __name__ == "__main__":
    main()
