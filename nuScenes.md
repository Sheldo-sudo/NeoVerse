# nuScenes 数据集原理与用法（结合本项目）

> 面向 NeoVerse「战争迷雾」渐进高斯世界实验，记录 nuScenes 的数据库原理、坐标变换链、
> 目录结构，以及在远程服务器（数据集在 `/mnt/ssd1/jzl/datasets/nuscenes`）上的用法。

---

## 0. 新建独立工作目录

见同目录的 `setup_fogworld.sh`：把 NeoVerse 的 `diffsynth` 包（v0 唯一 import 依赖）
软链到一个与 NeoVerse 同级的新目录 `FogWorld`，再放入 v0 脚本，互不污染。

```bash
bash setup_fogworld.sh
# 跑通后第一组 A/B 实验：
cd /home/beihang/wzq/FogWorld
python fog_world_v0_nuscenes.py --version v1.0-mini --scene_index 0 \
    --output_ply outputs/v0_baseline.ply           # 基线：纯预测，复现不连续
python fog_world_v0_nuscenes.py --version v1.0-mini --scene_index 0 --use_gt \
    --output_ply outputs/v0_gtcam.ply              # GT 强约束：看缝隙是否缓解
```

---

## 1. 它是什么

nuScenes：1000 个驾驶片段（scene），每段 ~20 秒，采集于波士顿 + 新加坡。每辆车装
**6 路环视相机（1600×900）+ 1 激光雷达 + 5 毫米波雷达**。标注以 **2Hz 关键帧**给出
（3D 框、属性、速度）。

- `v1.0-trainval` = 850 个 scene（700 train + 150 val）
- `v1.0-test` = 无标注
- `v1.0-mini` = 10 个 scene 的子集（开发调试用，秒级加载）

---

## 2. 核心原理：它是一个「关系数据库」，不是一堆文件

`v1.0-*/` 目录里是一组 **JSON 表**，靠 `token`（唯一 ID）互相外键链接。你几乎不直接读
文件名，而是通过 `nuscenes-devkit` 顺着 token 图取数据。和本项目最相关的表：

| 表 | 作用 | 关键字段 |
|---|---|---|
| `scene` | 一个 20s 片段 | `first_sample_token`, `log_token` |
| `sample` | 一个**关键帧时刻**(2Hz) | `data{CAM_FRONT:..., ...}`, `next/prev`, `anns` |
| `sample_data` | 某传感器某一帧 | `filename`(图路径), `ego_pose_token`, `calibrated_sensor_token`, `is_key_frame` |
| `ego_pose` | **车身在全局系的位姿**(每帧) | `translation`, `rotation`(wxyz) |
| `calibrated_sensor` | **传感器相对车身的安装**(每 scene 固定) | `translation`, `rotation`, `camera_intrinsic`(3×3 K) |
| `sample_annotation` | 一个 3D 框 | `translation/size/rotation`, `instance_token`；速度用 `nusc.box_velocity()` |
| `log` | scene 所属采集日志 | `location`(决定用哪张地图) |

**这正是我们「用真值强约束」的来源**：`calibrated_sensor` 给真实内外参，`ego_pose`
给真实车身轨迹。

---

## 3. 最关键的一环：坐标变换链

任何点 / 相机都靠这条链在三个系之间转换（v0/v1 全靠它）：

```
全局系(global) ──ego_pose──► 车身系(ego) ──calibrated_sensor──► 传感器系(camera, OpenCV RDF)
```

- `calibrated_sensor` = `T_ego←cam`（相机在车身系下的位姿）→ 对单帧来说它**就是 c2w**（世界取车身）。
- `ego_pose` = `T_global←ego`（每帧不同，因为车在动）。
- 相机内参 `camera_intrinsic` 是针对 1600×900 原图的；**裁剪 / 缩放后必须同步改 K**
  （v0 里的 `adjust_K_for_center_crop` 就是干这个）。
- nuScenes 相机系是 **OpenCV(x 右, y 下, z 前)**，与 WorldMirror 完全一致，无需翻轴；
  车身系是 **x 前, y 左, z 上**。
- ⚠️ 一个 `sample` 下 6 路相机**时间戳不完全对齐**（各自有独立的 `ego_pose`），所以严谨
  做法是每路先回全局再统一到参考车身系——v0 已按此处理（以 `CAM_FRONT` 为参考）。

构造矩阵（scipy 约定，四元数顺序 xyzw；nuScenes 存的是 wxyz）：

```python
from scipy.spatial.transform import Rotation
import numpy as np

def pose_to_matrix(translation, quat_wxyz):
    w, x, y, z = quat_wxyz
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = translation
    return T

# 相机 -> 全局 -> 参考车身系
c2w_ref = np.linalg.inv(T_global_egoref) @ T_global_ego @ T_ego_cam
```

---

## 4. 关键帧 vs sweeps（对「战争迷雾」累积很重要）

- `samples/`：**关键帧**(2Hz)，有标注 → 服务器上是 `samples/CAM_FRONT/*.jpg` 等 6 个相机子目录。
- `sweeps/`：关键帧之间的**中间帧**(相机 ~12Hz)，无标注，只有 `sample_data`+位姿 → 给我们
  做密集时间累积、让迷雾揭示更顺滑。
- v0 只用单个 `sample`；v1 做累积时可沿 `sample['next']`（2Hz）或进一步用 sweeps（12Hz）
  串起来，每帧都有 `ego_pose` 做米制对齐。

---

## 5. 地图（本服务器 `maps/` 已完整）

- `maps/expansion/*.json`：**矢量语义地图**（4 个地点：`boston-seaport`, `singapore-onenorth`,
  `singapore-hollandvillage`, `singapore-queenstown`），含 `drivable_area / lane /
  road_segment / ped_crossing` 等图层。
- `maps/basemap/*.png`：栅格底图；顶层 4 个哈希名 `.png` 是旧版语义掩码。
- 用法：

```python
from nuscenes.map_expansion.map_api import NuScenesMap
location = nusc.get('log', scene['log_token'])['location']
nmap = NuScenesMap(dataroot='/mnt/ssd1/jzl/datasets/nuscenes', map_name=location)
nmap.layers_on_point(x_global, y_global)   # 查某全局坐标落在哪些图层
```

- 对我们的价值：地图给**全局系下 z=0 的地面真值平面**和**可行驶区域 mask**——v1 用来定
  地面、约束 ground-fill 只铺真实路面、做迷雾 reveal 范围。v0 仅 `--load_map` 验证可用、
  暂不融合（因为注入 prior 后输出在网络归一化系，还没回到米制全局系）。

---

## 6. 本服务器目录映射 + dataroot

```
/mnt/ssd1/jzl/datasets/nuscenes/      ← 这就是 devkit 的 dataroot
├── samples/   关键帧(2Hz) 6相机+雷达
├── sweeps/    中间帧(~12Hz)
├── maps/      {expansion 矢量, basemap 栅格, prediction}
├── v1.0-mini / v1.0-trainval / v1.0-test   ← JSON 数据库(version 参数选这个)
├── can_bus/   每scene的IMU/姿态/方向盘(高频车辆状态, 可选用于运动先验)
├── depth/     预计算深度 .npy(部分帧, 可当深度 prior 的现成来源)
└── nuscenes -> /mnt/ssd1/nuscenes   (软链, 忽略; dataroot 用顶层即可)
```

所以 v0 默认的 `--dataroot /mnt/ssd1/jzl/datasets/nuscenes` 正确；选库用
`--version v1.0-mini`（调试）或 `v1.0-trainval`（正式）。

---

## 7. 最小验证片段（确认环境通）

```python
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='/mnt/ssd1/jzl/datasets/nuscenes', verbose=True)
s = nusc.get('sample', nusc.scene[0]['first_sample_token'])
sd = nusc.get('sample_data', s['data']['CAM_FRONT'])
print(nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])['camera_intrinsic'])  # 真值 K
print(nusc.get('ego_pose', sd['ego_pose_token'])['translation'])                          # 真值车身位姿
```

---

## 8. 备注：第三种真值约束——预计算深度

`depth/` 里有**预计算深度** `.npy`（命名和图一一对应，如
`..._CAM_FRONT__<timestamp>.npy`）。如果覆盖到我们用的帧，它能直接当 WorldMirror 的
`depthmap` prior（`cond_flags[0]=1`）——这是除了内外参之外**第三种真值约束**，v1 很
值得一并试，可能进一步压住深度尺度不一致导致的层叠缝。

---

## 9. WorldMirror 的真值注入接口（与本项目对接）

`reconstructor`（`ModelManager` fetch）就是一个 `WorldMirror` 实例，`forward` 原生支持
POW3R 风格 prior 注入，**无需改源码**：

```python
pred = reconstructor(
    views,                       # 含 img/is_target/is_static/timestamp
    cond_flags=[depth, rays, camera],   # 三个 0/1 开关
    is_inference=True, use_motion=False,
)
# views['camera_poses'] : [B,S,4,4] 相机->世界 c2w(OpenCV)  -> cond_flags[2]=1 注入(外参)
# views['camera_intrs'] : [B,S,3,3] 内参 K                  -> cond_flags[1]=1 注入(内参)
# views['depthmap']     : [B,S,H,W] 深度                    -> cond_flags[0]=1 注入(深度)
```

⚠️ 注意：注入 pose prior 后 `normalize_poses` 会把场景归一化到单位立方体，输出高斯落在
**网络归一化系**而非米制全局系。v0 仅看「连续性」足够（连续性与坐标系无关）；**度量尺度 /
地图融合**留到 v1，把每个时刻的高斯钉回米制全局系（见下一节 v1 实际做法）。

---

## 10. v1：沿关键帧累积的「战争迷雾」实现（`fog_world_v1_nuscenes.py`）

v1 = v0 单帧 6 路 GT 约束拼接的基础上，沿关键帧「开车前进」，把每个时刻的重建钉进
**同一个米制全局世界**并累积，渲染自车视角前进视频，世界随车逐步显现。

### 10.1 一帧两套相机位姿（关键）
`load_timestep` 对每路相机同时算两套 c2w，用途不同：

- `c2w_ref`（本时刻 `CAM_FRONT` 车身系下）：**数值小、条件好** → 喂给 WorldMirror 当 prior
  （`views['camera_poses']`，配合 `cond_flags[2]=1`）。
- `c2w_global`（米制全局系下）：**作为把网络输出搬回米制全局的对齐目标**。

### 10.2 摆放：朝向取 GT，尺度取地面（**不再用 Umeyama**）
v1 初版曾用「预测 6 个相机中心 → GT 全局相机中心」的 Umeyama 求相似变换，但 6 路相机
近乎共面、基线仅 ~2m，外推到 ~30m 场景导致**尺度逐帧乱跳（实测 5.3~13.5，差 2.5 倍）→
各帧云大小不一叠在一起 → 穿模**。现改为三步分别求 `(s, R, t)`：

1. **旋转 `R` 取 GT**：对 6 路各自的 `R_glb·R_netᵀ` 做 SVD 平均回 SO(3)（`average_rotation`）。
   用满 6 个**朝向**而非中心 → 不受相机共面影响，稳。同时报「旋转一致性 `maxdev`」做体检。
2. **尺度 `s` + 残余整平 `R_lev` 取地面**：把网络点云按 `R` 旋正后做地面 RANSAC
   （`level_and_scale`，复用 v4），用「已知相机离地高度 = GT 相机平均高度」定米制尺度。
   上千点拟合 → 比 6 个相机中心稳得多。失败则沿用上一帧尺度 `last_good_s`。
3. **平移 `t`**：把网络相机质心对齐到 GT 全局相机质心。

最终相似变换 `R_total = R_lev · R` 作用到该时刻所有高斯（`transform_gaussians_`，位置/尺度/
朝向/速度一起变）。

### 10.3 累积 + 去重 + 剔除（迷雾揭示）
搬运后每个时刻依次：

- `cull_points_`：剔除自车 XY 半径 `--max_range` 外、Z 超出 `[--z_min, --z_max]` 的点
  （地面已钉 z≈0，借此砍地下垃圾与远端卷曲/天空）。
- **跨时刻体素去重**：`voxel_keys` 把坐标量化成整数哈希（边长 `--voxel_size`），与全局
  `occupied` 集比对，**只留新体素** → 既消同表面叠层 smear，又天然实现「迷雾揭示」。

渲染时只画「累积到当前时刻」的高斯，世界随车前进逐步显现。

### 10.4 自车前进视角
`heading_source=motion`（默认）用**相邻时刻真实位移**定前进朝向（与高斯所在全局系严格一致，
推荐）；位移过小（近乎静止）退回车身姿态轴 `ego +x`。视角默认是抬高第三人称跟拍
（相机在 2.5D 壳外俯看，不会被埋），`--first_person` 才是车内视角。相邻关键帧间用
`--frames_per_step` 插值让前进顺滑。

### 10.5 已知局限（留待 v1.2 / v2）
- **动态物体重影**：移动车辆在各关键帧处于不同全局位置，却被当静态全部烘焙进同一世界 →
  沿轨迹留下多份重叠副本；体素去重消不掉（各时刻落在不同体素）。需 GT 3D 框做动静分离
  （v1.2）。
- **真·第一人称会钻进壳**：`--first_person` 把相机塞进 2.5D 壳内会看到空洞，需 NeoVerse
  扩散补洞（v2）。
- 残余尺度轻微波动（实测收紧到 12.6~17.7）可接受。

### 10.6 典型命令
```bash
python fog_world_v1_nuscenes.py --scene_index 0 --use_gt \
    --num_timesteps 20 --output_video outputs/v1_ego_forward.mp4 \
    --output_ply outputs/v1_fog_world.ply
# 降低跟拍视角高度示例：--cam_height 2.5 --back_off 6 --look_height 0.8
```
