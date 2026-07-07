"""
fuse_surround_drone.py — Surround-fused 4D ego-view video from 6 driving cameras (NeoVerse).

Design (v2: rigid-rig anchored fusion)
--------------------------------------
The released reconstructor (WorldMirror) has NO bundle adjustment; cross-view
consistency comes only from photometric correspondence inside the transformer.
Surround cameras barely overlap (<10%, ~0 on the diagonals), so a joint "static"
6-camera pass cannot tie them into one world frame — it produces 6 independent
curved depth shells ("billboards") with wedge-shaped gaps (verified in SuperSplat).

So we STOP asking the reconstructor to align cameras. Instead:

  1. PER-CAMERA 4D pass: each camera's clip alone (use_motion=True) -> a coherent
     4D Gaussian scene in that camera's own frame. (This part works: a single
     camera has 100% temporal overlap.)
  2. RIG ANCHOR: discard the reconstructor's cross-camera poses. Re-express each
     camera's scene relative to its OWN optical frame at the mid instant (via
     C_i^{-1}), then bolt it onto the ego using the KNOWN nuScenes 6-camera
     extrinsics T_ego<-cam_i. Conventions match (both OpenCV/RDF optical), so no
     calibration files are needed.
  3. METRIC SCALE: the reconstructor's depth is up-to-scale per camera. We fit the
     ground plane (RANSAC) in each camera's reconstruction and set the camera's
     height above ground to its known nuScenes mounting height -> per-camera scale
     s_i. This makes all 6 shells metric and co-planar on the ground.
  4. MERGE + RENDER: one fused 4D scene in the EGO frame; render an EGO chase-follow
     trajectory (third-person camera just behind and slightly above the car, always
     looking along the driving direction) over progressing timestamps (scene animates)
     with an optional ego-car box. World up is simply ego +z.

Gaps remain at the seams (thin overlap is a physical limit, holes are acceptable),
but the 6 shells now face the correct directions and share one ground plane, so the
chase view reads as "the real environment restored around the car" rather than a slideshow.

Usage
-----
    python fuse_surround_drone.py --res_dir res \
        --reconstructor_path /mnt/ssd1/wzq_models/NeoVerse/reconstructor.ckpt \
        --output_path outputs/ego_chase.mp4 --car
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from decord import VideoReader
from scipy.spatial.transform import Rotation
from torchvision.transforms import functional as TF

from diffsynth import save_video
from diffsynth.models import ModelManager
from diffsynth.utils.auxiliary import center_crop, homo_matrix_inverse
from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians
from diffsynth.auxiliary_models.worldmirror.models.utils.geometry import depth_to_world_coords_points
from diffsynth.auxiliary_models.worldmirror.utils.save_utils import save_gs_ply

SH_C0 = 0.28209479177387814

RING_ORDER = ["cam_front_left", "cam_front", "cam_front_right",
              "cam_back_right", "cam_back", "cam_back_left"]
FILE_TEMPLATE = "dreamforge_sequence_{view}.mp4"
FRONT_IDX = RING_ORDER.index("cam_front")

# --------------------------- nuScenes standard rig ---------------------------
# Canonical nuScenes camera extrinsics (sensor -> ego), used as a fixed prior so we
# never rely on the reconstructor to localize the cameras relative to each other.
# Each entry: (translation [x,y,z] in ego frame [m], rotation quaternion [w,x,y,z]).
# Ego frame: x forward, y left, z up; origin ~ground level. Camera frame: OpenCV RDF.
# The z of each translation IS that camera's mounting height above ground.
NUSCENES_RIG = {
    "cam_front":       ([1.70079, 0.01595, 1.51096], [0.49980, -0.50303, 0.49978, -0.49737]),
    "cam_front_right": ([1.55085, -0.49340, 1.49575], [0.20603, -0.20269, 0.68245, -0.67136]),
    "cam_back_right":  ([1.01488, -0.48057, 1.56240], [0.12281, -0.13240, -0.70043, 0.69050]),
    "cam_back":        ([0.02833, 0.00345, 1.57910], [0.50379, -0.49740, -0.49419, 0.50455]),
    "cam_back_left":   ([1.03569, 0.48480, 1.59097], [0.69242, -0.70316, -0.11648, 0.11203]),
    "cam_front_left":  ([1.52388, 0.49463, 1.50933], [0.67573, -0.67363, 0.21214, -0.21123]),
}


def build_rig():
    """nuScenes prior -> {view: (T_ego_cam [4,4] float32, mounting_height float)}."""
    rig = {}
    for view, (trans, quat_wxyz) in NUSCENES_RIG.items():
        w, x, y, z = quat_wxyz
        R = Rotation.from_quat([x, y, z, w]).as_matrix()   # scipy expects xyzw
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = trans
        rig[view] = (T, float(trans[2]))
    return rig


# --------------------------- manual per-camera tuning ---------------------------
# Hand-stitching knobs, applied in Phase B ON TOP OF the automatic rig+leveling result.
# Each correction is a rigid nudge in the EGO frame, applied ABOUT THAT CAMERA'S MOUNT
# point (t_ec), so a camera spins/slides around where it sits on the car — intuitive to
# eyeball in SuperSplat. Per view:
#   dyaw/dpitch/droll : extra rotation [deg] about ego axes  x=forward, y=left, z=up
#                       (yaw=turn left/right, pitch=nose up/down, roll=bank)
#   dx/dy/dz          : extra translation [m] in ego frame   (x forward, y left, z up)
#   ds                : extra scale MULTIPLIER about the mount (1.0 = no change)
# Defaults are all-identity -> no effect, fully backward compatible. Edit numbers and
# rerun to hand-align the 6 shells; verify with --dump_ply in SuperSplat.
MANUAL_TUNE = {
    "cam_front":       dict(dyaw=0.0, dpitch=0.0, droll=0.0, dx=0.0, dy=0.0, dz=0.0, ds=1.0),
    "cam_front_right": dict(dyaw=0.0, dpitch=0.0, droll=0.0, dx=0.0, dy=0.0, dz=0.0, ds=1.0),
    "cam_back_right":  dict(dyaw=0.0, dpitch=0.0, droll=0.0, dx=0.0, dy=0.0, dz=0.0, ds=1.0),
    "cam_back":        dict(dyaw=0.0, dpitch=0.0, droll=0.0, dx=0.0, dy=0.0, dz=0.0, ds=1.0),
    "cam_back_left":   dict(dyaw=0.0, dpitch=0.0, droll=0.0, dx=0.0, dy=0.0, dz=0.0, ds=1.0),
    "cam_front_left":  dict(dyaw=0.0, dpitch=0.0, droll=0.0, dx=0.0, dy=0.0, dz=0.0, ds=1.0),
}


def apply_manual_tune(view, s, R_total, t_total, cam_center):
    """Fold a manual per-camera nudge into the similarity (s, R_total, t_total).

    The base maps a recon point p to ego as  x = s * R_total @ p + t_total.
    The nudge adds, about the mount c = cam_center (t_ec):
        scale by ds  -> rotate by Rd  -> translate by t_delta
    All composed back into a SINGLE similarity (s', R', t') so the rest of the
    pipeline (transform_gaussians_) is unchanged. Returns (s', R', t', changed).
    """
    tune = MANUAL_TUNE.get(view)
    if tune is None:
        return s, R_total, t_total, False
    ds = float(tune.get("ds", 1.0))
    t_delta = np.array([tune.get("dx", 0.0), tune.get("dy", 0.0), tune.get("dz", 0.0)],
                       dtype=np.float64)
    # ego axes: x=forward, y=left, z=up -> roll about x, pitch about y, yaw about z
    Rd = Rotation.from_euler(
        "xyz", [tune.get("droll", 0.0), tune.get("dpitch", 0.0), tune.get("dyaw", 0.0)],
        degrees=True).as_matrix()
    changed = (ds != 1.0) or np.any(t_delta != 0.0) or not np.allclose(Rd, np.eye(3))
    if not changed:
        return s, R_total, t_total, False
    c = np.asarray(cam_center, dtype=np.float64)
    R_total = R_total.astype(np.float64)
    t_total = t_total.astype(np.float64)
    # x' = Rd @ ( ds*(x - c) + c ) + c_back? -> derived single similarity:
    #   x' = (ds) * Rd @ R_total @ p + Rd @ (ds*t_total + (1-ds)*c) - Rd @ c + c + t_delta
    s_f = ds * s
    R_f = (Rd @ R_total).astype(np.float32)
    t_f = (Rd @ (ds * t_total + (1.0 - ds) * c) - Rd @ c + c + t_delta).astype(np.float32)
    return s_f, R_f, t_f, True


# ----------------------------- io helpers -----------------------------------
def load_consecutive(video_path, start, count, stride, resolution):
    """Read `count` CONSECUTIVE frames (stride apart) from `start`, center-cropped.

    Unlike auxiliary.load_video (which linspace-samples the WHOLE clip), this keeps the
    inter-frame interval small so the reconstructor's "linear motion between adjacent
    frames" assumption holds — important for clean 4D dynamic reconstruction of fast
    driving. The same (start, count, stride) is used for all 6 synchronized cameras, so
    frame index k corresponds to the same real instant across cameras (temporal alignment).
    """
    vr = VideoReader(video_path)
    n = len(vr)
    idx = [min(start + k * stride, n - 1) for k in range(count)]
    frames = vr.get_batch(idx).asnumpy()
    return [center_crop(Image.fromarray(f), resolution) for f in frames]


# ----------------------------- geometry helpers -----------------------------
def normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v) + eps)


def look_at_opencv(eye, target, up):
    z = normalize(target - eye)
    x = normalize(np.cross(z, up))
    y = np.cross(z, x)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = x, y, z
    c2w[:3, 3] = eye
    return c2w


def unproject_frame(predictions, frame_idx, device):
    """World points + valid mask for one frame, from gs_depth + predicted camera."""
    depth = predictions["gs_depth"][0, frame_idx:frame_idx + 1]          # [1,H,W] or [1,H,W,1]
    if depth.dim() == 4:
        depth = depth[..., 0]                                            # -> [1,H,W]
    extr = predictions["rendered_extrinsics"][0, frame_idx:frame_idx + 1]  # [1,4,4] c2w
    intr = predictions["rendered_intrinsics"][0, frame_idx:frame_idx + 1]  # [1,3,3]
    pts, _, mask = depth_to_world_coords_points(depth.float(), extr.float(), intr.float())
    return pts[0].reshape(-1, 3), mask[0].reshape(-1)                     # [HW,3], [HW]


def rotation_align(a, b):
    """Rotation matrix that rotates unit vector a onto unit vector b (Rodrigues)."""
    a, b = normalize(a), normalize(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:                 # already aligned (or anti-parallel)
        return np.eye(3, dtype=np.float32) if c > 0 else -np.eye(3, dtype=np.float32)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    return R.astype(np.float32)


def level_and_scale(pts_recon, R_rig, t_rig, cam_center, known_h,
                    thresh_frac=0.02, n_iter=400, min_pts=200):
    """Ground-based leveling + metric scale, computed IN THE EGO FRAME.

    pts_recon:  [N,3] valid recon-frame points of the camera (mid frame).
    R_rig,t_rig: unit-scale rig placement (recon -> ego): p_ego = pts @ R_rig.T + t_rig.
    cam_center: the camera's center in ego frame = its rig mounting position (t_ec).
    known_h:    the camera's mounting height (ego z of cam_center).

    Doing the fit in ego frame lets us use the KNOWN up (+z) to robustly tell the road
    from facades: we keep points below the camera and accept only near-horizontal planes.
    Returns (s, R_lev): scale and the rotation that makes that camera's ground level.
    R_lev rotates the fitted ground normal onto +z; s puts the ground at z=0 with the
    camera at its known height. Returns (None, I) if the fit is unreliable.
    """
    p1 = pts_recon @ R_rig.T + t_rig                         # ego frame, unit scale
    below = p1[:, 2] < cam_center[2]
    cand = p1[below] if int(below.sum()) >= min_pts else p1
    if cand.shape[0] < min_pts:
        return None, np.eye(3, dtype=np.float32)

    thresh = thresh_frac * np.median(np.linalg.norm(cand - cand.mean(0), axis=1))
    up = np.array([0.0, 0.0, 1.0])
    N = cand.shape[0]
    best_inl, best = 0, None
    for _ in range(n_iter):
        tri = cand[np.random.choice(N, 3, replace=False)]
        nrm = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        ln = np.linalg.norm(nrm)
        if ln < 1e-8:
            continue
        nrm = nrm / ln
        if abs(nrm @ up) < 0.85:                 # reject non-horizontal (facade) planes
            continue
        d = -nrm @ tri[0]
        inl = int((np.abs(cand @ nrm + d) < thresh).sum())
        if inl > best_inl:
            best_inl, best = inl, (nrm, d)
    if best is None or best_inl < min_pts:
        return None, np.eye(3, dtype=np.float32)

    nrm, d = best
    if nrm[2] < 0:                               # orient normal upward
        nrm, d = -nrm, -d
    dist = abs(nrm @ cam_center + d)             # perpendicular camera->ground distance
    if dist < 1e-6:
        return None, np.eye(3, dtype=np.float32)
    s = float(known_h / dist)
    R_lev = rotation_align(nrm, up)
    return s, R_lev


def transform_gaussians_(g, s, R_np, t_np, device):
    """In-place apply similarity (s, R, t) to a Gaussians: means, scales, rotation, velocity."""
    R = torch.tensor(R_np, dtype=torch.float32, device=device)
    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    g.means = s * (g.means.float() @ R.T) + t
    g.scales = g.scales.float() * s
    # Rotate Gaussian orientations: R_new = R @ R_old  (quat stored as [w,x,y,z]).
    q = g.rotations.detach().cpu().numpy()
    q_xyzw = q[:, [1, 2, 3, 0]]
    new_xyzw = (Rotation.from_matrix(R_np) * Rotation.from_quat(q_xyzw)).as_quat()
    new_wxyz = np.concatenate([new_xyzw[:, 3:4], new_xyzw[:, :3]], axis=1)
    g.rotations = torch.tensor(new_wxyz, dtype=torch.float32, device=device)
    for attr in ("forward_vel", "backward_vel"):
        v = getattr(g, attr, None)
        if v is not None:
            setattr(g, attr, s * (v.float() @ R.T))
    return g


def cull_by_radius_(g, center_xy, radius):
    """In-place drop a Gaussians' points whose XY distance to center exceeds radius.
    Removes each camera's unreliable, curved far field so the near-field shells tile
    together more tightly (smaller seams). Masks every per-point tensor attribute."""
    n = g.means.shape[0]
    if n == 0:
        return g
    c = torch.tensor(center_xy, dtype=torch.float32, device=g.means.device)
    keep = torch.linalg.norm(g.means[:, :2].float() - c, dim=1) <= radius
    if bool(keep.all()):
        return g
    for k, v in list(g.__dict__.items()):
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n:
            setattr(g, k, v[keep])
    return g


def make_car_gaussians(center, heading, up, length, width, height, sh_dim, device,
                       color=(0.9, 0.1, 0.1), n_per_edge=20, thickness=0.06):
    """A hollow WIREFRAME box (12 edges only) marking the ego — small, see-through, so it
    does not occlude the surrounding scene. Dimensions are absolute meters (v3 is metric)."""
    fwd = normalize(heading)
    right = normalize(np.cross(up, fwd))   # right-handed: right x upn = fwd (det +1)
    upn = np.cross(fwd, right)
    R = np.stack([right, upn, fwd], axis=1)
    hw, hl = width / 2.0, length / 2.0
    xs, ys, zs = (-hw, hw), (0.0, height), (-hl, hl)   # local: x=right, y=up, z=fwd
    t = np.linspace(0.0, 1.0, n_per_edge)
    edges = []
    for y in ys:                                       # 4 edges along x (length: width)
        for z in zs:
            edges.append(np.stack([-hw + 2 * hw * t, np.full_like(t, y), np.full_like(t, z)], 1))
    for x in xs:                                       # 4 edges along y (height)
        for z in zs:
            edges.append(np.stack([np.full_like(t, x), height * t, np.full_like(t, z)], 1))
    for x in xs:                                       # 4 edges along z (length)
        for y in ys:
            edges.append(np.stack([np.full_like(t, x), np.full_like(t, y), -hl + 2 * hl * t], 1))
    local = np.concatenate(edges, axis=0)
    means = (local @ R.T) + center[None]
    n = means.shape[0]
    q_xyzw = Rotation.from_matrix(R).as_quat()
    quat = torch.tensor([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
                        dtype=torch.float32, device=device).repeat(n, 1)
    sh = torch.zeros(n, sh_dim, 3, dtype=torch.float32, device=device)
    sh[:, 0, :] = torch.tensor([(c - 0.5) / SH_C0 for c in color], device=device)
    return Gaussians(
        means=torch.tensor(means, dtype=torch.float32, device=device),
        harmonics=sh, opacities=torch.ones(n, device=device),
        scales=torch.full((n, 3), thickness, device=device),
        rotations=quat, timestamp=-1)


def make_ground_gaussians(center_xy, radius, sh_dim, device,
                          color=(0.22, 0.22, 0.24), spacing=0.4, z=0.0):
    """A flat disk of Gaussians at z=0 — a neutral road patch to fill the central donut
    hole around the rig (no camera sees directly under/at the ego) and the ground seams."""
    g = np.arange(-radius, radius + spacing, spacing)
    gx, gy = np.meshgrid(g, g)
    xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
    xy = xy[np.linalg.norm(xy, axis=1) <= radius] + np.asarray(center_xy)[None]
    n = xy.shape[0]
    means = np.concatenate([xy, np.full((n, 1), z)], axis=1).astype(np.float32)
    scales = torch.tensor([spacing * 0.7, spacing * 0.7, 0.02],
                          dtype=torch.float32, device=device).repeat(n, 1)  # flat in z
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device).repeat(n, 1)
    sh = torch.zeros(n, sh_dim, 3, dtype=torch.float32, device=device)
    sh[:, 0, :] = torch.tensor([(c - 0.5) / SH_C0 for c in color], device=device)
    return Gaussians(
        means=torch.tensor(means, device=device),
        harmonics=sh, opacities=torch.ones(n, device=device),
        scales=scales, rotations=quat, timestamp=-1)


def dump_fused_ply(fused, path, max_points):
    """Write the fused 4D scene to a standard 3DGS .ply for SuperSplat inspection.

    Exports a SNAPSHOT (each Gaussian at its own base timestamp) — enough to verify the
    6 shells now face the right directions and share one ground plane. Subsamples to
    `max_points` so the file stays small. Reuses NeoVerse's save_gs_ply, converting our
    activated tensors to the on-disk convention it expects: harmonics DC -> f_dc,
    scales -> log() (done inside save_gs_ply), opacities [0,1] -> logit.
    """
    total = sum(g.means.shape[0] for g in fused)
    keep = min(1.0, max_points / max(total, 1))
    ms, sc, ro, dc, op = [], [], [], [], []
    for g in fused:
        n = g.means.shape[0]
        if n == 0:
            continue
        if keep < 1.0:
            idx = torch.randperm(n, device=g.means.device)[:max(1, int(n * keep))]
        else:
            idx = slice(None)
        ms.append(g.means[idx].float())
        sc.append(g.scales[idx].float())
        ro.append(g.rotations[idx].float())
        dc.append(g.harmonics[idx, 0, :].float())          # SH DC -> f_dc_*
        op.append(g.opacities[idx].float())
    means, scales, rot = torch.cat(ms), torch.cat(sc), torch.cat(ro)
    rgbs = torch.cat(dc)
    opac = torch.logit(torch.cat(op).clamp(1e-4, 1 - 1e-4))  # activated [0,1] -> logit
    save_gs_ply(Path(path), means, scales, rot, rgbs, opac)
    print(f"Saved fused PLY ({means.shape[0]:,} of {total:,} Gaussians) -> {path}")


# --------------------------------- main -------------------------------------
@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda"
    torch_dtype = torch.bfloat16
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    if not os.path.isfile(args.reconstructor_path):
        raise FileNotFoundError(f"Reconstructor checkpoint not found: {args.reconstructor_path}")

    T = args.num_input_frames
    mid = T // 2
    res = (args.width, args.height)
    rig = build_rig()

    # Load all 6 clips with the SAME consecutive window, so frame k is the same real
    # instant across the 6 synchronized cameras.
    print(f"Loading 6 camera clips (consecutive window: start={args.start_frame}, "
          f"count={T}, stride={args.frame_stride}) ...")
    clips = []
    for view in RING_ORDER:
        path = os.path.join(args.res_dir, FILE_TEMPLATE.format(view=view))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing camera video: {path}")
        clips.append(load_consecutive(path, args.start_frame, T, args.frame_stride, res))

    print("Loading reconstructor ...")
    mm = ModelManager()
    mm.load_model(args.reconstructor_path, device=device, torch_dtype=torch_dtype)
    reconstructor = mm.fetch_model("reconstructor")

    # --- Phase A: per-camera 4D reconstruction + per-camera metric scale ---
    # We keep the Gaussian objects alive (their tensors stay on GPU) and discard the
    # rest of each prediction; transforms are applied later, after median-filling scale.
    recons = []   # per camera: {view, gaussians, R_ec, t_ec, R_wc, t_wc, scale, R_lev}
    front_ego = None    # front camera centers over time (recon frame)
    K_ref = None        # render intrinsics (front camera, mid frame)
    for i, view in enumerate(RING_ORDER):
        print(f"4D pass: {view} ...")
        views = {
            "img": torch.stack([TF.to_tensor(im)[None] for im in clips[i]], dim=1).to(device),
            "is_target": torch.zeros((1, T), dtype=torch.bool, device=device),
            "is_static": torch.zeros((1, T), dtype=torch.bool, device=device),
            "timestamp": torch.arange(0, T, dtype=torch.int64, device=device).unsqueeze(0),
        }
        with torch.amp.autocast("cuda", dtype=torch_dtype):
            pred = reconstructor(views, is_inference=True, use_motion=True)

        C = pred["rendered_extrinsics"][0, mid].float().cpu().numpy()     # c2w at mid
        C_inv = np.linalg.inv(C).astype(np.float32)                      # w2c (cam at origin)
        Te, known_h = rig[view]
        R_ec, t_ec = Te[:3, :3], Te[:3, 3]
        R_wc, t_wc = C_inv[:3, :3], C_inv[:3, 3]
        R_rig = (R_ec @ R_wc).astype(np.float32)                          # unit-scale rig rot
        t_rig = (R_ec @ t_wc + t_ec).astype(np.float32)                   # unit-scale rig trans

        if args.no_ground_scale:
            s, R_lev = 1.0, np.eye(3, dtype=np.float32)
        else:
            pts, mask = unproject_frame(pred, mid, device)
            pts_v = pts[mask].cpu().numpy()
            s, R_lev = level_and_scale(pts_v, R_rig, t_rig, t_ec, known_h,
                                       thresh_frac=args.ground_thresh_frac)
            print(f"    ground scale = {s if s is None else round(s, 4)}")

        gaussians = [g for g in pred["splats"][0] if g.means.shape[0] > 0]
        recons.append({"view": view, "gaussians": gaussians,
                       "R_ec": R_ec, "t_ec": t_ec, "R_wc": R_wc, "t_wc": t_wc,
                       "scale": s, "R_lev": R_lev})

        if i == FRONT_IDX:
            front_ego = pred["rendered_extrinsics"][0, :, :3, 3].float().cpu().numpy()  # [T,3]
            K_ref = pred["rendered_intrinsics"][0, mid].float()

        del pred
        torch.cuda.empty_cache()

    # Median-fill cameras whose ground fit failed (R_lev stays I, scale -> median).
    good = [r["scale"] for r in recons if r["scale"] and r["scale"] > 0]
    med = float(np.median(good)) if good else 1.0
    for r in recons:
        if not r["scale"] or r["scale"] <= 0:
            print(f"    {r['view']}: ground fit failed -> using median scale {med:.4f}")
            r["scale"] = med

    # --- Phase B: anchor + level every camera into the ego frame ---
    # p_ego = cam_center + s * R_lev @ ( (rig placement of p) - cam_center )
    #   rig placement: R_rig = R_ec @ R_wc,  cam_center maps to t_ec.
    #   => similarity (s, R, t):  R = R_lev @ R_ec @ R_wc,
    #                             t = s * R_lev @ (R_ec @ t_wc) + t_ec
    fused = []
    front_sRt = None
    for r in recons:
        s = r["scale"]
        R_ec, t_ec = r["R_ec"], r["t_ec"]
        R_wc, t_wc = r["R_wc"], r["t_wc"]
        R_lev = r["R_lev"]
        R_total = (R_lev @ R_ec @ R_wc).astype(np.float32)
        t_total = (s * (R_lev @ (R_ec @ t_wc)) + t_ec).astype(np.float32)
        # Anchor the chase path to the UN-tuned front transform, so manual nudges move
        # only the shells, not the camera framing.
        if r["view"] == "cam_front":
            front_sRt = (s, R_total, t_total)
        # Optional hand-stitch nudge for this camera (no-op unless MANUAL_TUNE edited).
        if not args.no_manual_tune:
            s, R_total, t_total, changed = apply_manual_tune(r["view"], s, R_total, t_total, t_ec)
            if changed:
                print(f"    manual tune applied: {r['view']} -> {MANUAL_TUNE[r['view']]}")
        for g in r["gaussians"]:
            tg = transform_gaussians_(g, s, R_total, t_total, device)
            if args.max_range > 0:
                cull_by_radius_(tg, (0.0, 0.0), args.max_range)   # disk around ego origin
            if tg.means.shape[0] > 0:
                fused.append(tg)

    sh_dim = next(g.harmonics.shape[1] for g in fused)

    if args.dump_ply:
        dump_fused_ply(fused, args.dump_ply, args.ply_max_points)

    # Ego path in the fused (ego) frame: front-camera centers under the front similarity.
    s, R_total, t_total = front_sRt
    ego = (s * (front_ego @ R_total.T) + t_total).astype(np.float32)     # [T,3]
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)               # ego up

    # Optional flat ground patch (static) covering the rig + a margin, to fill the central
    # hole and ground seams the 6 outward-looking cameras cannot see.
    ground = None
    if args.ground_fill:
        cx, cy = float(np.median(ego[:, 0])), float(np.median(ego[:, 1]))
        ground = make_ground_gaussians((cx, cy), args.ground_radius, sh_dim, device,
                                       spacing=args.ground_spacing)
        print(f"Ground fill: {ground.means.shape[0]:,} gaussians, r={args.ground_radius}m")

    # --- Phase C: EGO chase-follow trajectory over the fused 4D scene ---
    # Offsets are ABSOLUTE METERS (scene is metric after ground scaling). The camera is
    # a third-person chase cam bolted to the ego: just behind it, slightly above the roof,
    # always looking along the driving direction. Anchored to the GROUND point under the
    # ego, so heights are true heights above the road. This "restores" the scene as the
    # car experiences it (ego view) instead of an overhead drone.
    n = args.num_frames
    taus = np.linspace(0, T - 1, n)
    h_off, b_off, look, look_h = args.height_off, args.back_off, args.look_ahead, args.look_height

    print(f"Rendering {n} fused ego-chase frames ({len(fused)} Gaussian groups) ...")
    rasterizer = reconstructor.gs_renderer.rasterizer
    frames = []
    for i in range(n):
        tau = taus[i]
        lo = int(np.floor(tau)); hi = min(lo + 1, T - 1); fr = tau - lo
        p = ego[lo] * (1 - fr) + ego[hi] * fr
        a, b = max(lo - 1, 0), min(lo + 1, T - 1)
        head = normalize(ego[b] - ego[a])
        head_h = normalize(np.array([head[0], head[1], 0.0], dtype=np.float32))  # horizontal
        gnd = np.array([p[0], p[1], 0.0], dtype=np.float32)                      # ground anchor
        eye = gnd - head_h * b_off + up_world * h_off                            # behind + above roof
        target = gnd + head_h * look + up_world * look_h                         # ahead, lifted for a natural gaze
        c2w = look_at_opencv(eye, target, up_world)
        w2c = homo_matrix_inverse(torch.tensor(c2w, device=device))[None]
        ts = torch.tensor([float(mid) if args.freeze_time else float(tau)], device=device)
        render_list = list(fused)
        if ground is not None:
            render_list.append(ground)
        if args.car:
            render_list.append(make_car_gaussians(
                gnd, head_h, up_world, args.car_len, args.car_wid, args.car_hgt,
                sh_dim, device))
        rgb, _, _ = rasterizer.forward(
            [render_list], render_viewmats=[w2c], render_Ks=[K_ref[None]],
            render_timestamps=[ts], sh_degree=0, width=args.width, height=args.height)
        frames.append((rgb[0, 0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy())

    save_video([np.asarray(f) for f in frames], args.output_path, fps=args.fps)
    print(f"Saved surround-fused ego-chase video -> {args.output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Surround-fused 4D ego-view video (6 cameras)")
    p.add_argument("--res_dir", default="res")
    p.add_argument("--output_path", default="outputs/ego_chase.mp4")
    p.add_argument("--reconstructor_path", default="/mnt/ssd1/wzq_models/NeoVerse/reconstructor.ckpt")
    p.add_argument("--num_input_frames", type=int, default=49,
                   help="Frames per camera fed to the reconstructor (consecutive window length)")
    p.add_argument("--start_frame", type=int, default=0,
                   help="First frame index of the consecutive reconstruction window")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="Stride between sampled frames (1 = truly consecutive)")
    p.add_argument("--width", type=int, default=560)
    p.add_argument("--height", type=int, default=336)
    p.add_argument("--num_frames", type=int, default=97)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--ground_thresh_frac", type=float, default=0.02,
                   help="RANSAC inlier threshold as a fraction of scene extent")
    p.add_argument("--no_ground_scale", action="store_true",
                   help="Skip ground fitting; use unit scale for every camera (debug)")
    p.add_argument("--no_manual_tune", action="store_true",
                   help="Ignore the MANUAL_TUNE per-camera hand-stitch nudges (use pure rig)")
    p.add_argument("--dump_ply", default=None,
                   help="If set, also write the fused 4D scene to this .ply for SuperSplat")
    p.add_argument("--ply_max_points", type=int, default=2_000_000,
                   help="Subsample cap for the dumped PLY (keeps the file small)")
    p.add_argument("--height_off", type=float, default=3.0,
                   help="Chase-cam height above road [m] (ego third-person, just above the roof)")
    p.add_argument("--back_off", type=float, default=6.0,
                   help="Chase-cam distance behind the ego [m] (absolute)")
    p.add_argument("--look_ahead", type=float, default=8.0,
                   help="Look-at point ahead of the ego [m]")
    p.add_argument("--look_height", type=float, default=1.0,
                   help="Height of the look-at point above the road [m] (lifts the gaze so the "
                        "camera does not stare down at the asphalt)")
    p.add_argument("--max_range", type=float, default=0.0,
                   help="Cull each shell's points beyond this XY radius [m] from ego (0=off); "
                        "trims the unreliable curved far field to tighten view seams")
    p.add_argument("--freeze_time", action="store_true",
                   help="Render the whole flight at the mid instant (no inter-shell temporal "
                        "drift) — cleanest temporally-unified flythrough")
    p.add_argument("--ground_fill", action="store_true",
                   help="Add a flat road patch at z=0 to fill the central hole / seams")
    p.add_argument("--ground_radius", type=float, default=30.0,
                   help="Radius of the ground-fill patch [m]")
    p.add_argument("--ground_spacing", type=float, default=0.4,
                   help="Ground-fill point spacing [m]")
    p.add_argument("--car", action="store_true")
    p.add_argument("--car_len", type=float, default=4.5, help="Ego box length [m]")
    p.add_argument("--car_wid", type=float, default=2.0, help="Ego box width [m]")
    p.add_argument("--car_hgt", type=float, default=1.6, help="Ego box height [m]")
    return p.parse_args()


if __name__ == "__main__":
    main()
