"""
extract_intermediates.py — Extract & save intermediate products from NeoVerse pipeline.

Pipeline trace:
    Input Video
        │
        ├─[1] WorldMirror 4DGS Reconstruction
        │       └─ pristine Gaussians (完整、未劣化的高斯基元)
        │
        ├─[2] Pristine GS Render  ← 低质量GS渲染视频 (产物1)
        │       └─ rasterizer.forward(pristine_gaussians, target_poses)
        │
        ├─[3] Degradation Simulation  ← 模拟退化
        │       ├─ novel_view_sampling → 生成新视角
        │       └─ degradation_simulation(gaussians_copy, novel_poses)
        │            ├─ 可见性剔除 (culling)
        │            └─ 几何滤波 (avg filter, kernel>0)
        │
        ├─[4] Degraded GS Render  ← 退化后渲染视频 (产物2)
        │       └─ rasterizer.forward(degraded_gaussians, target_poses)
        │
        └─[5] WAN Diffusion → 高质量最终视频 (可选产物3)

Usage:
    # 完整提取: 只保存前两个中间产物 (快)
    CUDA_VISIBLE_DEVICES=0 python extract_intermediates.py \
        --input_path examples/videos/robot.mp4 \
        --trajectory tilt_up \
        --model_path /mnt/ssd1/wzq_models \
        --reconstructor_path /mnt/ssd1/wzq_models/NeoVerse/reconstructor.ckpt \
        --output_dir outputs/intermediates

    # 包含最终扩散模型输出 (慢，需加载WAN模型)
    CUDA_VISIBLE_DEVICES=0 python extract_intermediates.py \
        --input_path examples/videos/robot.mp4 \
        --trajectory tilt_up \
        --model_path /mnt/ssd1/wzq_models \
        --reconstructor_path /mnt/ssd1/wzq_models/NeoVerse/reconstructor.ckpt \
        --output_dir outputs/intermediates \
        --run_diffusion

    # 只对比纯渲染 vs 退化 (跳过扩散)
    CUDA_VISIBLE_DEVICES=0 python extract_intermediates.py \
        --input_path examples/videos/robot.mp4 \
        --trajectory tilt_up \
        --model_path /mnt/ssd1/wzq_models \
        --reconstructor_path /mnt/ssd1/wzq_models/NeoVerse/reconstructor.ckpt \
        --output_dir outputs/intermediates \
        --no_diffusion
"""

import torch
import os
import argparse
import copy
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image

from diffsynth.pipelines.wan_video_neoverse import (
    WanVideoNeoVersePipeline,
    WanVideoUnit_4DPreprocesser,
)
from diffsynth import save_video
from diffsynth.utils.auxiliary import (
    CameraTrajectory,
    load_video,
    homo_matrix_inverse,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def clone_gaussians(gaussians):
    """
    Deep-copy nested Gaussians list.
    degradation_simulation mutates Gaussians in-place, so we need a pristine copy
    for the "before degradation" rendering.
    """
    cloned = []
    for b_idx in range(len(gaussians)):
        frame_list = []
        for s_idx in range(len(gaussians[b_idx])):
            g = gaussians[b_idx][s_idx]
            # Collect all tensor attributes dynamically
            kwargs = {}
            for attr_name, value in g.__dict__.items():
                if isinstance(value, torch.Tensor):
                    kwargs[attr_name] = value.clone()
                elif isinstance(value, (int, float)) or value is None:
                    kwargs[attr_name] = value
            frame_list.append(type(g)(**kwargs))
        cloned.append(frame_list)
    return cloned


def render_frames_to_video(rasterizer, gaussians, world2cam_list, K_list,
                           timestamps_list, height, width):
    """
    Render frames from Gaussians using given camera parameters.
    Returns list of PIL Images.
    """
    frames = []
    for i in range(len(world2cam_list)):
        w2c = world2cam_list[i:i+1]  # [1, 4, 4]
        K = K_list[i:i+1]            # [1, 3, 3]
        ts = timestamps_list[i:i+1]   # [1]

        rgb, depth, alpha = rasterizer.forward(
            gaussians,
            render_viewmats=[w2c],
            render_Ks=[K],
            render_timestamps=[ts],
            sh_degree=0,
            width=width,
            height=height,
        )
        # rgb: [1, 1, H, W, 3]
        img = rgb[0, 0].clamp(0, 1)
        img_uint8 = (img * 255).to(torch.uint8).cpu().numpy()
        frames.append(Image.fromarray(img_uint8))
    return frames


def render_context_frames(rasterizer, gaussians, rendered_extrinsics,
                          rendered_intrinsics, rendered_timestamps, height, width):
    """
    Render all context frames from their original camera poses.
    Returns list of PIL Images.
    """
    frames = []
    B = len(gaussians)
    for b_idx in range(B):
        S = len(gaussians[b_idx])
        w2c_list = homo_matrix_inverse(rendered_extrinsics[b_idx])  # [S, 4, 4]
        K_list = rendered_intrinsics[b_idx]                          # [S, 3, 3]
        ts_list = rendered_timestamps[b_idx]                         # [S]

        for s_idx in range(S):
            w2c = w2c_list[s_idx:s_idx+1]
            K = K_list[s_idx:s_idx+1]
            ts = ts_list[s_idx:s_idx+1]

            single_gaussian = [gaussians[b_idx][s_idx:s_idx+1]]

            rgb, depth, alpha = rasterizer.forward(
                single_gaussian,
                render_viewmats=[w2c],
                render_Ks=[K],
                render_timestamps=[ts],
                sh_degree=0,
                width=width,
                height=height,
            )
            img = rgb[0, 0].clamp(0, 1)
            img_uint8 = (img * 255).to(torch.uint8).cpu().numpy()
            frames.append(Image.fromarray(img_uint8))
    return frames


def count_gaussians(gaussians):
    """Return total and per-frame Gaussian counts."""
    total = 0
    per_frame = []
    for b_idx in range(len(gaussians)):
        for s_idx in range(len(gaussians[b_idx])):
            n = gaussians[b_idx][s_idx].means.shape[0]
            total += n
            per_frame.append(n)
    return total, per_frame


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_intermediates(
    pipe,
    input_video,
    cam_traj: CameraTrajectory,
    output_dir: str,
    prompt: str = "",
    negative_prompt: str = "",
    alpha_threshold: float = 1.0,
    static_flag: bool = False,
    seed: int = 42,
    cfg_scale: float = 1.0,
    num_inference_steps: int = 4,
    # Degradation params
    kernel_size: int = 21,
    occlusion_thresh: float = 0.1,
    novel_view_trans: tuple = (0.01, 0.1),
    run_diffusion: bool = True,
):
    device = pipe.device
    height, width = input_video[0].size[1], input_video[0].size[0]
    num_context = len(input_video)

    print(f"\n{'='*60}")
    print(f"  Configuration")
    print(f"{'='*60}")
    print(f"  Input frames:   {num_context}")
    print(f"  Resolution:     {width}x{height}")
    print(f"  Static scene:   {static_flag}")
    print(f"  Degradation:    kernel_size={kernel_size}, occlusion={occlusion_thresh}")
    print(f"  Novel trans:    {novel_view_trans}")
    print(f"  Run diffusion:  {run_diffusion}")
    print(f"  Output dir:     {output_dir}")

    # ═══════════════════════════════════════════════════════════════
    # Step 1: WorldMirror 4DGS Reconstruction
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  [1/5] WorldMirror 4DGS Reconstruction")
    print(f"{'─'*60}")

    views = {
        "img": torch.stack(
            [F.to_tensor(image)[None] for image in input_video], dim=1
        ).to(device),
        "is_target": torch.zeros((1, len(input_video)), dtype=torch.bool, device=device),
    }
    if static_flag:
        views["is_static"] = torch.ones((1, len(input_video)), dtype=torch.bool, device=device)
        views["timestamp"] = torch.zeros((1, len(input_video)), dtype=torch.int64, device=device)
    else:
        views["is_static"] = torch.zeros((1, len(input_video)), dtype=torch.bool, device=device)
        views["timestamp"] = torch.arange(0, len(input_video), dtype=torch.int64, device=device).unsqueeze(0)

    if pipe.vram_management_enabled:
        pipe.reconstructor.to(device)

    with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
        predictions = pipe.reconstructor(views, is_inference=True, use_motion=False)

    if pipe.vram_management_enabled:
        pipe.reconstructor.cpu()
        torch.cuda.empty_cache()

    gaussians = predictions["splats"]
    total_gs, per_frame_gs = count_gaussians(gaussians)
    print(f"  Total Gaussians:  {total_gs:,}")
    print(f"  Per-frame Gaussians: {per_frame_gs}")
    print(f"  Avg per frame:    {total_gs // len(per_frame_gs):,}")

    # Build target trajectory camera params
    K = predictions["rendered_intrinsics"][0]
    target_cam2world = cam_traj.c2w.to(device)
    if cam_traj.mode == "relative" and not static_flag:
        target_cam2world = predictions["rendered_extrinsics"][0] @ target_cam2world

    ratio = torch.linspace(1, cam_traj.zoom_ratio, K.shape[0], device=device)
    K_zoomed = K.clone()
    K_zoomed[:, 0, 0] *= ratio
    K_zoomed[:, 1, 1] *= ratio

    timestamps = predictions["rendered_timestamps"][0]

    if static_flag:
        K_zoomed = K_zoomed[:1].repeat(len(cam_traj), 1, 1)
        tgt_ts = timestamps[:1].repeat(len(cam_traj))
    else:
        tgt_ts = timestamps

    target_world2cam = homo_matrix_inverse(target_cam2world)
    num_target = len(target_cam2world)

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Pristine GS Render (产物1 — 低质量GS渲染)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  [2/5] Pristine GS Render (低质量GS渲染)")
    print(f"{'─'*60}")

    rasterizer = pipe.reconstructor.gs_renderer.rasterizer

    # 2a. Target trajectory render (新视角)
    print(f"  Rendering {num_target} frames from target trajectory...")
    pristine_target_frames = render_frames_to_video(
        rasterizer, gaussians,
        [target_world2cam[i] for i in range(num_target)],
        [K_zoomed[i] for i in range(num_target)],
        [tgt_ts[i] for i in range(num_target)],
        height, width,
    )

    pristine_target_path = os.path.join(output_dir, "1_pristine_gs_target.mp4")
    save_video(pristine_target_frames, pristine_target_path, fps=16)
    print(f"  Saved: {pristine_target_path}")

    # 2b. Context frame render (原始视角)
    print(f"  Rendering {num_context} context frames from original poses...")
    context_extrinsics = predictions["rendered_extrinsics"]
    context_intrinsics = predictions["rendered_intrinsics"]
    context_timestamps = predictions["rendered_timestamps"]

    pristine_context_frames = render_context_frames(
        rasterizer, gaussians,
        context_extrinsics, context_intrinsics, context_timestamps,
        height, width,
    )
    pristine_context_path = os.path.join(output_dir, "1_pristine_gs_context.mp4")
    save_video(pristine_context_frames, pristine_context_path, fps=16)
    print(f"  Saved: {pristine_context_path}")

    # Also save alpha mask for reference
    target_rgb, target_depth, target_alpha = rasterizer.forward(
        gaussians,
        render_viewmats=[target_world2cam],
        render_Ks=[K_zoomed],
        render_timestamps=[tgt_ts],
        sh_degree=0, width=width, height=height,
    )
    target_mask = (target_alpha > alpha_threshold).float()
    mask_frames = []
    for i in range(num_target):
        m = target_mask[0, i, :, :, 0].clamp(0, 1)
        m_uint8 = (m * 255).to(torch.uint8).cpu().numpy()
        mask_frames.append(Image.fromarray(m_uint8))
    mask_path = os.path.join(output_dir, "1_pristine_gs_mask.mp4")
    save_video(mask_frames, mask_path, fps=16)
    print(f"  Saved: {mask_path} (alpha mask)")

    # ═══════════════════════════════════════════════════════════════
    # Step 3: Degradation Simulation (模拟退化)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  [3/5] Degradation Simulation")
    print(f"{'─'*60}")

    # Deep-copy Gaussians before degradation (degradation mutates in-place)
    degraded_gaussians = clone_gaussians(gaussians)

    # Create the preprocessor unit for novel_view_sampling + degradation
    unit = WanVideoUnit_4DPreprocesser(
        novel_view_sampling_trans=list(novel_view_trans),
        novel_view_sampling_max_rot=0.0,
        culling_prob=0.0,  # We explicitly choose kernel_size
        kernel_size_range=[kernel_size, kernel_size],
        occlusion_thresh=occlusion_thresh,
        alpha_thresh=alpha_threshold,
        color_thresh=[50, 100],
    )

    # Generate novel view poses from context frames
    context_num = num_context
    novel_context_poses = unit.novel_view_sampling(
        context_extrinsics[:, :context_num],
        predictions["gs_depth"].squeeze(-1),
    )
    print(f"  Novel view poses shape: {novel_context_poses.shape}")
    print(f"  Degradation mode: {'Visibility Culling' if kernel_size == 0 else f'Avg Geometry Filter (k={kernel_size})'}")

    # Run degradation
    degraded_gaussians = unit.degradation_simulation(
        degraded_gaussians,
        novel_context_poses,
        context_intrinsics[:, :context_num],
        (height, width),
        kernel_size=kernel_size,
        occlusion_thresh=occlusion_thresh,
    )

    degraded_total, degraded_per_frame = count_gaussians(degraded_gaussians)
    print(f"  Gaussians before degradation: {total_gs:,}")
    print(f"  Gaussians after degradation:  {degraded_total:,}")
    print(f"  Removed: {total_gs - degraded_total:,} ({(1 - degraded_total/total_gs)*100:.1f}%)")
    print(f"  Per-frame after: {degraded_per_frame}")

    # ═══════════════════════════════════════════════════════════════
    # Step 4: Degraded GS Render (产物2 — 劣化后渲染)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  [4/5] Degraded GS Render (退化后渲染)")
    print(f"{'─'*60}")

    # 4a. Target trajectory render from degraded Gaussians
    print(f"  Rendering {num_target} frames from target trajectory (degraded)...")
    degraded_target_frames = render_frames_to_video(
        rasterizer, degraded_gaussians,
        [target_world2cam[i] for i in range(num_target)],
        [K_zoomed[i] for i in range(num_target)],
        [tgt_ts[i] for i in range(num_target)],
        height, width,
    )
    degraded_target_path = os.path.join(output_dir, "2_degraded_gs_target.mp4")
    save_video(degraded_target_frames, degraded_target_path, fps=16)
    print(f"  Saved: {degraded_target_path}")

    # 4b. Context frame render from degraded Gaussians
    print(f"  Rendering {num_context} context frames from original poses (degraded)...")
    degraded_context_frames = render_context_frames(
        rasterizer, degraded_gaussians,
        context_extrinsics, context_intrinsics, context_timestamps,
        height, width,
    )
    degraded_context_path = os.path.join(output_dir, "2_degraded_gs_context.mp4")
    save_video(degraded_context_frames, degraded_context_path, fps=16)
    print(f"  Saved: {degraded_context_path}")

    # Also save degraded alpha mask
    deg_rgb, deg_depth, deg_alpha = rasterizer.forward(
        degraded_gaussians,
        render_viewmats=[target_world2cam],
        render_Ks=[K_zoomed],
        render_timestamps=[tgt_ts],
        sh_degree=0, width=width, height=height,
    )
    deg_mask = (deg_alpha > alpha_threshold).float()
    deg_mask_frames = []
    for i in range(num_target):
        m = deg_mask[0, i, :, :, 0].clamp(0, 1)
        m_uint8 = (m * 255).to(torch.uint8).cpu().numpy()
        deg_mask_frames.append(Image.fromarray(m_uint8))
    deg_mask_path = os.path.join(output_dir, "2_degraded_gs_mask.mp4")
    save_video(deg_mask_frames, deg_mask_path, fps=16)
    print(f"  Saved: {deg_mask_path} (degraded alpha mask)")

    # ═══════════════════════════════════════════════════════════════
    # Step 5: Save comparison summary
    # ═══════════════════════════════════════════════════════════════

    # Count visible pixels in pristine vs degraded mask
    pristine_visible = target_mask.sum().item()
    degraded_visible = deg_mask.sum().item()
    print(f"\n{'─'*60}")
    print(f"  Visibility Comparison")
    print(f"{'─'*60}")
    print(f"  Pristine visible pixels: {pristine_visible:,}")
    print(f"  Degraded visible pixels: {degraded_visible:,}")
    if pristine_visible > 0:
        print(f"  Visibility ratio: {degraded_visible/pristine_visible*100:.1f}%")

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("NeoVerse Intermediate Products Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input frames: {num_context}\n")
        f.write(f"Resolution: {width}x{height}\n")
        f.write(f"Static scene: {static_flag}\n")
        f.write(f"Target trajectory frames: {num_target}\n\n")
        f.write(f"Degradation config:\n")
        f.write(f"  kernel_size: {kernel_size}\n")
        f.write(f"  occlusion_thresh: {occlusion_thresh}\n")
        f.write(f"  novel_view_trans: {novel_view_trans}\n\n")
        f.write(f"Gaussians before degradation: {total_gs:,}\n")
        f.write(f"Gaussians after degradation:  {degraded_total:,}\n")
        f.write(f"Removed: {total_gs - degraded_total:,} ({(1 - degraded_total/total_gs)*100:.1f}%)\n\n")
        f.write(f"Per-frame Gaussians (before): {per_frame_gs}\n")
        f.write(f"Per-frame Gaussians (after):  {degraded_per_frame}\n\n")
        f.write(f"Visibility:\n")
        f.write(f"  Pristine visible pixels: {pristine_visible:,}\n")
        f.write(f"  Degraded visible pixels: {degraded_visible:,}\n")
        if pristine_visible > 0:
            f.write(f"  Visibility ratio: {degraded_visible/pristine_visible*100:.1f}%\n\n")
        f.write(f"Output files:\n")
        f.write(f"  1_pristine_gs_target.mp4   — Pristine GS render, target trajectory\n")
        f.write(f"  1_pristine_gs_context.mp4  — Pristine GS render, context views\n")
        f.write(f"  1_pristine_gs_mask.mp4     — Pristine alpha mask\n")
        f.write(f"  2_degraded_gs_target.mp4   — Degraded GS render, target trajectory\n")
        f.write(f"  2_degraded_gs_context.mp4  — Degraded GS render, context views\n")
        f.write(f"  2_degraded_gs_mask.mp4     — Degraded alpha mask\n")
        if run_diffusion:
            f.write(f"  3_diffusion_output.mp4     — Final diffusion-enhanced output\n")
    print(f"  Saved: {summary_path}")

    # ═══════════════════════════════════════════════════════════════
    # Step 5 (optional): WAN Diffusion (产物3 — 最终高质量输出)
    # ═══════════════════════════════════════════════════════════════
    if run_diffusion:
        print(f"\n{'─'*60}")
        print(f"  [5/5] WAN Diffusion (最终高质量输出)")
        print(f"{'─'*60}")

        if cam_traj.use_first_frame:
            target_rgb[0, 0] = views["img"][0, 0].permute(1, 2, 0)
            target_mask[0, 0] = 1.0

        wrapped_data = {
            "source_views": views,
            "target_rgb": target_rgb,
            "target_depth": target_depth,
            "target_mask": target_mask,
            "target_poses": target_cam2world.unsqueeze(0),
            "target_intrs": K_zoomed.unsqueeze(0),
        }
        generated_frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed, rand_device=pipe.device,
            height=height, width=width, num_frames=num_target,
            cfg_scale=cfg_scale, num_inference_steps=num_inference_steps,
            tiled=False,
            **wrapped_data,
        )
        diffusion_path = os.path.join(output_dir, "3_diffusion_output.mp4")
        save_video(generated_frames, diffusion_path, fps=16)
        print(f"  Saved: {diffusion_path}")

    print(f"\n{'='*60}")
    print(f"  Done! All outputs saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"\n  Output files:")
    print(f"    1_pristine_gs_target.mp4   ← 产物1: 低质量GS渲染 (目标轨迹)")
    print(f"    1_pristine_gs_context.mp4  ← 产物1: 低质量GS渲染 (原始视角)")
    print(f"    1_pristine_gs_mask.mp4     ← 产物1: Alpha遮罩")
    print(f"    2_degraded_gs_target.mp4   ← 产物2: 退化后渲染 (目标轨迹)")
    print(f"    2_degraded_gs_context.mp4  ← 产物2: 退化后渲染 (原始视角)")
    print(f"    2_degraded_gs_mask.mp4     ← 产物2: 退化后Alpha遮罩")
    if run_diffusion:
        print(f"    3_diffusion_output.mp4     ← 产物3: WAN扩散模型最终输出")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract intermediate products from NeoVerse pipeline"
    )

    # --- Trajectory ---
    traj_group = parser.add_mutually_exclusive_group(required=True)
    traj_group.add_argument("--trajectory",
                            choices=["pan_left", "pan_right", "tilt_up", "tilt_down",
                                     "move_left", "move_right", "push_in", "pull_out",
                                     "boom_up", "boom_down", "orbit_left", "orbit_right",
                                     "static"],
                            help="Predefined trajectory type")
    traj_group.add_argument("--trajectory_file",
                            help="Path to JSON trajectory file")
    parser.add_argument("--angle", type=float)
    parser.add_argument("--distance", type=float)
    parser.add_argument("--orbit_radius", type=float)
    parser.add_argument("--traj_mode", choices=["relative", "global"], default="relative")
    parser.add_argument("--zoom_ratio", type=float, default=1.0)

    # --- Input/Output ---
    parser.add_argument("--input_path", required=True, help="Input video path")
    parser.add_argument("--output_dir", default="outputs/intermediates",
                        help="Output directory (default: outputs/intermediates)")
    parser.add_argument("--prompt",
                        default="A smooth video with complete scene content. Inpaint any missing regions or margins naturally to match the surrounding scene.",
                        help="Text prompt for diffusion")
    parser.add_argument("--negative_prompt", default="")

    # --- Model ---
    parser.add_argument("--model_path", default="models", help="Model directory")
    parser.add_argument("--reconstructor_path", default="models/NeoVerse/reconstructor.ckpt")
    parser.add_argument("--disable_lora", action="store_true",
                        help="Skip distilled LoRA loading")

    # --- Video ---
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--resize_mode", choices=["center_crop", "resize"], default="center_crop")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha_threshold", type=float, default=1.0)
    parser.add_argument("--static_scene", action="store_true")

    # --- Degradation ---
    parser.add_argument("--kernel_size", type=int, default=21,
                        help="Degradation kernel size: 0=culling only, >0=avg geometry filter (default: 21)")
    parser.add_argument("--occlusion_thresh", type=float, default=0.1,
                        help="Occlusion threshold for visibility check (default: 0.1)")
    parser.add_argument("--novel_view_min", type=float, default=0.01,
                        help="Min novel view translation distance (default: 0.01)")
    parser.add_argument("--novel_view_max", type=float, default=0.1,
                        help="Max novel view translation distance (default: 0.1)")

    # --- Diffusion ---
    parser.add_argument("--no_diffusion", action="store_true",
                        help="Skip diffusion step (only extract GS renders)")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low-VRAM mode")

    return parser.parse_args()


def main():
    args = parse_args()

    use_lora = not args.disable_lora
    num_inference_steps = 4 if use_lora else 50
    cfg_scale = 1.0 if use_lora else 5.0
    lora_path = os.path.join(
        args.model_path,
        "NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
    ) if use_lora else None

    run_diffusion = not args.no_diffusion

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build trajectory
    if args.trajectory:
        cam_traj = CameraTrajectory.from_predefined(
            args.trajectory,
            num_frames=args.num_frames,
            mode=args.traj_mode,
            angle=args.angle,
            distance=args.distance,
            orbit_radius=args.orbit_radius,
            zoom_ratio=args.zoom_ratio,
        )
    else:
        cam_traj = CameraTrajectory.from_json(args.trajectory_file)

    # Load model
    print(f"Loading model from {args.model_path}...")
    pipe = WanVideoNeoVersePipeline.from_pretrained(
        local_model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        lora_path=lora_path,
        lora_alpha=1.0,
        device="cuda",
        torch_dtype=torch.bfloat16,
        enable_vram_management=args.low_vram,
    )
    print("Model loaded!")

    # Load video
    print(f"Loading video from {args.input_path}...")
    images = load_video(
        args.input_path, args.num_frames,
        resolution=(args.width, args.height),
        resize_mode=args.resize_mode,
        static_scene=args.static_scene,
    )

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Run extraction
    extract_intermediates(
        pipe=pipe,
        input_video=images,
        cam_traj=cam_traj,
        output_dir=args.output_dir,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        alpha_threshold=args.alpha_threshold,
        static_flag=args.static_scene,
        seed=args.seed,
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
        kernel_size=args.kernel_size,
        occlusion_thresh=args.occlusion_thresh,
        novel_view_trans=(args.novel_view_min, args.novel_view_max),
        run_diffusion=run_diffusion,
    )

    print("Done!")


if __name__ == "__main__":
    main()
