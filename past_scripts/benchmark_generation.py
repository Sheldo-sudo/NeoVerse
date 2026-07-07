"""
Benchmark the diffusion generation speed of NeoVerse.

Measures how fast the Control Branch + DiT + VAE generates high-quality output
from pre-computed rendered/degraded conditions.

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmark_generation.py \
        --input_path examples/videos/robot.mp4 \
        --model_path models \
        --reconstructor_path models/NeoVerse/reconstructor.ckpt

What this script does:
    1. Load pipeline + reconstructor
    2. Run reconstruct + render ONCE (not benchmarked) → get degraded conditions
    3. Warmup: call pipe() a few times with pre-computed conditions
    4. Benchmark: call pipe() N times with CUDA events, report breakdown

The benchmark focuses on Phase 2 (diffusion generation):
    VAE encode (conditions) → Control Branch + DiT denoising → VAE decode
"""

import torch
import argparse
import time
import numpy as np
from torchvision.transforms import functional as F

from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth.utils.auxiliary import CameraTrajectory, load_video, homo_matrix_inverse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark NeoVerse diffusion generation speed"
    )
    # Input/Output
    parser.add_argument("--input_path", required=True, help="Input video path")
    parser.add_argument("--model_path", default="models", help="Model directory")
    parser.add_argument("--reconstructor_path", default="models/NeoVerse/reconstructor.ckpt")

    # Video params
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Number of input context frames")
    parser.add_argument("--num_output_frames", type=int, default=81,
                        help="Number of output frames")

    # Inference params
    parser.add_argument("--num_inference_steps", type=int, default=4,
                        help="Denoising steps (4 with LoRA, 50 without)")
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="CFG scale (1.0 with LoRA, 5.0 without)")
    parser.add_argument("--disable_lora", action="store_true",
                        help="Skip distilled LoRA (50-step inference)")

    # Benchmark params
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--benchmark_iterations", type=int, default=5,
                        help="Benchmark iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--trajectory", default="static",
                        choices=["static", "pan_left", "pan_right", "push_in", "pull_out",
                                 "orbit_left", "orbit_right"],
                        help="Camera trajectory for rendering")
    return parser.parse_args()


def precompute_conditions(pipe, images, cam_traj, static_flag, device):
    """Run reconstructor + render ONCE to get degraded conditions. Not timed."""
    height, width = images[0].size[1], images[0].size[0]

    views = {
        "img": torch.stack(
            [F.to_tensor(image)[None] for image in images], dim=1
        ).to(device),
        "is_target": torch.zeros((1, len(images)), dtype=torch.bool, device=device),
        "is_static": torch.zeros((1, len(images)), dtype=torch.bool, device=device),
        "timestamp": torch.arange(0, len(images), dtype=torch.int64, device=device).unsqueeze(0),
    }
    if static_flag:
        views["is_static"][:] = True
        views["timestamp"][:] = 0

    if pipe.vram_management_enabled:
        pipe.reconstructor.to(device)

    with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
        predictions = pipe.reconstructor(views, is_inference=True, use_motion=False)

    if pipe.vram_management_enabled:
        pipe.reconstructor.cpu()
        torch.cuda.empty_cache()

    # --- Render target trajectory ---
    gaussians = predictions["splats"]
    K = predictions["rendered_intrinsics"][0]
    input_c2w = predictions["rendered_extrinsics"][0]
    timestamps = predictions["rendered_timestamps"][0]

    if static_flag:
        K = K[:1].repeat(len(cam_traj), 1, 1)
        timestamps = timestamps[:1].repeat(len(cam_traj))

    ratio = torch.linspace(1, cam_traj.zoom_ratio, K.shape[0], device=device)
    K_zoomed = K.clone()
    K_zoomed[:, 0, 0] *= ratio
    K_zoomed[:, 1, 1] *= ratio

    target_c2w = cam_traj.c2w.to(device)
    if cam_traj.mode == "relative" and not static_flag:
        target_c2w = input_c2w @ target_c2w
    target_w2c = homo_matrix_inverse(target_c2w)

    target_rgb, target_depth, target_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
        gaussians,
        render_viewmats=[target_w2c],
        render_Ks=[K_zoomed],
        render_timestamps=[timestamps],
        sh_degree=0, width=width, height=height,
    )
    target_mask = (target_alpha > 1.0).float()

    print(f"  Reconstruction + render done.")
    print(f"  Gaussians: {sum(g.means.shape[0] for batch in gaussians for g in batch):,}")
    print(f"  Output frames: {len(cam_traj)}")
    print(f"  target_rgb:   {list(target_rgb.shape)}")
    print(f"  target_depth: {list(target_depth.shape)}")
    print(f"  target_mask:  {list(target_mask.shape)}")

    return {
        "target_rgb": target_rgb,
        "target_depth": target_depth,
        "target_mask": target_mask,
        "target_poses": target_c2w.unsqueeze(0),
        "target_intrs": K_zoomed.unsqueeze(0),
        "views": views,
    }


def benchmark_generation(pipe, conditions, args):
    """Benchmark pipe() call (diffusion generation only)."""
    prompt = "A smooth video with complete scene content."
    negative_prompt = ""

    shared = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=args.seed,
        rand_device=pipe.device,
        height=args.height,
        width=args.width,
        num_frames=args.num_output_frames,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        tiled=False,
        target_rgb=conditions["target_rgb"],
        target_depth=conditions["target_depth"],
        target_mask=conditions["target_mask"],
        target_poses=conditions["target_poses"],
        target_intrs=conditions["target_intrs"],
    )

    # Warmup
    print(f"\n  Warming up ({args.warmup} iterations)...")
    for i in range(args.warmup):
        _ = pipe(**shared)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark
    print(f"  Benchmarking ({args.benchmark_iterations} iterations)...")
    times_s = []
    for i in range(args.benchmark_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        _ = pipe(**shared)

        end.record()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_s.append(start.elapsed_time(end) / 1000.0)

    times = np.array(times_s)
    return times


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ===== Load pipeline =====
    use_lora = not args.disable_lora
    num_inference_steps = 4 if use_lora else 50
    cfg_scale = 1.0 if use_lora else 5.0
    args.num_inference_steps = num_inference_steps
    args.cfg_scale = cfg_scale

    import os
    lora_path = os.path.join(
        args.model_path,
        "NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
    ) if use_lora else None

    print(f"\n[1/4] Loading pipeline from {args.model_path}...")
    print(f"  LoRA: {'enabled (4-step)' if use_lora else 'disabled (50-step)'}")
    t0 = time.time()
    pipe = WanVideoNeoVersePipeline.from_pretrained(
        local_model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        lora_path=lora_path,
        lora_alpha=1.0,
        device="cuda",
        torch_dtype=torch.bfloat16,
        enable_vram_management=args.low_vram,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ===== Load video =====
    print(f"\n[2/4] Loading video from {args.input_path}...")
    images = load_video(
        args.input_path, args.num_frames,
        resolution=(args.width, args.height),
        resize_mode="center_crop",
    )
    print(f"  {len(images)} frames, {images[0].size[0]}x{images[0].size[1]}")

    # ===== Build trajectory =====
    cam_traj = CameraTrajectory.from_predefined(
        args.trajectory,
        num_frames=args.num_output_frames,
        mode="relative",
    )
    print(f"  Trajectory: {cam_traj.name} ({args.num_output_frames} frames)")

    # ===== Precompute conditions (not benchmarked) =====
    print(f"\n[3/4] Running reconstructor + render (preparation, not timed)...")
    t0 = time.time()
    conditions = precompute_conditions(pipe, images, cam_traj,
                                       static_flag=(args.trajectory == "static"),
                                       device=device)
    prep_time = time.time() - t0
    print(f"  Preparation done in {prep_time:.1f}s")

    # ===== Benchmark diffusion generation =====
    print(f"\n[4/4] Benchmarking diffusion generation...")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Output frames: {args.num_output_frames}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  CFG scale: {args.cfg_scale}")

    times = benchmark_generation(pipe, conditions, args)

    # ===== Report =====
    print(f"\n{'='*65}")
    print(f"  DIFFUSION GENERATION BENCHMARK RESULTS")
    print(f"{'='*65}")
    print(f"  Configuration:")
    print(f"    GPU:                 {torch.cuda.get_device_name(0)}")
    print(f"    Resolution:          {args.width}x{args.height}")
    print(f"    Output frames:       {args.num_output_frames}")
    print(f"    Inference steps:     {args.num_inference_steps}")
    print(f"    CFG scale:           {args.cfg_scale}")
    print(f"    LoRA:                {'yes' if use_lora else 'no'}")
    print(f"    Preparation time:    {prep_time:.1f}s")
    print(f"    Warmup iters:        {args.warmup}")
    print(f"    Benchmark iters:     {args.benchmark_iterations}")
    print(f"{'='*65}")
    print(f"  Generation Time:")
    print(f"    Mean:  {times.mean():.2f}s  ({times.mean()*1000:.0f}ms)")
    print(f"    Std:   {times.std():.2f}s  ({times.std()*1000:.0f}ms)")
    print(f"    Min:   {times.min():.2f}s  ({times.min()*1000:.0f}ms)")
    print(f"    Max:   {times.max():.2f}s  ({times.max()*1000:.0f}ms)")
    if args.num_output_frames > 0:
        fps = args.num_output_frames / times.mean()
        print(f"    FPS:   {fps:.1f} frames/s")
    print(f"{'='*65}")
    print(f"  Total pipeline (prep + generation): {prep_time + times.mean():.1f}s")
    print(f"{'='*65}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
