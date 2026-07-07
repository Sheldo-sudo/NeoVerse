"""
Benchmark script for NeoVerse degradation_simulation step.

Measures the speed of the Gaussian splat degradation step (Section 3.2 of the paper)
across different kernel_size modes: culling (kernel_size=0) and average geometry
filter (kernel_size > 0).

Usage:
    CUDA_VISIBLE_DEVICES=1 python benchmark_degradation.py \
        --input_path examples/videos/robot.mp4 \
        --model_path /mnt/ssd1/wzq_models \
        --reconstructor_path /mnt/ssd1/wzq_models/NeoVerse/reconstructor.ckpt

This script loads the reconstructor (same as inference.py), runs reconstruction,
then benchmarks the degradation_simulation step without needing a full training loop.
"""

import torch
import argparse
import time
import copy
import numpy as np
from torchvision.transforms import functional as F

from diffsynth.pipelines.wan_video_neoverse import (
    WanVideoNeoVersePipeline,
    WanVideoUnit_4DPreprocesser,
)
from diffsynth.utils.auxiliary import load_video


def clone_gaussians(gaussians):
    """Deep-copy the nested list of Gaussians objects (degradation mutates them in-place)."""
    return copy.deepcopy(gaussians)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark NeoVerse degradation_simulation step"
    )
    parser.add_argument("--input_path", required=True, help="Input video path")
    parser.add_argument("--model_path", default="models", help="Model directory")
    parser.add_argument("--reconstructor_path", default="models/NeoVerse/reconstructor.ckpt")
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Number of context frames (fewer = faster load)")
    parser.add_argument("--num_benchmark_frames", type=int, default=4,
                        help="Number of novel views to generate for benchmark")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--benchmark_iterations", type=int, default=10,
                        help="Benchmark iterations per kernel_size")
    parser.add_argument("--kernel_sizes", type=str, default="0,21,51",
                        help="Comma-separated kernel sizes to benchmark")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low-VRAM mode")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # ===== Load pipeline =====
    print(f"\n[1/5] Loading pipeline from {args.model_path}...")
    t0 = time.time()
    pipe = WanVideoNeoVersePipeline.from_pretrained(
        local_model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        lora_path=None,  # No LoRA needed for degradation benchmark
        device="cuda",
        torch_dtype=torch.bfloat16,
        enable_vram_management=args.low_vram,
    )
    print(f"  Pipeline loaded in {time.time() - t0:.1f}s")

    # ===== Load video =====
    print(f"\n[2/5] Loading video from {args.input_path}...")
    images = load_video(
        args.input_path, args.num_frames,
        resolution=(args.width, args.height),
        resize_mode="center_crop",
    )
    height, width = images[0].size[1], images[0].size[0]
    print(f"  Loaded {len(images)} frames, resolution={width}x{height}")

    # ===== Build source views dict =====
    views = {
        "img": torch.stack(
            [F.to_tensor(image)[None] for image in images], dim=1
        ).to(device),
        "is_target": torch.zeros((1, len(images)), dtype=torch.bool, device=device),
        "is_static": torch.zeros((1, len(images)), dtype=torch.bool, device=device),
        "timestamp": torch.arange(0, len(images), dtype=torch.int64, device=device).unsqueeze(0),
    }

    # ===== Run reconstructor =====
    print(f"\n[3/5] Running reconstructor...")
    if pipe.vram_management_enabled:
        pipe.reconstructor.to(device)

    t0 = time.time()
    with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
        predictions = pipe.reconstructor(views, is_inference=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    recon_time = time.time() - t0
    print(f"  Reconstructor done in {recon_time:.1f}s")

    if pipe.vram_management_enabled:
        pipe.reconstructor.cpu()
        torch.cuda.empty_cache()

    # ===== Report Gaussian stats =====
    gaussians = predictions["splats"]
    context_num = (~views["is_target"]).sum().item()
    total_gs = 0
    for b_idx in range(len(gaussians)):
        for s_idx in range(min(len(gaussians[b_idx]), context_num)):
            total_gs += gaussians[b_idx][s_idx].means.shape[0]
    print(f"  Context frames: {context_num}")
    print(f"  Total Gaussians (all context frames): {total_gs:,}")
    print(f"  Avg Gaussians per frame: {total_gs // context_num:,}")

    # ===== Create preprocesser unit for novel_view_sampling =====
    novel_view_sampling_trans = [0.01, 0.1]
    occlusion_thresh = 0.1

    unit = WanVideoUnit_4DPreprocesser(
        novel_view_sampling_trans=novel_view_sampling_trans,
        novel_view_sampling_max_rot=0.0,
        culling_prob=0.3,
        kernel_size_range=[11, 51],
        occlusion_thresh=occlusion_thresh,
        alpha_thresh=0.5,
        color_thresh=[50, 100],
    )

    # ===== Generate novel view poses =====
    print(f"\n[4/5] Generating novel view poses...")
    t0 = time.time()
    novel_context_poses = unit.novel_view_sampling(
        predictions["rendered_extrinsics"][:, :context_num],
        predictions["gs_depth"].squeeze(-1),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"  Novel view sampling done in {time.time() - t0:.1f}s")
    print(f"  Novel view poses shape: {novel_context_poses.shape}")

    # ===== Benchmark degradation_simulation =====
    kernel_sizes = [int(x) for x in args.kernel_sizes.split(",")]
    results = {}

    print(f"\n[5/5] Benchmarking degradation_simulation...")
    print(f"  Image size: {width}x{height}")
    print(f"  Context frames: {context_num}")
    print(f"  Warmup: {args.warmup}, Iterations: {args.benchmark_iterations}")
    print(f"  Kernel sizes: {kernel_sizes}")
    print()

    for ks in kernel_sizes:
        mode = "culling" if ks == 0 else f"avg_filter(k={ks})"
        print(f"  Testing kernel_size={ks} ({mode})...")

        # Warmup
        for i in range(args.warmup):
            # Deep copy gaussians (degradation mutates them in-place)
            gaussians_copy = []
            for b_idx in range(len(gaussians)):
                frame_gaussians = []
                for s_idx in range(len(gaussians[b_idx])):
                    g = gaussians[b_idx][s_idx]
                    frame_gaussians.append(type(g)(
                        means=g.means.clone(),
                        harmonics=g.harmonics.clone() if g.harmonics is not None else None,
                        opacities=g.opacities.clone() if g.opacities is not None else None,
                        scales=g.scales.clone() if g.scales is not None else None,
                        rotations=g.rotations.clone() if g.rotations is not None else None,
                        confidences=g.confidences.clone() if g.confidences is not None else None,
                        timestamp=g.timestamp,
                    ))
                gaussians_copy.append(frame_gaussians)
            _ = unit.degradation_simulation(
                gaussians_copy,
                novel_context_poses,
                predictions["rendered_intrinsics"][:, :context_num],
                (height, width),
                kernel_size=ks,
                occlusion_thresh=occlusion_thresh,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed iterations
        times_ms = []
        for i in range(args.benchmark_iterations):
            gaussians_copy = []
            for b_idx in range(len(gaussians)):
                frame_gaussians = []
                for s_idx in range(len(gaussians[b_idx])):
                    g = gaussians[b_idx][s_idx]
                    frame_gaussians.append(type(g)(
                        means=g.means.clone(),
                        harmonics=g.harmonics.clone() if g.harmonics is not None else None,
                        opacities=g.opacities.clone() if g.opacities is not None else None,
                        scales=g.scales.clone() if g.scales is not None else None,
                        rotations=g.rotations.clone() if g.rotations is not None else None,
                        confidences=g.confidences.clone() if g.confidences is not None else None,
                        timestamp=g.timestamp,
                    ))
                gaussians_copy.append(frame_gaussians)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            _ = unit.degradation_simulation(
                gaussians_copy,
                novel_context_poses,
                predictions["rendered_intrinsics"][:, :context_num],
                (height, width),
                kernel_size=ks,
                occlusion_thresh=occlusion_thresh,
            )

            end.record()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))

        times = np.array(times_ms)
        results[ks] = {"mean": times.mean(), "std": times.std(), "min": times.min(), "max": times.max()}
        print(f"    mean={times.mean():.2f}ms  std={times.std():.2f}ms  "
              f"min={times.min():.2f}ms  max={times.max():.2f}ms")

    # ===== Summary =====
    print(f"\n{'='*65}")
    print(f"  DEGRADATION SIMULATION BENCHMARK RESULTS")
    print(f"{'='*65}")
    print(f"  Configuration:")
    print(f"    GPU:             {torch.cuda.get_device_name(0)}")
    print(f"    Resolution:      {width}x{height}")
    print(f"    Context frames:  {context_num}")
    print(f"    Total Gaussians: {total_gs:,}")
    print(f"    Reconstructor:   {recon_time:.1f}s")
    print(f"{'='*65}")
    print(f"  {'Kernel Size':<14} {'Mode':<22} {'Mean(ms)':<12} {'Std(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12}")
    print(f"  {'-'*65}")
    for ks in kernel_sizes:
        r = results[ks]
        mode = "Culling" if ks == 0 else f"Avg Filter (k={ks})"
        print(f"  {ks:<14} {mode:<22} {r['mean']:<12.2f} {r['std']:<12.2f} {r['min']:<12.2f} {r['max']:<12.2f}")
    print(f"{'='*65}")
    print(f"\nDone! All results above.")


if __name__ == "__main__":
    main()
