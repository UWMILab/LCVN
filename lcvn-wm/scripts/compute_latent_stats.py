"""
Compute exact float64 per-channel mean and std across pre-encoded VAE latents
and check for NaN, Inf, or extreme outlier values.
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm


def analyze_split(latent_dir: Path, outlier_threshold: float = 100.0):
    files = sorted(latent_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {latent_dir}")

    channel_sum = None
    channel_sum_sq = None
    total_elements_per_channel = 0
    total_frames = 0
    min_val = float("inf")
    max_val = float("-inf")
    nan_files = 0
    inf_files = 0
    outlier_files = 0

    for f in tqdm(files, desc=f"Stats[{latent_dir.name}]"):
        data = torch.load(f, map_location="cpu", weights_only=False)
        t = data["latents"] if isinstance(data, dict) else data
        if not isinstance(t, torch.Tensor) or t.ndim != 4:
            raise ValueError(f"Unexpected tensor shape in {f}: {getattr(t, 'shape', type(t))}")

        t64 = t.to(dtype=torch.float64)
        if torch.isnan(t64).any():
            nan_files += 1
        if torch.isinf(t64).any():
            inf_files += 1
        cur_min = float(t64.min().item())
        cur_max = float(t64.max().item())
        if cur_min < min_val:
            min_val = cur_min
        if cur_max > max_val:
            max_val = cur_max
        if max(abs(cur_min), abs(cur_max)) > outlier_threshold:
            outlier_files += 1

        # (T, C, H, W)
        c = t64.shape[1]
        if channel_sum is None:
            channel_sum = torch.zeros(c, dtype=torch.float64)
            channel_sum_sq = torch.zeros(c, dtype=torch.float64)

        channel_sum += t64.sum(dim=(0, 2, 3))
        channel_sum_sq += (t64 * t64).sum(dim=(0, 2, 3))
        total_elements_per_channel += t64.shape[0] * t64.shape[2] * t64.shape[3]
        total_frames += t64.shape[0]

    mean = channel_sum / total_elements_per_channel
    var = (channel_sum_sq / total_elements_per_channel) - (mean * mean)
    std = torch.sqrt(var.clamp(min=0.0))

    return {
        "split": latent_dir.name,
        "num_files": len(files),
        "total_frames": total_frames,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": min_val,
        "max": max_val,
        "nan_files": nan_files,
        "inf_files": inf_files,
        "outlier_files": outlier_files,
    }


def format_yaml_list(values):
    inner = ", ".join(f"[[ {v:.4f} ]]" for v in values)
    return f"[ {inner} ]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents-root", type=str, default="data/lcvn/latents")
    parser.add_argument("--splits", type=str, default="training")
    parser.add_argument("--outlier-threshold", type=float, default=100.0)
    args = parser.parse_args()

    latents_root = Path(args.latents_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    has_bad_values = False

    for split in splits:
        res = analyze_split(latents_root / split, outlier_threshold=args.outlier_threshold)
        print(f"\n=== Split: {res['split']} ===")
        print(f"  files: {res['num_files']}, total_frames: {res['total_frames']}")
        print(f"  min: {res['min']:.4f}, max: {res['max']:.4f}")
        print(f"  nan_files: {res['nan_files']}, inf_files: {res['inf_files']}, |x|>{args.outlier_threshold}: {res['outlier_files']}")
        print(f"  mean: {[round(v, 6) for v in res['mean']]}")
        print(f"  std : {[round(v, 6) for v in res['std']]}")
        if split == "training":
            print("\nYAML snippet for configurations/dataset/lcvn.yaml:")
            print(f"data_mean: {format_yaml_list(res['mean'])}")
            print(f"data_std: {format_yaml_list(res['std'])}")
        if res["nan_files"] > 0 or res["inf_files"] > 0:
            has_bad_values = True

    if has_bad_values:
        print("\nERROR: NaN or Inf values detected in latents!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
