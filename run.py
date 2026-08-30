#!/usr/bin/env python3
"""
Standalone evaluation script — KLA AI Hackathon submission.
Team: OOM Survivors

Usage:
    python run.py <input_dir> <output_dir>

Behavior:
  * Reads every .npy file in <input_dir> (degraded / low-res images).
  * Restores each one with the trained NAFNet super-resolution model.
  * Writes one restored .npy file per input to <output_dir>, using the
    SAME filename as the corresponding input.
  * Creates <output_dir> if it does not already exist.
  * Outputs are float32 grayscale arrays, shape (H, W) or (H, W, 1)
    (mirrors the input's ndim), values clipped to [0, 1], no NaN/Inf.
  * Runs fully offline: no internet access, no API keys, no manual steps.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make the local `src` package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model import build_model  # noqa: E402
from src.tta import tta_forward   # noqa: E402

SCALE = 2           # fixed by the trained checkpoint (512<->256 or 256<->128 SR factor)
PAD_MULTIPLE = 2    # NAFNet_UNet has one stride-2 down/up level -> pad H,W to a multiple of 2

# Look for weights in weights/ (for final submission) with a fallback to artifacts/ (for local Kaggle testing)
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "weights" / "best_nafnet.pt"
FALLBACK_WEIGHTS = BASE_DIR / "artifacts" / "best_nafnet.pt"


def pad_to_multiple(x: torch.Tensor, m: int):
    """Reflect-pad the last two dims of x up to the next multiple of m."""
    if m <= 1:
        return x, (0, 0)
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (ph, pw)


def load_npy(path: Path):
    """Load a .npy file and return (2D float32 array, original_ndim)."""
    arr = np.load(path).astype(np.float32)
    orig_ndim = arr.ndim
    if arr.ndim == 3:
        # (H, W, 1) -> (H, W). Also tolerate an accidental (H, W, C>1) by averaging.
        arr = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
    elif arr.ndim != 2:
        raise ValueError(
            f"Unexpected array shape {arr.shape} for {path.name}; expected (H,W) or (H,W,1)"
        )
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr, orig_ndim


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=str, help="Directory containing degraded .npy images")
    ap.add_argument("output_dir", type=str, help="Directory to write restored .npy images")
    ap.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    ap.add_argument("--weights", type=str, default=None,
                    help="Checkpoint path. Defaults to weights/best_nafnet.pt, "
                         "then artifacts/best_nafnet.pt")
    ap.add_argument("--dim", type=int, default=None,
                    help="Model width. Read from the checkpoint when omitted.")
    ap.add_argument("--tta", type=int, default=1, choices=[1, 4, 8],
                    help="Test-time augmentation: 1 = off, 4 = rotations, "
                         "8 = full dihedral group. Cost is linear; throughput "
                         "is scored, so measure before enabling.")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.is_dir():
        print(f"ERROR: input directory not found: {in_dir}")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = sorted(in_dir.glob("*.npy"))
    if not files:
        print(f"No .npy files found in {in_dir}")
        return 2

    # Determine which weights file to use
    if args.weights:
        active_weights_path = Path(args.weights)
    else:
        active_weights_path = WEIGHTS_PATH
        if not active_weights_path.exists() and FALLBACK_WEIGHTS.exists():
            active_weights_path = FALLBACK_WEIGHTS

    state_dict, dim = None, args.dim
    if active_weights_path.exists():
        sd = torch.load(active_weights_path, map_location=device)
        state_dict = (sd["model"] if isinstance(sd, dict) and "model" in sd
                      and hasattr(sd["model"], "keys") else sd)
        if dim is None:
            # Infer width from the checkpoint so a dim=96 model loads correctly
            # without needing the flag. intro.weight is (dim, in_ch, 3, 3).
            cfg_blob = sd.get("config") if isinstance(sd, dict) else None
            if isinstance(cfg_blob, dict):
                dim = (cfg_blob.get("model") or {}).get("dim")
            if dim is None and "intro.weight" in state_dict:
                dim = int(state_dict["intro.weight"].shape[0])
    dim = dim or 64

    model = build_model("nafnet", scale=SCALE, dim=dim).to(device).eval()

    if state_dict is not None:
        model.load_state_dict(state_dict)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"Loaded weights from: {active_weights_path}  (dim={dim}, {n_par/1e6:.2f}M params)")
    else:
        print(f"WARNING: no weights at {active_weights_path}. Running UNINITIALIZED!")

    # Load everything up front and group by shape so batches stay rectangular.
    t_start = time.perf_counter()
    cache: dict[Path, tuple[np.ndarray, int]] = {}
    by_shape: dict[tuple, list[Path]] = defaultdict(list)
    for f in files:
        arr, orig_ndim = load_npy(f)
        cache[f] = (arr, orig_ndim)
        by_shape[arr.shape].append(f)

    n_done = 0
    with torch.no_grad():
        for shape, group in by_shape.items():
            for i in range(0, len(group), args.batch_size):
                chunk = group[i:i + args.batch_size]
                batch_np = np.stack([cache[f][0] for f in chunk])[:, None, :, :]  # (B,1,H,W)
                x = torch.from_numpy(batch_np).to(device, non_blocking=True)

                xp, (ph, pw) = pad_to_multiple(x, PAD_MULTIPLE)
                y = tta_forward(model, xp, args.tta, clamp=None) if args.tta > 1 else model(xp)
                if ph or pw:
                    y = y[..., : y.shape[-2] - ph * SCALE, : y.shape[-1] - pw * SCALE]

                y = torch.clamp(y, 0.0, 1.0)
                y_np = y.float().cpu().numpy()[:, 0]  # (B,H,W)
                y_np = np.nan_to_num(y_np, nan=0.0, posinf=1.0, neginf=0.0)

                for f, out_arr in zip(chunk, y_np):
                    _, orig_ndim = cache[f]
                    save_arr = out_arr[..., None] if orig_ndim == 3 else out_arr
                    np.save(out_dir / f.name, save_arr.astype(np.float32))
                    n_done += 1

    elapsed = time.perf_counter() - t_start
    print(f"\nRestored {n_done}/{len(files)} images -> {out_dir}")
    print(f"device={device}  elapsed={elapsed:.3f}s  ({1000 * elapsed / max(n_done, 1):.2f} ms/img)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())