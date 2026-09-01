#!/usr/bin/env python3
"""Verify the model runs at every input size the evaluator might hand it.

WHY THIS MATTERS
----------------
Every training pair is 256 from 128. The KLA specification says evaluation may
include 512x512 ground truths, which means 256x256 inputs - and possibly 512
inputs if they hand over full frames. That path has never been executed. A crash
or a shape error there fails the submission outright, independently of how good
the model is.

This checks, for each size:
  * the forward pass completes at all
  * the output is exactly 2x the input
  * no NaN or Inf appears
  * peak GPU memory stays within budget
  * throughput, since inference time is scored

Usage:
    python scripts/check_sizes.py --weights weights/best_nafnet.pt
    python scripts/check_sizes.py --weights ... --sizes 128 256 512 1024 --tta 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import build_model, is_legacy_state_dict, remap_legacy_state_dict  # noqa: E402
from src.tta import tta_forward  # noqa: E402


def load_model(weights: Path, device):
    sd = torch.load(weights, map_location=device)
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd

    dim = None
    cfg_blob = sd.get("config") if isinstance(sd, dict) else None
    if isinstance(cfg_blob, dict):
        dim = (cfg_blob.get("model") or {}).get("dim")
    if dim is None and "intro.weight" in state:
        dim = int(state["intro.weight"].shape[0])
    dim = dim or 64

    levels, blocks, mblocks = None, 2, 2
    if is_legacy_state_dict(state):
        state = remap_legacy_state_dict(state)
        levels, blocks, mblocks = 1, 2, 4
    else:
        levels = 1 + max((int(k.split(".")[1]) for k in state
                          if k.startswith("encoders.")), default=0)

    non_local = any(".to_kv." in k for k in state)
    if non_local:
        mblocks = max(mblocks, sum(1 for k in state if k.startswith("middle.")
                                   and k.endswith(".conv1.weight")))

    model = build_model("nafnet", scale=2, dim=dim, levels=levels,
                        blocks=blocks, middle_blocks=mblocks,
                        non_local=non_local).to(device).eval()
    model.load_state_dict(state)
    if non_local:
        print("non-local attention present - watch the 512 memory row, "
              "attention cost grows faster than convolution")

    vst_cfg = sd.get("vst") if isinstance(sd, dict) else None
    if vst_cfg:
        from src.vst import VST
        t = VST.from_dict(vst_cfg)
        inner = model

        class W(torch.nn.Module):
            def forward(self, x):
                return t.inverse(inner(t.forward(x)))

        model = W().to(device).eval()
        print(f"checkpoint uses VST: {t}")

    return model, dim, levels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="weights/best_nafnet.pt")
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--tta", type=int, default=1, choices=[1, 2, 4, 8])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5,
                    help="Timed iterations after warm-up")
    args = ap.parse_args()

    w = Path(args.weights)
    if not w.exists():
        print(f"ERROR: no checkpoint at {w}")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dim, levels = load_model(w, device)
    print(f"model: dim={dim} levels={levels} on {device}, tta={args.tta}, "
          f"batch={args.batch}\n")

    hdr = f"{'input':>12} {'output':>12} {'ok':>5} {'range':>18} {'ms/img':>9} {'peak MB':>9}"
    print(hdr)
    print("-" * len(hdr))

    rng = np.random.default_rng(1337)
    failures = []
    for s in args.sizes:
        # Realistic input: NoisyLR is roughly [0,1] with genuine out-of-range
        # excursions, so feeding a plain uniform would be an easier test.
        x = rng.normal(0.5, 0.2, (args.batch, 1, s, s)).astype(np.float32)
        x = torch.from_numpy(x).to(device)

        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                y = tta_forward(model, x, args.tta) if args.tta > 1 else model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

                t0 = time.perf_counter()
                for _ in range(args.repeats):
                    y = tta_forward(model, x, args.tta) if args.tta > 1 else model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / (args.repeats * args.batch)

            peak = (torch.cuda.max_memory_allocated() / 1e6
                    if device.type == "cuda" else float("nan"))
            exp = (args.batch, 1, s * 2, s * 2)
            shape_ok = tuple(y.shape) == exp
            finite = bool(torch.isfinite(y).all())
            ok = shape_ok and finite

            print(f"{s:>6}x{s:<5} {y.shape[-2]:>5}x{y.shape[-1]:<6} "
                  f"{'PASS' if ok else 'FAIL':>5} "
                  f"[{float(y.min()):+.3f},{float(y.max()):+.3f}]".ljust(58)
                  + f"{1000 * dt:>9.2f} {peak:>9.1f}")

            if not shape_ok:
                failures.append(f"{s}: got {tuple(y.shape)}, expected {exp}")
            if not finite:
                failures.append(f"{s}: output contains NaN or Inf")

        except RuntimeError as e:
            msg = str(e).split("\n")[0][:60]
            print(f"{s:>6}x{s:<5} {'-':>12} {'FAIL':>5}  {msg}")
            failures.append(f"{s}: {msg}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        print("\nThe model is fully convolutional, so a shape failure usually "
              "means the input is not divisible by 2**levels. run.py reflect-pads "
              "for this; if a size fails here but works through run.py, that is "
              "why. A memory failure at 512 needs tiled inference.")
        return 1

    print("all sizes passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
