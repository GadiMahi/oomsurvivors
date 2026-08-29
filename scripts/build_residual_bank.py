#!/usr/bin/env python3
"""Build the empirical noise-residual bank from the real paired data.

Round-2 noise is right-skewed (+0.55) with excess kurtosis ~1.4-2.1 measured on
the residual. A Gaussian generator matched to the same variance reproduces the
bulk correctly but produces roughly 6x too few 5-sigma outliers - and those are
exactly the pixels that survive restoration as visible speckle artefacts.

This script collects real residuals, normalised by their predicted sigma and
binned by clean intensity, so degrade() can resample the true distribution
instead of assuming a parametric one.

    python scripts/build_residual_bank.py --set data.root=/kaggle/input/<slug>/semicon_train_data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, load_config          # noqa: E402
from src.degrade import downsample                            # noqa: E402
from src.io_utils import pair_by_stem                         # noqa: E402
from src.noise_fit import (bank_stats, build_residual_bank,   # noqa: E402
                           estimate, save_residual_bank)
from src.transforms import load_stats, save_stats             # noqa: E402


def main() -> int:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--out", default="artifacts/residual_bank.npz")
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--max-images", type=int, default=80)
    ap.add_argument("--refit", action="store_true",
                    help="re-estimate the variance curve instead of reading stats.json")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    root = Path(cfg.get_path("data.root"))
    pairs = pair_by_stem(root / cfg.get_path("data.gt_subdir"),
                         root / cfg.get_path("data.lr_subdir"))
    if not pairs:
        print(f"!! no paired images under {root}")
        return 2
    print(f"{len(pairs)} paired images available")

    stats = load_stats()
    kernel = cfg.get_path("degrade.kernels", ["area"])[0]

    if args.refit or "noise_var_fit" not in stats:
        print("fitting variance curve ...")
        fit = estimate(pairs, downsample, kernel, max_images=args.max_images)
        a, b, c = fit["a_mult"], fit["b_poisson"], fit["c_additive"]
        stats["noise_var_fit"] = {"a_mult": a, "b_poisson": b, "c_additive": c}
        stats["meas_mult"] = float(np.sqrt(max(a, 0)))
        stats["meas_add"] = float(np.sqrt(max(c, 0)))
        save_stats(stats)
        print("  updated artifacts/stats.json")
    else:
        f = stats["noise_var_fit"]
        a, b, c = f["a_mult"], f["b_poisson"], f["c_additive"]

    print(f"variance curve: var = {a:.4e}*mu^2 + {b:.4e}*mu + {c:.4e}")

    banks = build_residual_bank(pairs, downsample, kernel, a, b, c,
                                nbins=args.nbins, max_images=args.max_images)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_residual_bank(banks, out)

    bs = bank_stats(banks)
    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print(json.dumps({k: v for k, v in bs.items() if k != "per_bin_counts"}, indent=2))
    print(f"per-bin counts: {bs['per_bin_counts']}")
    empty = [i for i, n in enumerate(bs["per_bin_counts"]) if n < 1000]
    if empty:
        print(f"!! bins with <1000 samples (will fall back to Gaussian): {empty}")
    print("\nNormalised residuals should have std ~1.0 and NON-zero skew/kurtosis.")
    print("Zero skew would mean the bank collapsed to a Gaussian - check the fit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
