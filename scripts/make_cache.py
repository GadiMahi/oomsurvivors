#!/usr/bin/env python3
"""Decode the dataset once into memmaps, then measure dataloader throughput.

    python scripts/make_cache.py --set data.root=/kaggle/input/kla-dataset \
                                      cache.dir=/kaggle/working/cache
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader  # noqa: E402

from src.cache import build_cache        # noqa: E402
from src.config import add_config_args, load_config  # noqa: E402
from src.dataset import (RestorationDataset, degrade_cfg_from_stats,  # noqa: E402
                         estimate_grad_threshold)
from src.io_utils import list_images, pair_by_stem    # noqa: E402


def main() -> int:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    root = Path(cfg.get_path("data.root"))
    cache_dir = Path(cfg.get_path("cache.dir"))

    gt_dir = root / cfg.get_path("data.gt_subdir")
    lr_dir = root / cfg.get_path("data.lr_subdir")

    if not args.skip_build:
        pairs = pair_by_stem(gt_dir, lr_dir)
        paired_stems = {p.stem for p, _ in pairs}
        # Round 2 ships far more clean images than pairs. Everything unpaired is
        # still usable through degrade(), so cache it as a separate pool.
        gt_only = sorted(p for p in list_images(gt_dir) if p.stem not in paired_stems)
        print(f"caching {len(pairs)} pairs + {len(gt_only)} unpaired GT -> {cache_dir}")
        if gt_only:
            print(f"  unpaired GT is {len(gt_only)/(len(gt_only)+len(pairs))*100:.1f}% "
                  f"of clean images and is reachable ONLY through synthesis")
        build_cache(pairs, cache_dir,
                    gt_dtype=cfg.get_path("cache.gt_dtype"),
                    lr_dtype=cfg.get_path("cache.lr_dtype"),
                    gt_only=gt_only)

    thr = estimate_grad_threshold(cache_dir,
                                  percentile=cfg.get_path("dataset.grad_percentile"),
                                  lr_patch=cfg.get_path("dataset.lr_patch"),
                                  scale=cfg.get_path("dataset.scale"))
    print(f"grad_thresh (GT p{cfg.get_path('dataset.grad_percentile')}) = {thr:.6f}")

    bank_path = Path(cfg.get_path("degrade.residual_bank", "artifacts/residual_bank.npz"))
    if not bank_path.exists():
        print(f"!! {bank_path} missing - run scripts/build_residual_bank.py first.")
        print("   Falling back to Gaussian noise, which produces ~6x too few 5-sigma outliers.")
        bank_path = None

    ds = RestorationDataset(cache_dir,
                            lr_patch=cfg.get_path("dataset.lr_patch"),
                            scale=cfg.get_path("dataset.scale"),
                            grad_thresh=thr,
                            crop_tries=cfg.get_path("dataset.crop_tries"),
                            degrade_cfg=degrade_cfg_from_stats(
                                width=cfg.get_path("degrade.width", 1.0),
                                jitter=cfg.get_path("degrade.jitter", 0.30),
                                residual_bank=bank_path),
                            real_frac=cfg.get_path("dataset.real_frac"))
    print(f"dataset composition: {ds.composition()}")
    dl = DataLoader(ds, batch_size=cfg.get_path("train.batch_size"), shuffle=True,
                    num_workers=cfg.get_path("train.num_workers"),
                    pin_memory=True, persistent_workers=True, drop_last=True)

    n, t0 = 0, time.perf_counter()
    for i, b in enumerate(dl):
        n += b["lr"].shape[0]
        if i >= 50:
            break
    dt = time.perf_counter() - t0
    print(f"dataloader: {n / dt:,.0f} samples/s over {n} samples "
          f"(workers={cfg.get_path('train.num_workers')})")
    print("If this is below the model's step rate, the GPU is starving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
