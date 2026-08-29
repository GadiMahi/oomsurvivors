#!/usr/bin/env python3
"""Cluster images by structure and hold out one cluster as an OOD proxy.

    python scripts/make_splits.py --set data.root=/kaggle/input/kla-dataset
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, load_config   # noqa: E402
from src.io_utils import list_images, load_image, pair_by_stem  # noqa: E402
from src.splits import make_splits, save_splits, structure_features  # noqa: E402


def main() -> int:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    root = Path(cfg.get_path("data.root"))
    gt_dir = root / cfg.get_path("data.gt_subdir")
    lr_dir = root / cfg.get_path("data.lr_subdir")

    pairs = pair_by_stem(gt_dir, lr_dir)
    stems = [g.stem for g, _ in pairs]
    paired = set(stems)

    # Unpaired clean images go to training wholesale. They cannot be validated
    # against - no real degraded counterpart exists, so scoring on them would
    # only measure how well the model inverts our own synthetic degradation.
    gt_only = sorted(p.stem for p in list_images(gt_dir) if p.stem not in paired)

    print(f"paired images : {len(pairs)}")
    print(f"unpaired GT   : {len(gt_only)}  -> training only, via degrade()")
    if not pairs:
        print("!! no paired images found; check data.root / subdir names")
        return 2

    print(f"\nfeaturising {len(pairs)} paired images ...")
    feats = structure_features(load_image(g) for g, _ in pairs)

    sp = make_splits(stems, feats,
                     n_clusters=cfg.get_path("split.n_clusters"),
                     ood_cluster=cfg.get_path("split.ood_cluster"),
                     val_frac=cfg.get_path("split.val_frac"),
                     seed=cfg.get_path("split.seed"),
                     max_ood_frac=cfg.get_path("split.max_ood_frac", 0.25),
                     min_ood_n=cfg.get_path("split.min_ood_n", 150),
                     gt_only_stems=gt_only)
    save_splits(sp)

    c = sp["counts"]
    print(f"\nheld-out OOD cluster: {sp['ood_cluster']}")
    print(f"cluster sizes: {sp['cluster_sizes']}")
    print(f"\n  train (real pairs)  : {c['train_paired']}")
    print(f"  train (gt-only)     : {c['train_gt_only']}")
    print(f"  val_id  (real)      : {c['val_id']}")
    print(f"  val_ood (real)      : {c['val_ood']}   <- PRIMARY metric")
    print(f"  real share of train : {c['real_share_of_train']*100:.1f}%")
    print("\nEyeball a few images per cluster before trusting this split.")
    print("Validation uses REAL pairs only - never synthetic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
