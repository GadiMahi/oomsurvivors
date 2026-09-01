#!/usr/bin/env python3
"""Does SEM texture repeat within a single frame enough to denoise with?

THE QUESTION
------------
Our measured ceiling is information-theoretic: one noisy observation, and the
noise on it is irreducible because there is no second measurement to average
against. Frame averaging would solve it, but frame averaging costs scan time,
which is the whole constraint.

Non-local self-similarity is the one legitimate way around that argument. If the
same texture patch recurs 50 times inside a single frame, and the noise on each
instance is independent, averaging those 50 instances cuts noise by sqrt(50) -
with no second acquisition. It is the same statistical trick as frame averaging,
paid for in computation rather than scan time. BM3D exploits this classically;
non-local attention blocks exploit it in networks.

Our model cannot. It is convolutional with a 30-60px receptive field, so it
physically cannot see that a patch in one corner matches a patch in the other.

Before building anything, this script measures whether the redundancy is there.

HOW IT AVOIDS FOOLING ITSELF
----------------------------
Three controls, because "patches in an image look similar to each other" is
trivially true and means nothing on its own:

  spatial exclusion  Matches must come from at least `--exclude` pixels away.
                     Without this we would be measuring local smoothness -
                     adjacent patches overlap, so averaging them is just a blur,
                     which is exactly what the model already does.

  local control      Average the k spatially NEAREST patches instead. This is
                     the fair comparison: same amount of averaging, but local.
                     Non-local has to beat this to be worth building.

  cross-image null   Average k patches drawn from a DIFFERENT image. Any two
                     patches share global statistics (brightness, contrast), so
                     this is the floor that a "match" has to clear to be real.

ORACLE VS PRACTICAL
-------------------
Two different questions get conflated in this area, so both are measured:

  oracle     Find matches using the CLEAN image, then average the corresponding
             NOISY patches. Answers: does the structure genuinely repeat?
  practical  Find matches using the NOISY image, as you would at test time.
             Answers: can we still find them once noise is in the way?

The gap between the two is the retrieval penalty. Structure can repeat
perfectly and still be unusable if noise destroys our ability to locate it.

READING THE RESULT
------------------
  oracle >> single        redundancy exists
  practical ~ oracle      noise does not prevent retrieval
  practical >> local      non-local beats what the model already does  <-- the
                          only condition that justifies building it
  null <= single          the controls are working

Usage:
    python scripts/measure_recurrence.py --set data.root=$DATA
    python scripts/measure_recurrence.py --set data.root=$DATA \\
        --images 30 --patch 8 --k 16 --queries 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.degrade import downsample  # noqa: E402


# ------------------------------------------------------------------ patch utils

def extract_patches(img: np.ndarray, patch: int, stride: int):
    """Return (patches (N, patch*patch), positions (N, 2)) on a strided grid."""
    from numpy.lib.stride_tricks import sliding_window_view

    win = sliding_window_view(img, (patch, patch))          # (H-p+1, W-p+1, p, p)
    win = win[::stride, ::stride]
    gh, gw = win.shape[:2]
    rows, cols = np.mgrid[0:gh, 0:gw]
    pos = np.stack([rows.ravel() * stride, cols.ravel() * stride], axis=1)
    return win.reshape(gh * gw, patch * patch).astype(np.float32), pos.astype(np.int32)


def knn(queries, bank, pos_q, pos_b, k, exclude_radius):
    """k nearest patches by L2, refusing any match within `exclude_radius` px.

    The exclusion is the whole point: without it the nearest neighbours are the
    overlapping patches immediately around the query, and averaging those is
    a blur wearing a disguise.
    """
    # |a-b|^2 = |a|^2 - 2ab + |b|^2, computed as a matrix product.
    d = (np.einsum("ij,ij->i", queries, queries)[:, None]
         - 2.0 * queries @ bank.T
         + np.einsum("ij,ij->i", bank, bank)[None, :])

    # Chebyshev distance in pixels between every query and every candidate.
    dr = np.abs(pos_q[:, None, 0] - pos_b[None, :, 0])
    dc = np.abs(pos_q[:, None, 1] - pos_b[None, :, 1])
    d[np.maximum(dr, dc) < exclude_radius] = np.inf

    idx = np.argpartition(d, k, axis=1)[:, :k]
    took = np.take_along_axis(d, idx, axis=1)
    order = np.argsort(took, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(took, order, axis=1)


def nearest_spatial(pos_q, pos_b, k, min_radius=1):
    """The k spatially closest patches - the local-averaging control."""
    dr = pos_q[:, None, 0].astype(np.float32) - pos_b[None, :, 0]
    dc = pos_q[:, None, 1].astype(np.float32) - pos_b[None, :, 1]
    d = np.hypot(dr, dc)
    d[d < min_radius] = np.inf          # refuse the query patch itself
    return np.argpartition(d, k, axis=1)[:, :k]


def psnr_from_mse(mse):
    return float(10.0 * np.log10(1.0 / max(float(mse), 1e-12)))


# ------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides k=v")
    ap.add_argument("--images", type=int, default=25)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--k", type=int, default=16, help="neighbours averaged")
    ap.add_argument("--queries", type=int, default=250, help="query patches per image")
    ap.add_argument("--exclude", type=int, default=12,
                    help="minimum pixel distance for a match to count. Must "
                         "exceed the patch size or you are measuring blur.")
    ap.add_argument("--split", default="val_ood")
    ap.add_argument("--out", default="artifacts/recurrence.json")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if args.exclude <= args.patch:
        print(f"WARNING: --exclude {args.exclude} <= --patch {args.patch}. "
              "Matches may overlap the query, which inflates the result.")

    cfg = load_config(overrides=args.set)
    root = Path(cfg.get_path("data.root"))
    gt_dir = root / cfg.get_path("data.gt_subdir", "GT")
    lr_dir = root / cfg.get_path("data.lr_subdir", "NoisyLR")
    kernel = cfg.get_path("degrade.kernels", ["gauss:0.6"])[0]
    if isinstance(kernel, str) and not kernel.startswith("gauss"):
        kernel = "gauss:0.6"

    sp_path = Path("artifacts/splits.json")
    if sp_path.exists():
        stems = json.load(open(sp_path))[args.split][:args.images]
    else:
        stems = [p.stem for p in sorted(lr_dir.glob("*.npy"))[:args.images]]
    print(f"images={len(stems)}  patch={args.patch}  k={args.k}  "
          f"exclude={args.exclude}px  kernel={kernel}\n")

    rng = np.random.default_rng(args.seed)

    def load_pair(stem):
        gt = np.load(gt_dir / f"{stem}.npy").astype(np.float32)
        lr = np.load(lr_dir / f"{stem}.npy").astype(np.float32)
        if gt.ndim == 3:
            gt = gt[..., 0]
        if lr.ndim == 3:
            lr = lr[..., 0]
        # Clean reference at LR resolution: what NoisyLR would be with no noise.
        # Uses the kernel we identified, so this is not an assumption.
        clean = downsample(gt, kernel, scale=2).astype(np.float32)
        h = min(clean.shape[0], lr.shape[0])
        w = min(clean.shape[1], lr.shape[1])
        return lr[:h, :w], clean[:h, :w]

    variants = ["single", "oracle", "practical", "local", "null"]
    sse = {v: 0.0 for v in variants}
    n_px = 0
    match_dists = []
    t0 = time.perf_counter()

    prev_noisy = None
    for i, stem in enumerate(stems):
        try:
            noisy, clean = load_pair(stem)
        except Exception as e:
            print(f"  skip {stem}: {type(e).__name__}")
            continue

        pn, pos = extract_patches(noisy, args.patch, args.stride)
        pc, _ = extract_patches(clean, args.patch, args.stride)

        nq = min(args.queries, pn.shape[0])
        q = rng.choice(pn.shape[0], nq, replace=False)

        # practical: search with the noisy patch, as at test time
        idx_p, d_p = knn(pn[q], pn, pos[q], pos, args.k, args.exclude)
        # oracle: search with the clean patch, then average the NOISY matches
        idx_o, _ = knn(pc[q], pc, pos[q], pos, args.k, args.exclude)
        # local control: same k, but spatially nearest
        idx_l = nearest_spatial(pos[q], pos, args.k, min_radius=1)

        est = {
            "single": pn[q],
            "practical": pn[idx_p].mean(axis=1),
            "oracle": pn[idx_o].mean(axis=1),
            "local": pn[idx_l].mean(axis=1),
        }
        # cross-image null: k patches from a different image entirely
        if prev_noisy is not None:
            po, _ = extract_patches(prev_noisy, args.patch, args.stride)
            pick = rng.choice(po.shape[0], (nq, args.k), replace=True)
            est["null"] = po[pick].mean(axis=1)

        target = pc[q]
        for v, e in est.items():
            sse[v] += float(np.sum((e - target) ** 2))
        n_px += target.size
        match_dists.append(d_p / max(args.patch * args.patch, 1))
        prev_noisy = noisy

        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(stems)}")

    if n_px == 0:
        print("no images processed")
        return 2

    # `null` starts one image late, so normalise it against its own count.
    res = {}
    for v in variants:
        cnt = n_px if v != "null" else n_px * (len(stems) - 1) / max(len(stems), 1)
        res[v] = {"mse": sse[v] / cnt, "psnr": psnr_from_mse(sse[v] / cnt)}

    single_mse = res["single"]["mse"]
    dt = time.perf_counter() - t0

    print(f"\n{'variant':>12} {'PSNR':>8} {'vs single':>11} {'noise x-reduction':>18}")
    print("-" * 53)
    labels = {"single": "single patch", "local": "local avg (k)",
              "null": "cross-image", "practical": "non-local", "oracle": "non-local*"}
    for v in ["single", "local", "null", "practical", "oracle"]:
        r = res[v]
        print(f"{labels[v]:>12} {r['psnr']:>8.2f} "
              f"{r['psnr'] - res['single']['psnr']:>+11.2f} "
              f"{single_mse / max(r['mse'], 1e-12):>18.2f}")
    print("\n* oracle = matches located using the clean image (upper bound)")

    md = np.concatenate(match_dists) if match_dists else np.zeros((1, 1))
    print(f"\nmean per-pixel sq. distance to the {args.k} matches: "
          f"{float(md.mean()):.5f}  (nearest {float(md[:, 0].mean()):.5f}, "
          f"furthest {float(md[:, -1].mean()):.5f})")

    # ------------------------------------------------------------- verdict
    gain_vs_local = res["practical"]["psnr"] - res["local"]["psnr"]
    gain_vs_single = res["practical"]["psnr"] - res["single"]["psnr"]
    retrieval_gap = res["oracle"]["psnr"] - res["practical"]["psnr"]
    null_ok = res["null"]["psnr"] <= res["single"]["psnr"] + 0.5

    print("\n" + "=" * 53)
    print(f"controls valid (null no better than single) : {'yes' if null_ok else 'NO'}")
    print(f"redundancy exists (oracle - single)         : {res['oracle']['psnr'] - res['single']['psnr']:+.2f} dB")
    print(f"retrieval penalty (oracle - practical)      : {retrieval_gap:+.2f} dB")
    print(f"helps at all      (practical - single)      : {gain_vs_single:+.2f} dB")
    print(f"beats local averaging (practical - local)   : {gain_vs_local:+.2f} dB")

    # Both gates are required, and the second alone is NOT sufficient. On
    # validation against synthetic images with known answers, a pure-noise
    # image scored practical - local = +3.07 dB purely because local averaging
    # was even more destructive than non-local; practical was 6.35 dB WORSE
    # than doing nothing. Without the vs-single gate that reads as a success.
    HELP_DB, BEAT_LOCAL_DB = 0.3, 1.5

    print("\nVERDICT")
    if not null_ok:
        print("  Controls failed. The cross-image null scores as well as a real")
        print("  match, so patch distance is not measuring structure here.")
        print("  Do not act on the other numbers.")
    elif gain_vs_single < HELP_DB:
        print(f"  NOT WORTH IT. Non-local averaging is {gain_vs_single:+.2f} dB against")
        print("  simply keeping the noisy patch, so the matches it finds are not")
        print("  the same structure. Averaging them destroys detail rather than")
        print("  cancelling noise.")
    elif gain_vs_local > BEAT_LOCAL_DB:
        print(f"  WORTH BUILDING. Non-local beats doing nothing by {gain_vs_single:.2f} dB")
        print(f"  and beats local averaging by {gain_vs_local:.2f} dB, so there is real")
        print("  information in distant patches that a purely convolutional model")
        print("  cannot reach. Add a non-local block at the bottleneck, where the")
        print("  spatial dimensions are already small enough to afford attention.")
    else:
        print(f"  MARGINAL. Non-local beats a single patch by {gain_vs_single:.2f} dB, but")
        print(f"  only beats local averaging by {gain_vs_local:.2f} dB, so most of the")
        print("  gain is plain smoothing - which the model already does, for free,")
        print("  with a convolution. Not worth the inference cost of attention.")
    if retrieval_gap > 1.0:
        print(f"\n  Note: the {retrieval_gap:.2f} dB retrieval penalty is large. Structure")
        print("  repeats more than we can currently find under noise; matching on")
        print("  a pre-denoised image would recover part of that gap.")

    print(f"\n{dt:.1f}s total")

    payload = {"config": vars(args), "results": res,
               "gain_vs_local_db": gain_vs_local,
               "gain_vs_single_db": gain_vs_single,
               "retrieval_penalty_db": retrieval_gap,
               "controls_valid": bool(null_ok),
               "n_images": len(stems), "seconds": dt}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
