#!/usr/bin/env python3
"""Is the non-local block doing anything, and is it costing texture?

Two questions that look the same from a PSNR table but need opposite responses:

  A) gamma stayed near zero  -> the block is inert. Smoothing came from
     something else, and the experiment never actually ran.
  B) gamma trained large     -> the block is active and averaging. The extra
     smoothing IS the mechanism working as designed, and the approach is
     wrong for this problem rather than wrongly implemented.

Also reports high-frequency retention, which is the measurement that should
have been used to judge non-local averaging in the first place. PSNR rewards
smoothing, so a method that blurs can score well while destroying exactly the
texture we are trying to keep.

Usage:
    python scripts/inspect_nonlocal.py --weights runs/nonlocal/best_nafnet.pt
    python scripts/inspect_nonlocal.py --weights A.pt --compare B.pt \\
        --set data.root=$DATA
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.model import build_model, is_legacy_state_dict, remap_legacy_state_dict  # noqa: E402


def load_model(path, device):
    sd = torch.load(path, map_location=device)
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd

    dim = None
    blob = sd.get("config") if isinstance(sd, dict) else None
    if isinstance(blob, dict):
        dim = (blob.get("model") or {}).get("dim")
    if dim is None and "intro.weight" in state:
        dim = int(state["intro.weight"].shape[0])

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

    m = build_model("nafnet", scale=2, dim=dim or 64, levels=levels,
                    blocks=blocks, middle_blocks=mblocks,
                    non_local=non_local).to(device).eval()
    m.load_state_dict(state)
    return m, state, non_local


def report_gamma(state):
    """The zero-initialised residual scale tells us whether training used it."""
    gammas = {k: v for k, v in state.items()
              if k.endswith("gamma") and "middle" in k}
    if not gammas:
        print("  no non-local gamma found - this checkpoint has no attention block")
        return None
    print(f"  {'parameter':<28} {'mean|g|':>10} {'max|g|':>10} {'frac>0.01':>10}")
    peak = 0.0
    for k, v in gammas.items():
        a = v.abs().flatten()
        peak = max(peak, float(a.max()))
        print(f"  {k:<28} {float(a.mean()):>10.5f} {float(a.max()):>10.5f} "
              f"{float((a > 0.01).float().mean()):>10.3f}")
    return peak


def hf_retention(pred, gt, hi_from=0.5):
    """Fraction of the ground truth's high-frequency power the output keeps.

    1.0 means the output has as much fine detail as the reference. Our dim96
    model measured 0.28 - it erased 72% of the fine texture while scoring well
    on PSNR, which is the failure this function exists to catch.
    """
    def hf_power(x):
        f = np.fft.fftshift(np.fft.fft2(x - x.mean()))
        h, w = x.shape
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
        return float((np.abs(f) ** 2)[r >= hi_from].sum())

    g = hf_power(gt)
    return hf_power(pred) / max(g, 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--compare", default=None,
                    help="Second checkpoint to measure against, e.g. the "
                         "non-attention baseline")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--images", type=int, default=40)
    ap.add_argument("--split", default="val_ood")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== {args.weights} ===")
    model, state, nl = load_model(args.weights, device)
    print(f"non-local block present: {nl}")

    peak = None
    if nl:
        print("\n[gamma] zero-initialised, so this is how much training used it:")
        peak = report_gamma(state)

    # ------------------------------------------------------ texture retention
    cfg = load_config(overrides=args.set)
    root = Path(cfg.get_path("data.root"))
    gt_dir = root / cfg.get_path("data.gt_subdir", "GT")
    lr_dir = root / cfg.get_path("data.lr_subdir", "NoisyLR")

    stems = []
    sp = Path("artifacts/splits.json")
    if sp.exists():
        cand = json.load(open(sp)).get(args.split) or []
        stems = [s for s in cand if (lr_dir / f"{s}.npy").exists()][:args.images]
    if not stems:
        stems = [p.stem for p in sorted(lr_dir.glob("*.npy"))[:args.images]]

    models = [("this", model)]
    if args.compare:
        m2, _, nl2 = load_model(args.compare, device)
        models.append(("compare", m2))
        print(f"\n=== {args.compare} === non-local: {nl2}")

    acc = {n: {"hf": 0.0, "psnr": 0.0, "std": 0.0} for n, _ in models}
    acc_gt_std = 0.0
    n = 0
    for stem in stems:
        gt = np.load(gt_dir / f"{stem}.npy").astype(np.float32)
        lr = np.load(lr_dir / f"{stem}.npy").astype(np.float32)
        if gt.ndim == 3:
            gt = gt[..., 0]
        if lr.ndim == 3:
            lr = lr[..., 0]
        x = torch.from_numpy(lr)[None, None].to(device)
        with torch.no_grad():
            for name, m in models:
                p = m(x).clamp(0, 1)[0, 0].cpu().numpy()
                acc[name]["hf"] += hf_retention(p, gt)
                acc[name]["psnr"] += 10 * np.log10(
                    1.0 / max(float(np.mean((p - gt) ** 2)), 1e-12))
                acc[name]["std"] += float(p.std())
        acc_gt_std += float(gt.std())
        n += 1

    print(f"\n[texture]  {n} images from {args.split}")
    print(f"  {'model':<10} {'PSNR':>8} {'HF retained':>13} {'pixel std':>11}")
    print(f"  {'ground truth':<10} {'-':>8} {'1.000':>13} {acc_gt_std/n:>11.4f}")
    for name, _ in models:
        a = acc[name]
        print(f"  {name:<10} {a['psnr']/n:>8.2f} {a['hf']/n:>13.3f} "
              f"{a['std']/n:>11.4f}")

    # ---------------------------------------------------------------- verdict
    print("\nVERDICT")
    if nl and peak is not None and peak < 0.02:
        print(f"  Block is INERT. Peak |gamma| is {peak:.5f}, essentially the")
        print("  zero it was initialised to, so attention contributed nothing.")
        print("  Whatever changed the images came from somewhere else - check")
        print("  loss.gt_smooth_sigma and confirm you compared against the same")
        print("  baseline checkpoint.")
    elif nl:
        print(f"  Block is ACTIVE (peak |gamma| = {peak:.4f}). Attention is being")
        print("  used, so the implementation works.")
        print("  If the output is smoother, that is the MECHANISM, not a bug:")
        print("  averaging approximately-matching patches removes their")
        print("  individual texture along with the noise. Non-local averaging")
        print("  raises PSNR by smoothing, which is the same trap PSNR always")
        print("  sets on this problem.")
    hf = acc["this"]["hf"] / n
    print(f"\n  High-frequency retention {hf:.3f}. Below ~0.3 the model is")
    print("  erasing most of the fine texture regardless of what PSNR says.")
    if args.compare:
        d = hf - acc["compare"]["hf"] / n
        print(f"  Against the comparison checkpoint: {d:+.3f}. Negative means "
              f"attention cost texture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
