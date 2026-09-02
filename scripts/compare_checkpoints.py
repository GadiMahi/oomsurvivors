#!/usr/bin/env python3
"""Rank every checkpoint on texture retention, not just PSNR and SSIM.

WHY THIS EXISTS
---------------
Every architecture and loss decision in this project was made on PSNR and SSIM.
We have since established twice that both metrics reward smoothing on this data:

  * the TTA sweep: +1.8% SSIM bought by destroying a third of the high-frequency
    content and worsening LPIPS by 22%
  * the spectral-loss run: HF retention tripled while every metric got worse

So the conclusions drawn from those metrics - "capacity is not the bottleneck",
"depth changes nothing", "higher LPIPS weight is a bad trade" - were all reached
with an instrument that cannot see texture loss. This script re-runs the
comparison with high-frequency retention and LPIPS included.

The specific hypothesis worth testing: `lpips02` was rejected because PSNR fell
0.27 dB and SSIM fell 0.017, but it had the best LPIPS of any model trained
(0.3448). If it also retains substantially more texture, it was rejected using
the wrong instrument.

Usage:
    python scripts/compare_checkpoints.py --set data.root=$DATA \\
        weights/best_nafnet.pt /kaggle/input/ckpts/lpips02_best_nafnet.pt

    python scripts/compare_checkpoints.py --set data.root=$DATA --images 80 \\
        --figure results/checkpoint_texture.png  A.pt B.pt C.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.eval_utils import stratified_ssim  # noqa: E402
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
    levels, blocks, mb = None, 2, 2
    if is_legacy_state_dict(state):
        state = remap_legacy_state_dict(state)
        levels, blocks, mb = 1, 2, 4
    else:
        levels = 1 + max((int(k.split(".")[1]) for k in state
                          if k.startswith("encoders.")), default=0)
    nl = any(".to_kv." in k or ".to_qkv." in k for k in state)
    if nl:
        mb = max(mb, sum(1 for k in state if k.startswith("middle.")
                         and k.endswith(".conv1.weight")))
    m = build_model("nafnet", scale=2, dim=dim or 64, levels=levels,
                    blocks=blocks, middle_blocks=mb, non_local=nl,
                    nl_mode=("window" if any(".to_qkv." in k for k in state)
                             else "global")).to(device).eval()
    m.load_state_dict(state)
    n_par = sum(p.numel() for p in m.parameters())
    return m, f"dim{dim} L{levels}{' +nl' if nl else ''}", n_par


def hf_retention(pred, gt, hi_from=0.5):
    def p(x):
        f = np.fft.fftshift(np.fft.fft2(x - x.mean()))
        h, w = x.shape
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
        return float((np.abs(f) ** 2)[r >= hi_from].sum())
    return p(pred) / max(p(gt), 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--split", default="val_ood")
    ap.add_argument("--images", type=int, default=80)
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--figure", default=None,
                    help="also render a same-image comparison across all models")
    ap.add_argument("--out", default="artifacts/checkpoint_comparison.json")
    args = ap.parse_args()

    cfg = load_config(overrides=args.set)
    root = Path(cfg.get_path("data.root"))
    gt_dir = root / cfg.get_path("data.gt_subdir", "GT")
    lr_dir = root / cfg.get_path("data.lr_subdir", "NoisyLR")

    stems = []
    sp = Path("artifacts/splits.json")
    if sp.exists():
        cand = json.load(open(sp)).get(args.split) or []
        stems = [s for s in cand if (lr_dir / f"{s}.npy").exists()]
    if not stems:
        stems = [p.stem for p in sorted(lr_dir.glob("*.npy"))]
    stems = stems[:args.images]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lp = None
    if not args.no_lpips:
        import lpips as _l
        lp = _l.LPIPS(net="vgg").to(device).eval()
        for p_ in lp.parameters():
            p_.requires_grad = False

    models = []
    for c in args.checkpoints:
        if not Path(c).exists():
            print(f"!! missing, skipping: {c}")
            continue
        m, desc, npar = load_model(Path(c), device)
        models.append((Path(c).stem, desc, npar, m))
        print(f"loaded {Path(c).stem:28} {desc:14} {npar/1e6:.2f}M")
    if not models:
        print("no checkpoints loaded")
        return 2
    print(f"\n{len(stems)} images from {args.split}\n")

    acc = {k[0]: {"psnr": 0.0, "ssim": 0.0, "edge": 0.0, "flat": 0.0,
                  "hf": 0.0, "lpips": 0.0} for k in models}
    bic = {"psnr": 0.0, "ssim": 0.0, "hf": 0.0}
    keep = []
    n = 0
    for stem in stems:
        gt = np.load(gt_dir / f"{stem}.npy").astype(np.float32)
        lr = np.load(lr_dir / f"{stem}.npy").astype(np.float32)
        if gt.ndim == 3:
            gt = gt[..., 0]
        if lr.ndim == 3:
            lr = lr[..., 0]
        x = torch.from_numpy(lr)[None, None].to(device)
        g = torch.from_numpy(gt)[None, None].to(device)

        with torch.no_grad():
            b = F.interpolate(x, scale_factor=2, mode="bicubic",
                              align_corners=False).clamp(0, 1)[0, 0].cpu().numpy()
            bic["psnr"] += 10 * np.log10(1 / max(float(np.mean((b - gt) ** 2)), 1e-12))
            bic["ssim"] += stratified_ssim(b, gt)["ssim"]
            bic["hf"] += hf_retention(b, gt)

            row = {}
            for name, _, _, m in models:
                o_t = m(x).clamp(0, 1)
                if lp is not None:
                    acc[name]["lpips"] += float(lp(o_t.repeat(1, 3, 1, 1) * 2 - 1,
                                                   g.repeat(1, 3, 1, 1) * 2 - 1).sum())
                o = o_t[0, 0].cpu().numpy()
                acc[name]["psnr"] += 10 * np.log10(
                    1 / max(float(np.mean((o - gt) ** 2)), 1e-12))
                s = stratified_ssim(o, gt)
                acc[name]["ssim"] += s["ssim"]
                acc[name]["edge"] += s["ssim_edge"]
                acc[name]["flat"] += s["ssim_flat"]
                acc[name]["hf"] += hf_retention(o, gt)
                row[name] = o
        if args.figure and len(keep) < 3:
            keep.append((stem, lr, b, row, gt))
        n += 1
        if n % 20 == 0:
            print(f"  {n}/{len(stems)}")

    for k in acc:
        for mm in acc[k]:
            acc[k][mm] /= n
    for mm in bic:
        bic[mm] /= n

    print(f"\n{'checkpoint':<28} {'params':>8} {'PSNR':>8} {'SSIM':>8} "
          f"{'flat':>8} {'HF ret':>8} {'LPIPS':>8}")
    print("-" * 82)
    print(f"{'bicubic':<28} {'-':>8} {bic['psnr']:>8.2f} {bic['ssim']:>8.4f} "
          f"{'-':>8} {bic['hf']:>8.3f} {'-':>8}")
    print(f"{'ground truth':<28} {'-':>8} {'inf':>8} {'1.0000':>8} "
          f"{'-':>8} {'1.000':>8} {'0.0000':>8}")
    order = sorted(models, key=lambda t: -acc[t[0]]["hf"])
    for name, desc, npar, _ in order:
        a = acc[name]
        print(f"{name[:27]:<28} {npar/1e6:>7.2f}M {a['psnr']:>8.2f} "
              f"{a['ssim']:>8.4f} {a['flat']:>8.4f} {a['hf']:>8.3f} "
              f"{a['lpips']:>8.4f}")
    print("\n(sorted by high-frequency retention, best first)")

    best_hf = order[0][0]
    best_ssim = max(models, key=lambda t: acc[t[0]]["ssim"])[0]
    best_lpips = min(models, key=lambda t: acc[t[0]]["lpips"])[0]
    print(f"\n  most texture : {best_hf}   (HF {acc[best_hf]['hf']:.3f})")
    print(f"  best SSIM    : {best_ssim}  (SSIM {acc[best_ssim]['ssim']:.4f})")
    print(f"  best LPIPS   : {best_lpips} (LPIPS {acc[best_lpips]['lpips']:.4f})")
    if best_hf != best_ssim:
        d_ssim = acc[best_ssim]["ssim"] - acc[best_hf]["ssim"]
        d_hf = acc[best_hf]["hf"] - acc[best_ssim]["hf"]
        print(f"\n  The SSIM winner and the texture winner are DIFFERENT models.")
        print(f"  Choosing on SSIM costs {100*d_hf/max(acc[best_ssim]['hf'],1e-9):.0f}% "
              f"of the high-frequency content to gain {d_ssim:.4f} SSIM.")
        print("  That is the same trade TTA was making, and we rejected it there.")

    if args.figure and keep:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [m[0] for m in models]
        cols = 3 + len(names)
        fig, axes = plt.subplots(len(keep), cols,
                                 figsize=(3.1 * cols, 3.3 * len(keep)),
                                 squeeze=False)
        for i, (stem, lr, b, row, gt) in enumerate(keep):
            panels = [("input", lr), ("bicubic", b)] + \
                     [(nm, row[nm]) for nm in names] + [("ground truth", gt)]
            for j, (t, img) in enumerate(panels):
                ax = axes[i][j]
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(t[:22], fontsize=9.5, fontweight="bold")
        fig.suptitle("Same images, every checkpoint (no TTA)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"\nwrote {args.figure}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"n": n, "split": args.split, "bicubic": bic,
                   "models": {k: acc[k] for k in acc}}, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
