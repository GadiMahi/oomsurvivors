#!/usr/bin/env python3
"""Comparison figure: noisy input / bicubic / our model / ground truth.

Selection matters more than rendering here. Picking images at random, or worse
picking the ones that look best, produces a figure a reviewer will not trust.
`--mode spread` (the default) sorts the sample by how much the model beats
bicubic and then takes an even spread from worst to best, so the panel contains
our failures as well as our successes. A figure that includes a case where we
barely help is more persuasive than one that does not.

Note the default is NO test-time augmentation. TTA raises PSNR and SSIM but
costs 22% of LPIPS and a third of the high-frequency content, so the images it
produces are visibly smoother than what we actually submit. See section 3.2 of
the handover report.

Usage:
    python scripts/make_figure.py --weights weights/best_nafnet.pt \\
        --set data.root=$DATA --n 8 --out results/comparison.png

    # a single image blown up for a slide, with a zoom inset
    python scripts/make_figure.py --weights ... --n 1 --mode best --zoom 0.25
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
from src.tta import tta_forward  # noqa: E402


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
    print(f"model: dim={dim} levels={levels} non_local={nl}")
    return m


def psnr(a, b):
    return 10 * np.log10(1.0 / max(float(np.mean((a - b) ** 2)), 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="weights/best_nafnet.pt")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--split", default="val_ood")
    ap.add_argument("--n", type=int, default=8, help="rows in the figure")
    ap.add_argument("--pool", type=int, default=60,
                    help="images to score before selecting")
    ap.add_argument("--mode", default="spread",
                    choices=["spread", "best", "worst", "random"])
    ap.add_argument("--tta", type=int, default=1, choices=[1, 2, 4, 8])
    ap.add_argument("--zoom", type=float, default=0.0,
                    help="if >0, crop this fraction of the image (centre) so "
                         "fine texture is actually visible at slide size")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--out", default="results/comparison.png")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    rng = np.random.default_rng(args.seed)
    pool = list(rng.permutation(stems)[:args.pool])
    print(f"scoring {len(pool)} candidates from {args.split} ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.weights), device)

    rows = []
    for stem in pool:
        gt = np.load(gt_dir / f"{stem}.npy").astype(np.float32)
        lr = np.load(lr_dir / f"{stem}.npy").astype(np.float32)
        if gt.ndim == 3:
            gt = gt[..., 0]
        if lr.ndim == 3:
            lr = lr[..., 0]
        x = torch.from_numpy(lr)[None, None].to(device)
        with torch.no_grad():
            bic = F.interpolate(x, scale_factor=2, mode="bicubic",
                                align_corners=False).clamp(0, 1)[0, 0].cpu().numpy()
            out = (tta_forward(model, x, args.tta) if args.tta > 1
                   else model(x)).clamp(0, 1)[0, 0].cpu().numpy()
        rows.append({
            "stem": stem, "lr": lr, "bic": bic, "out": out, "gt": gt,
            "p_bic": psnr(bic, gt), "p_out": psnr(out, gt),
            "s_bic": stratified_ssim(bic, gt)["ssim"],
            "s_out": stratified_ssim(out, gt)["ssim"],
        })

    for r in rows:
        r["gain"] = r["p_out"] - r["p_bic"]
    rows.sort(key=lambda r: r["gain"])

    n = min(args.n, len(rows))
    if args.mode == "spread":
        # Even spread from our worst case to our best, so the figure is honest.
        idx = np.linspace(0, len(rows) - 1, n).round().astype(int)
        pick = [rows[i] for i in idx]
    elif args.mode == "worst":
        pick = rows[:n]
    elif args.mode == "best":
        pick = rows[-n:][::-1]
    else:
        pick = [rows[i] for i in rng.permutation(len(rows))[:n]]

    def crop(a):
        if args.zoom <= 0:
            return a
        h, w = a.shape
        ch, cw = int(h * args.zoom), int(w * args.zoom)
        y, x = (h - ch) // 2, (w - cw) // 2
        return a[y:y + ch, x:x + cw]

    titles = ["Noisy input (128x128)", "Bicubic (256x256)",
              "Ours (256x256)", "Ground truth (256x256)"]
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.35 * n), squeeze=False)
    for i, r in enumerate(pick):
        panels = [crop(r["lr"]), crop(r["bic"]), crop(r["out"]), crop(r["gt"])]
        labels = [
            r["stem"],
            f"PSNR {r['p_bic']:.2f} dB   SSIM {r['s_bic']:.3f}",
            f"PSNR {r['p_out']:.2f} dB   SSIM {r['s_out']:.3f}",
            "reference",
        ]
        colours = ["#333333", "#b03030", "#1a7a3a", "#333333"]
        for j, (img, lab, col) in enumerate(zip(panels, labels, colours)):
            ax = axes[i][j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=11, fontweight="bold", pad=8)
            ax.text(0.02, 0.975, lab, transform=ax.transAxes, fontsize=8.5,
                    va="top", ha="left", color=col,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="none", alpha=0.82))

    tta_note = "no TTA" if args.tta == 1 else f"TTA x{args.tta}"
    zoom_note = f", centre {int(args.zoom*100)}% crop" if args.zoom > 0 else ""
    fig.suptitle(f"SEM restoration - held-out {args.split} images "
                 f"({args.mode} selection, {tta_note}{zoom_note})",
                 fontsize=12.5, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.995])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}  ({n} rows, {args.dpi} dpi)")

    mg = np.mean([r["gain"] for r in rows])
    print(f"\nover the {len(rows)}-image pool: mean PSNR gain over bicubic "
          f"{mg:+.2f} dB")
    print(f"  worst case {rows[0]['gain']:+.2f} dB ({rows[0]['stem']})")
    print(f"  best case  {rows[-1]['gain']:+.2f} dB ({rows[-1]['stem']})")
    print("\nQuote the worst case too. A reviewer who finds it themselves will "
          "trust the rest less.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
