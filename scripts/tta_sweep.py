#!/usr/bin/env python3
"""Measure the full cost of test-time augmentation, LPIPS included.

WHY THIS EXISTS
---------------
TTA was adopted on the strength of PSNR and SSIM alone (+0.18 dB, +0.011 SSIM),
which looked like the largest post-data gain available. A later spot check on 50
images with LPIPS in the table told a different story:

    tta=1   PSNR 23.58   SSIM 0.5282   HF 0.182   LPIPS 0.3618
    tta=4   PSNR 23.74   SSIM 0.5371   HF 0.119   LPIPS 0.4470

TTA averages several rotated predictions, and averaging smooths. It buys 0.7% of
PSNR and 1.7% of SSIM while costing 24% of LPIPS and a THIRD of the remaining
high-frequency content. On this problem PSNR and SSIM both reward smoothing, so
measuring only those hid the trade completely.

This runs the whole validation split rather than a 50-image sample, so the
decision rests on the real numbers.

Usage:
    python scripts/tta_sweep.py --weights weights/best_nafnet.pt \\
        --set data.root=$DATA
    python scripts/tta_sweep.py --weights ... --images 300 --variants 1 2 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

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

    vst_cfg = sd.get("vst") if isinstance(sd, dict) else None
    if vst_cfg:
        from src.vst import VST
        t = VST.from_dict(vst_cfg)
        inner = m

        class W(torch.nn.Module):
            def forward(self, x):
                return t.inverse(inner(t.forward(x)))

        m = W().to(device).eval()
        print(f"checkpoint uses VST: {t}")
    return m


def hf_retention(pred, gt, hi_from=0.5):
    """Share of the ground truth's high-frequency power the output reproduces.

    This is the column that exposes smoothing. PSNR and SSIM will not.
    """
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
    ap.add_argument("--weights", default="weights/best_nafnet.pt")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--split", default="val_ood")
    ap.add_argument("--images", type=int, default=0,
                    help="0 = the whole split")
    ap.add_argument("--variants", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--out", default="artifacts/tta_sweep.json")
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
    if args.images:
        stems = stems[:args.images]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.weights), device)
    print(f"{len(stems)} images from {args.split}, device={device}\n")

    lp = None
    if not args.no_lpips:
        import lpips as _l
        lp = _l.LPIPS(net=cfg.get_path("train.lpips_net", "vgg")).to(device).eval()
        for p_ in lp.parameters():
            p_.requires_grad = False

    res = {}
    for n_t in args.variants:
        acc = {"psnr": 0.0, "ssim": 0.0, "edge": 0.0, "flat": 0.0,
               "hf": 0.0, "lpips": 0.0}
        n = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for stem in stems:
            gt = np.load(gt_dir / f"{stem}.npy").astype(np.float32)
            lr = np.load(lr_dir / f"{stem}.npy").astype(np.float32)
            if gt.ndim == 3:
                gt = gt[..., 0]
            if lr.ndim == 3:
                lr = lr[..., 0]
            x = torch.from_numpy(lr)[None, None].to(device)
            with torch.no_grad():
                out = (tta_forward(model, x, n_t) if n_t > 1
                       else model(x)).clamp(0, 1)
                if lp is not None:
                    g = torch.from_numpy(gt)[None, None].to(device)
                    acc["lpips"] += float(lp(out.repeat(1, 3, 1, 1) * 2 - 1,
                                             g.repeat(1, 3, 1, 1) * 2 - 1).sum())
            o = out[0, 0].cpu().numpy()
            acc["psnr"] += 10 * np.log10(
                1.0 / max(float(np.mean((o - gt) ** 2)), 1e-12))
            s = stratified_ssim(o, gt)
            acc["ssim"] += s["ssim"]
            acc["edge"] += s["ssim_edge"]
            acc["flat"] += s["ssim_flat"]
            acc["hf"] += hf_retention(o, gt)
            n += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        r = {k: v / max(n, 1) for k, v in acc.items()}
        r["ms_per_image"] = 1000 * dt / max(n, 1)
        res[n_t] = r
        print(f"  tta={n_t} done ({r['ms_per_image']:.1f} ms/img)")

    base = res[args.variants[0]]
    print(f"\n{'tta':>4} {'PSNR':>8} {'SSIM':>8} {'edge':>8} {'flat':>8} "
          f"{'HF ret':>8} {'LPIPS':>8} {'ms/img':>8}")
    print("-" * 68)
    for n_t in args.variants:
        r = res[n_t]
        print(f"{n_t:>4} {r['psnr']:>8.3f} {r['ssim']:>8.4f} {r['edge']:>8.4f} "
              f"{r['flat']:>8.4f} {r['hf']:>8.3f} {r['lpips']:>8.4f} "
              f"{r['ms_per_image']:>8.1f}")

    print(f"\nCHANGE vs tta={args.variants[0]} (relative %)")
    print(f"{'tta':>4} {'PSNR':>10} {'SSIM':>10} {'HF ret':>10} {'LPIPS':>10}")
    print("-" * 48)
    for n_t in args.variants[1:]:
        r = res[n_t]
        print(f"{n_t:>4} "
              f"{100*(r['psnr']/base['psnr']-1):>+9.2f}% "
              f"{100*(r['ssim']/base['ssim']-1):>+9.2f}% "
              f"{100*(r['hf']/max(base['hf'],1e-9)-1):>+9.2f}% "
              f"{100*(r['lpips']/max(base['lpips'],1e-9)-1):>+9.2f}%")

    print("\nREADING THIS")
    print("  LPIPS is a DISTANCE, so lower is better and a positive % is worse.")
    print("  PSNR and SSIM both reward smoothing on this data, so if they rise")
    print("  while HF retention falls and LPIPS worsens, TTA is buying score by")
    print("  erasing texture. Decide using whichever metric KLA actually weights,")
    print("  and note that dropping TTA also returns the inference time.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"n_images": len(stems), "split": args.split,
                   "weights": args.weights,
                   "results": {str(k): v for k, v in res.items()}}, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
