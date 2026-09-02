#!/usr/bin/env python3
"""Can we add the model's discarded high frequencies back from the input?

THE OBSERVATION
---------------
Put bicubic, our output, and the ground truth side by side and the ground truth
looks like a superposition of the other two: our model has the structure,
bicubic has the fine grain, and the reference appears to have both.

    output = model(x) + alpha * highpass(bicubic(x), sigma)

THE CATCH
---------
Bicubic's grain is the INPUT's noise realisation. The ground truth's grain is a
DIFFERENT realisation from the original capture. They are uncorrelated, so
adding one to the other produces something that looks right and is pointwise
wrong. That is why every unsharp-masking setting we tried lost PSNR.

WHY THIS IS STILL WORTH MEASURING
---------------------------------
Unsharp masking amplified high frequencies ALREADY PRESENT in the output. This
adds back frequencies the model REMOVED, and those are not pure noise - real
structure that survived downsampling got discarded along with the noise. So
there may be a small alpha where recovered detail outweighs injected noise.
Nobody has measured where that optimum sits, or whether it exists.

WHAT TO EXPECT
--------------
PSNR should fall monotonically with alpha, because uncorrelated noise always
increases squared error. The interesting columns are LPIPS and HF retention: if
LPIPS improves while PSNR degrades, that is the perception-distortion trade-off
appearing in a form you can actually choose a point on, rather than an argument.

Usage:
    python scripts/detail_injection.py --weights runs/final/best_nafnet.pt \\
        --set data.root=$DATA
    python scripts/detail_injection.py --weights ... --alphas 0 0.1 0.2 0.3 0.5 \\
        --sigmas 0.8 1.2 --images 60 --save-fig results/detail_injection.png
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
    nl = any(".to_kv." in k for k in state)
    if nl:
        mb = max(mb, sum(1 for k in state if k.startswith("middle.")
                         and k.endswith(".conv1.weight")))
    m = build_model("nafnet", scale=2, dim=dim or 64, levels=levels,
                    blocks=blocks, middle_blocks=mb, non_local=nl).to(device).eval()
    m.load_state_dict(state)
    return m


def gauss_kernel(sigma, device, ksize=None):
    ksize = ksize or max(3, int(2 * round(3 * sigma) + 1))
    g = torch.arange(ksize, device=device, dtype=torch.float32) - (ksize - 1) / 2
    g = torch.exp(-g.pow(2) / (2 * sigma ** 2))
    return (g / g.sum()), ksize


def blur(x, sigma):
    g, k = gauss_kernel(sigma, x.device)
    p = k // 2
    x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"), g.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"), g.view(1, 1, -1, 1))


def hf_retention(pred, gt, hi_from=0.5):
    """Share of the ground truth's high-frequency power the output reproduces."""
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
    ap.add_argument("--weights", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--split", default="val_ood")
    ap.add_argument("--images", type=int, default=50)
    ap.add_argument("--tta", type=int, default=4, choices=[1, 2, 4, 8])
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0])
    ap.add_argument("--lpips", action="store_true", help="also compute LPIPS (slower)")
    ap.add_argument("--out", default="artifacts/detail_injection.json")
    args = ap.parse_args()

    from src.config import load_config
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    print(f"{len(stems)} images, tta={args.tta}, device={device}\n")

    lp = None
    if args.lpips:
        import lpips as _l
        lp = _l.LPIPS(net="vgg").to(device).eval()
        for p_ in lp.parameters():
            p_.requires_grad = False

    keys = [(s, a) for s in args.sigmas for a in args.alphas]
    acc = {k: {"psnr": 0.0, "ssim": 0.0, "flat": 0.0, "hf": 0.0, "lpips": 0.0}
           for k in keys}
    bic_acc = {"psnr": 0.0, "ssim": 0.0, "hf": 0.0}
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
            bic = F.interpolate(x, scale_factor=2, mode="bicubic",
                                align_corners=False)
            base = (tta_forward(model, x, args.tta) if args.tta > 1
                    else model(x)).clamp(0, 1)

            b = bic.clamp(0, 1)[0, 0].cpu().numpy()
            bic_acc["psnr"] += 10 * np.log10(1.0 / max(float(np.mean((b - gt) ** 2)), 1e-12))
            bic_acc["ssim"] += stratified_ssim(b, gt)["ssim"]
            bic_acc["hf"] += hf_retention(b, gt)

            for sigma in args.sigmas:
                # The detail the model discarded: bicubic minus its own low-pass.
                hp = bic - blur(bic, sigma)
                for a in args.alphas:
                    out = (base + a * hp).clamp(0, 1)
                    o = out[0, 0].cpu().numpy()
                    k = (sigma, a)
                    acc[k]["psnr"] += 10 * np.log10(
                        1.0 / max(float(np.mean((o - gt) ** 2)), 1e-12))
                    s = stratified_ssim(o, gt)
                    acc[k]["ssim"] += s["ssim"]
                    acc[k]["flat"] += s["ssim_flat"]
                    acc[k]["hf"] += hf_retention(o, gt)
                    if lp is not None:
                        acc[k]["lpips"] += float(lp(
                            out.repeat(1, 3, 1, 1) * 2 - 1,
                            g.repeat(1, 3, 1, 1) * 2 - 1).sum())
        n += 1
        if n % 10 == 0:
            print(f"  {n}/{len(stems)}")

    for k in acc:
        for m in acc[k]:
            acc[k][m] /= n
    for m in bic_acc:
        bic_acc[m] /= n

    hdr = f"\n{'sigma':>6} {'alpha':>6} {'PSNR':>8} {'SSIM':>8} {'flat':>8} {'HF ret':>8}"
    if lp is not None:
        hdr += f" {'LPIPS':>8}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{'bicubic':>13} {bic_acc['psnr']:>8.2f} {bic_acc['ssim']:>8.4f} "
          f"{'-':>8} {bic_acc['hf']:>8.3f}")
    print(f"{'GT':>13} {'inf':>8} {'1.0000':>8} {'-':>8} {'1.000':>8}")
    for (sigma, a) in keys:
        r = acc[(sigma, a)]
        line = (f"{sigma:>6.1f} {a:>6.2f} {r['psnr']:>8.2f} {r['ssim']:>8.4f} "
                f"{r['flat']:>8.4f} {r['hf']:>8.3f}")
        if lp is not None:
            line += f" {r['lpips']:>8.4f}"
        print(line + ("   <- no injection" if a == 0.0 else ""))

    base_k = (args.sigmas[0], 0.0)
    print("\nREADING THIS")
    print("  alpha=0 is the model alone. PSNR should fall as alpha rises,")
    print("  because the injected grain is the INPUT's noise realisation and is")
    print("  uncorrelated with the ground truth's. If SSIM or LPIPS improve")
    print("  while PSNR falls, that is a real choice on the")
    print("  perception-distortion curve rather than a free win.")
    best_ssim = max(keys, key=lambda k: acc[k]["ssim"])
    print(f"\n  best SSIM at sigma={best_ssim[0]}, alpha={best_ssim[1]}: "
          f"{acc[best_ssim]['ssim']:.4f} vs {acc[base_k]['ssim']:.4f} at alpha=0 "
          f"({acc[best_ssim]['ssim'] - acc[base_k]['ssim']:+.4f}, costing "
          f"{acc[best_ssim]['psnr'] - acc[base_k]['psnr']:+.2f} dB)")
    if lp is not None:
        best_lp = min(keys, key=lambda k: acc[k]["lpips"])
        print(f"  best LPIPS at sigma={best_lp[0]}, alpha={best_lp[1]}: "
              f"{acc[best_lp]['lpips']:.4f} vs {acc[base_k]['lpips']:.4f} "
              f"({acc[best_lp]['lpips'] - acc[base_k]['lpips']:+.4f}, costing "
              f"{acc[best_lp]['psnr'] - acc[base_k]['psnr']:+.2f} dB)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"bicubic": bic_acc, "n": n, "tta": args.tta,
                   "grid": {f"s{s}_a{a}": acc[(s, a)] for (s, a) in keys}}, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
