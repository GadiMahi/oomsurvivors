#!/usr/bin/env python3
"""Classical Wiener baseline: the optimal LINEAR estimator for this problem.

WHY THIS EXISTS
---------------
Bicubic is the conventional super-resolution baseline, but it is a weak
comparator here because it is purely an interpolation method - it addresses the
upsampling half of the degradation and does nothing at all about noise. Beating
it by 3 dB partly just says "we denoise and it does not".

The Wiener filter is the honest classical comparator. Given the signal and noise
power spectra it is the provably optimal linear minimum-MSE estimator, and we
have measured both quantities:

  * noise power from the fitted variance model var(y) = a*y^2 + b*y + c
  * signal power from the radial power spectrum of the ground truth

The filter is

    H(f) = S(f) / (S(f) + N(f))

which passes frequencies where signal dominates and attenuates those where noise
does. Note what this means: at high frequencies, where our own spectral analysis
showed the ground truth is essentially a flat noise floor, the optimal gain goes
to zero. The classical theory independently derives the same low-pass behaviour
we observed empirically in the network - the blur is not a modelling failure,
it is what optimal estimation looks like when the information is gone.

The network should beat this, and the margin is a direct measurement of what
nonlinearity buys over the best possible linear filter. That number is worth
having in the report.

Usage:
    python scripts/wiener_baseline.py --set data.root=/kaggle/input/.../semicon_train_data
    python scripts/wiener_baseline.py --split val_ood --limit 200
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
from src.eval_utils import stratified_ssim  # noqa: E402
from src.io_utils import load_stats  # noqa: E402


def radial_power(img: np.ndarray, nbins: int = 128) -> tuple:
    """Azimuthally averaged power spectrum. Returns (freq, power)."""
    f = np.fft.fftshift(np.fft.fft2(img - img.mean()))
    p = np.abs(f) ** 2
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    rmax = min(h, w) / 2
    bins = np.linspace(0, rmax, nbins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < nbins)
    power = np.bincount(idx[ok], weights=p.ravel()[ok], minlength=nbins)
    count = np.bincount(idx[ok], minlength=nbins).clip(1)
    return (bins[:-1] + bins[1:]) / 2 / rmax, power / count


def estimate_signal_psd(gt_paths, nbins=128, limit=100):
    """Average GT power spectrum - our estimate of S(f)."""
    acc, n = None, 0
    for p in gt_paths[:limit]:
        img = np.load(p).astype(np.float32)
        if img.ndim == 3:
            img = img[..., 0]
        fr, pw = radial_power(img, nbins)
        acc = pw if acc is None else acc + pw
        n += 1
    return fr, acc / max(n, 1)


def wiener_restore(lr: np.ndarray, freq: np.ndarray, sig_psd: np.ndarray,
                   a: float, b: float, c: float, scale: int = 2) -> np.ndarray:
    """Upsample to the target grid, then apply the Wiener gain in frequency.

    Order matters: the estimator has to act on the grid we want the answer on,
    so we interpolate first and filter second. The noise power is computed from
    the measured variance model at this image's own mean intensity, which is
    what makes this a fair use of the noise study rather than a tuned constant.
    """
    from scipy.ndimage import zoom

    up = zoom(lr, scale, order=3).astype(np.float32)

    mu = float(np.clip(up, 0, 1).mean())
    noise_var = a * mu * mu + b * mu + c

    h, w = up.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)

    # Interpolate the measured 1D signal PSD onto the 2D frequency grid.
    S = np.interp(r, freq, sig_psd, left=sig_psd[0], right=sig_psd[-1])
    # White noise: flat across frequency, total power = variance * pixel count.
    N = noise_var * h * w / (scale ** 2)

    H = S / (S + N)
    F = np.fft.fftshift(np.fft.fft2(up - up.mean()))
    out = np.real(np.fft.ifft2(np.fft.ifftshift(F * H))) + up.mean()
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides k=v")
    ap.add_argument("--split", default="val_ood")
    ap.add_argument("--limit", type=int, default=200,
                    help="Images to evaluate. The full split takes a while "
                         "because this is CPU FFT work.")
    ap.add_argument("--out", default="artifacts/wiener_baseline.json")
    args = ap.parse_args()

    cfg = load_config(overrides=args.set)
    stats = load_stats()
    fit = stats["noise_var_fit"]
    a, b, c = fit["a_mult"], fit["b_poisson"], fit["c_additive"]
    print(f"noise model: var = {a:.6g}*mu^2 + {b:.6g}*mu + {c:.6g}")

    root = Path(cfg.get_path("data.root"))
    gt_dir = root / cfg.get_path("data.gt_subdir", "GT")
    lr_dir = root / cfg.get_path("data.lr_subdir", "NoisyLR")

    with open("artifacts/splits.json") as f:
        stems = json.load(f)[args.split][:args.limit]
    print(f"split={args.split}  n={len(stems)}")

    print("estimating signal PSD from ground truth ...")
    freq, sig = estimate_signal_psd([gt_dir / f"{s}.npy" for s in stems], limit=100)

    from scipy.ndimage import zoom
    acc = {k: 0.0 for k in ("psnr", "ssim", "ssim_edge", "ssim_flat")}
    acc_bic = dict(acc)
    n, t0 = 0, time.perf_counter()

    for s in stems:
        gt = np.load(gt_dir / f"{s}.npy").astype(np.float32)
        lr = np.load(lr_dir / f"{s}.npy").astype(np.float32)
        if gt.ndim == 3:
            gt = gt[..., 0]
        if lr.ndim == 3:
            lr = lr[..., 0]

        for name, pred, store in (
                ("wiener", wiener_restore(lr, freq, sig, a, b, c), acc),
                ("bicubic", np.clip(zoom(lr, 2, order=3), 0, 1).astype(np.float32), acc_bic)):
            mse = float(np.mean((pred - gt) ** 2))
            store["psnr"] += 10 * np.log10(1.0 / max(mse, 1e-12))
            sv = stratified_ssim(pred, gt)
            store["ssim"] += sv["ssim"]
            store["ssim_edge"] += sv["ssim_edge"]
            store["ssim_flat"] += sv["ssim_flat"]
        n += 1
        if n % 25 == 0:
            print(f"  {n}/{len(stems)}")

    res = {k: v / max(n, 1) for k, v in acc.items()}
    bic = {k: v / max(n, 1) for k, v in acc_bic.items()}
    dt = time.perf_counter() - t0

    print(f"\n{'':>10} {'PSNR':>8} {'SSIM':>8} {'edge':>8} {'flat':>8}")
    print(f"{'bicubic':>10} {bic['psnr']:>8.2f} {bic['ssim']:>8.4f} "
          f"{bic['ssim_edge']:>8.4f} {bic['ssim_flat']:>8.4f}")
    print(f"{'wiener':>10} {res['psnr']:>8.2f} {res['ssim']:>8.4f} "
          f"{res['ssim_edge']:>8.4f} {res['ssim_flat']:>8.4f}")
    print(f"{'delta':>10} {res['psnr'] - bic['psnr']:>+8.2f} "
          f"{res['ssim'] - bic['ssim']:>+8.4f} "
          f"{res['ssim_edge'] - bic['ssim_edge']:>+8.4f} "
          f"{res['ssim_flat'] - bic['ssim_flat']:>+8.4f}")
    print(f"\n{1000 * dt / max(n, 1):.1f} ms/image (CPU FFT, not comparable to GPU timings)")

    payload = {"n": n, "split": args.split, "wiener": res, "bicubic": bic,
               "noise_var_fit": fit, "ms_per_image_cpu": 1000 * dt / max(n, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
