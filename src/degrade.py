"""Synthetic degradation for extra training pairs (explicitly permitted, 4B).

Two design decisions come straight from the problem statement:

  * "The three degradations may have been applied in any order" -> the order is
    sampled per image, using the mix ratio measured in 02_degradation.ipynb.
  * "Noise mechanisms remain the same; sampled levels may vary within a similar
    range" -> jitter is deliberately modest (+/-30%). Widening it further makes
    the model hedge and blur, which the spec explicitly penalises.

Nothing here clips: the out-of-range values are the point.
"""
from __future__ import annotations

import numpy as np

try:  # optional, only needed for some kernels
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from PIL import Image

_PIL_FILTERS = {
    "bicubic_aa": Image.BICUBIC,
    "bilinear_aa": Image.BILINEAR,
    "lanczos": Image.LANCZOS,
    "area": Image.BOX,
    "nearest": Image.NEAREST,
}


def _decimate(img: np.ndarray, scale: int) -> np.ndarray:
    """Pure subsampling. Maximum aliasing - keeps all high-frequency energy."""
    return np.ascontiguousarray(img[::scale, ::scale], dtype=np.float32)


def _zoom(img: np.ndarray, scale: int, order: int) -> np.ndarray:
    """scipy interpolation WITHOUT a prefilter low-pass, so aliasing survives.

    This is what torch.interpolate(antialias=False) and cv2.resize do, and both
    are common defaults. PIL, by contrast, always antialiases on downscale.
    """
    from scipy.ndimage import zoom as _z
    h, w = img.shape[-2:]
    out = _z(img.astype(np.float32), 1.0 / scale, order=order, mode="reflect",
             grid_mode=True, prefilter=False)
    return np.ascontiguousarray(out[: h // scale, : w // scale], dtype=np.float32)


# Non-antialiased variants. Their presence matters: an antialiased downsample
# removes high-frequency content that a non-antialiased one folds back into the
# image as aliasing, and that difference is invisible to a low-pass comparison
# but dominates the high end of the power spectrum.
_NOAA = {
    "decimate": lambda im, s: _decimate(im, s),
    "cubic_noaa": lambda im, s: _zoom(im, s, 3),
    "linear_noaa": lambda im, s: _zoom(im, s, 1),
    "nearest_noaa": lambda im, s: _zoom(im, s, 0),
}


def _gauss_decimate(img: np.ndarray, scale: int, sigma: float) -> np.ndarray:
    """Blur by `sigma`, then decimate. Tunable antialiasing strength.

    sigma = 0     -> pure decimation, maximum aliasing (measured too SHARP:
                     autocorr -0.086 vs real, local variance ratio 1.16)
    sigma ~ 0.8+  -> approaches box/area averaging (measured too SMOOTH:
                     autocorr +0.050, local variance ratio 0.95)

    Real round-2 data sits between those two, i.e. KLA used partial
    antialiasing. Sweeping sigma finds where.
    """
    from scipy.ndimage import gaussian_filter
    if sigma > 0:
        img = gaussian_filter(img.astype(np.float32), sigma, mode="reflect")
    return _decimate(img, scale)


def downsample(img: np.ndarray, kernel: str, scale: int = 2) -> np.ndarray:
    if isinstance(kernel, str) and kernel.startswith("gauss:"):
        return _gauss_decimate(img, scale, float(kernel.split(":", 1)[1]))
    if kernel in _NOAA:
        return _NOAA[kernel](img, scale)
    h, w = img.shape[-2:]
    size = (max(1, w // scale), max(1, h // scale))  # PIL wants (W, H)
    if kernel not in _PIL_FILTERS:
        raise ValueError(f"Unknown kernel {kernel!r}. Known: "
                         f"{sorted(set(_PIL_FILTERS) | set(_NOAA))}")
    out = Image.fromarray(img.astype(np.float32), mode="F").resize(size, _PIL_FILTERS[kernel])
    return np.asarray(out, dtype=np.float32)


def spectral_distance(a: np.ndarray, b: np.ndarray, hi_from: float = 0.5) -> float:
    """Relative power-spectrum mismatch in the HIGH-frequency band.

    Low-pass MSE deliberately discards exactly the band where antialiased and
    non-antialiased kernels differ, so it cannot tell them apart. This can.
    """
    def radial(x, nb=32):
        F = np.fft.fftshift(np.abs(np.fft.fft2(x - x.mean())))
        h, w = F.shape
        yy, xx = np.mgrid[:h, :w]
        r = np.hypot(yy - h / 2, xx - w / 2)
        r = (r / r.max() * (nb - 1)).astype(int)
        return np.array([F[r == k].mean() for k in range(nb)])

    ra, rb = radial(a), radial(b)
    k0 = int(len(ra) * hi_from)
    return float(np.mean(np.abs(rb[k0:] / np.maximum(ra[k0:], 1e-12) - 1.0)))


def add_noise(img: np.ndarray, sigma_mult: float, sigma_add: float, rng) -> np.ndarray:
    """Speckle (multiplicative) then additive Gaussian. No clipping.

    NOTE: this only reproduces the a*mu^2 and c terms of a fitted variance
    curve. If the fit has a significant linear (b) term, use add_noise_varfit
    instead - otherwise most of the measured variance is silently dropped.
    """
    out = img
    if sigma_mult > 0:
        out = out * (1.0 + rng.normal(0.0, sigma_mult, out.shape).astype(np.float32))
    if sigma_add > 0:
        out = out + rng.normal(0.0, sigma_add, out.shape).astype(np.float32)
    return out.astype(np.float32)


def add_noise_varfit(img: np.ndarray, a: float, b: float, c: float, rng,
                     scale: float = 1.0) -> np.ndarray:
    """Signal-dependent noise matching a measured variance curve.

        var(r | mu) = a*mu^2 + b*mu + c

    This reproduces the observed second-order statistics regardless of how the
    variance decomposes into speckle / shot / read noise - which matters because
    those three components are strongly correlated in the fit and cannot be
    separated reliably from paired data alone.

    `scale` multiplies the noise standard deviation (the curriculum knob).
    """
    mu = np.clip(img, 0.0, None)
    var = a * mu * mu + b * mu + c
    sigma = np.sqrt(np.clip(var, 0.0, None)).astype(np.float32) * np.float32(scale)
    return (img + sigma * rng.standard_normal(img.shape).astype(np.float32)).astype(np.float32)


def add_noise_empirical(img: np.ndarray, banks, a: float, b: float, c: float, rng,
                        scale: float = 1.0) -> np.ndarray:
    """Signal-dependent noise with the REAL tail shape, resampled from data.

    `banks` comes from noise_fit.build_residual_bank: one array of normalised
    residuals per intensity bin, drawn from the genuine paired data. Sampling
    from these reproduces the measured skew (+0.52) and excess kurtosis (+2.08)
    exactly, where a Gaussian generator produces roughly a quarter of the real
    3-sigma outliers.

    Falls back to Gaussian for any bin with no collected residuals.
    """
    mu = np.clip(img, 0.0, None)
    sigma = np.sqrt(np.clip(a * mu * mu + b * mu + c, 0.0, None)).astype(np.float32)
    sigma *= np.float32(scale)

    nb = len(banks)
    idx = np.clip((np.clip(mu, 0.0, 1.0) * nb).astype(np.int32), 0, nb - 1)
    z = np.empty(img.shape, dtype=np.float32)
    for k in range(nb):
        m = idx == k
        cnt = int(m.sum())
        if not cnt:
            continue
        bk = banks[k]
        if bk.size > 1:
            z[m] = bk[rng.integers(0, bk.size, cnt)]
        else:
            z[m] = rng.standard_normal(cnt).astype(np.float32)
    return (img + sigma * z).astype(np.float32)


def degrade(gt: np.ndarray, rng, cfg, width: float | None = None) -> np.ndarray:
    """Produce a NoisyLR-like image from a clean GT image.

    Noise generator, in order of preference:
      1. 'residual_bank' present -> empirical resampling, reproduces the real
         heavy tails (round-2 excess kurtosis 2.08). Preferred.
      2. 'noise_var_fit' present -> Gaussian matched to the measured variance
         curve. Correct in the bulk, ~4x too few 3-sigma outliers.
      3. otherwise -> simple speckle + Gaussian parameterisation.
    """
    width = cfg["width"] if width is None else width
    j = float(cfg.get("jitter", 0.30)) * float(width)

    kernel = rng.choice(cfg["kernels"], p=np.asarray(cfg["kernel_p"], dtype=float))
    order = rng.choice(["noise_first", "noise_last"], p=np.asarray(cfg["order_mix"], dtype=float))

    fit = cfg.get("noise_var_fit")
    bank = cfg.get("residual_bank")
    if fit and bank is not None:
        scale = float(rng.uniform(1.0 - j, 1.0 + j))
        noise = lambda x: add_noise_empirical(  # noqa: E731
            x, bank, float(fit["a_mult"]), float(fit["b_poisson"]),
            float(fit["c_additive"]), rng, scale=scale)
    elif fit:
        # Jitter the whole noise level, preserving the shape of the curve.
        scale = float(rng.uniform(1.0 - j, 1.0 + j))
        noise = lambda x: add_noise_varfit(  # noqa: E731
            x, float(fit["a_mult"]), float(fit["b_poisson"]), float(fit["c_additive"]),
            rng, scale=scale)
    else:
        s_mult = float(cfg["meas_mult"]) * float(rng.uniform(1.0 - j, 1.0 + j))
        s_add = float(cfg["meas_add"]) * float(rng.uniform(1.0 - j, 1.0 + j))
        noise = lambda x: add_noise(x, s_mult, s_add, rng)  # noqa: E731

    if order == "noise_first":
        return downsample(noise(gt), kernel)
    return noise(downsample(gt, kernel))
