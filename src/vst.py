"""Generalised Anscombe variance-stabilising transform (VST).

WHY THIS EXISTS
---------------
`scripts/build_residual_bank.py` fits the noise variance against signal
intensity and gets

    var(y) = a*y^2 + b*y + c      a=0.023807  b=0.010394  c=0.0030539

so noise magnitude depends on brightness.  Bright pixels carry roughly

    (a + b + c) / c  =  0.0373 / 0.0031  ~=  12x

the variance of dark ones.  Charbonnier weights every pixel equally, so the
training signal is dominated by bright regions purely because they are noisier,
not because they matter more.

A variance-stabilising transform is the function whose derivative is 1/sigma(y).
Applying it makes the noise approximately unit-variance everywhere, so the loss
becomes homoscedastic and dark regions stop being drowned out.

    f(y) = integral dy / sqrt(a*y^2 + b*y + c)

For a > 0 and 4ac > b^2 (true here: 4ac-b^2 = 1.828e-4) this integrates in
closed form to an arcsinh:

    f(y) = (1/sqrt(a)) * asinh( (2a*y + b) / sqrt(4ac - b^2) )

and inverts exactly:

    y = ( sqrt(4ac - b^2) * sinh(sqrt(a) * f) - b ) / (2a)

WHY ARCSINH AND NOT A LOG
-------------------------
The obvious move for multiplicative speckle is homomorphic filtering: take a
log, which turns multiplication into addition.  Two things kill it here.

  1. NoisyLR ranges down to -0.31.  log is undefined on negatives.
  2. The additive term c is real and non-zero.  In dark regions it dominates,
     and a log transform amplifies rather than stabilises it.

arcsinh is defined on the whole real line and is derived from the *measured*
variance model including c, so it handles both.  It behaves like a linear
function near zero and like a log for large arguments - exactly the
interpolation the noise model calls for.

NORMALISATION
-------------
Raw f maps [0,1] to about [4.60, 13.98], which is a poor range for a network
whose other hyperparameters were tuned on [0,1] data.  `normalize=True`
rescales linearly so f(0)=0 and f(1)=1.  This is an affine map, so variance
stays constant across intensities - the stabilisation is unaffected.

BIAS
----
E[f(y)] != f(E[y]) because f is nonlinear, so the naive inverse is slightly
biased.  For arcsinh the effect is far weaker than for the classic Anscombe
root transform, and it is a smooth function of intensity that the network can
absorb, since it trains and predicts entirely in transformed space.  If it ever
matters, the reference is Makitalo & Foi, "Optimal inversion of the generalized
Anscombe transformation", IEEE SPL 2013.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["VST", "vst_from_stats"]


class VST:
    """Arcsinh variance-stabilising transform for var(y) = a*y^2 + b*y + c.

    Works on numpy arrays and torch tensors alike; the arithmetic is plain
    operators plus asinh/sinh, both of which exist in either library.

    >>> v = VST(0.023807, 0.010394, 0.0030539)
    >>> float(v.inverse(v.forward(0.42)))
    0.42
    """

    def __init__(self, a: float, b: float, c: float, normalize: bool = True):
        disc = 4.0 * a * c - b * b
        if a <= 0.0:
            raise ValueError(f"VST needs a positive quadratic term, got a={a}")
        if disc <= 0.0:
            raise ValueError(
                f"4ac - b^2 = {disc:.6g} is not positive; the arcsinh form does not "
                "apply. Refit the noise model or use a different stabiliser.")

        self.a, self.b, self.c = float(a), float(b), float(c)
        self.sqrt_a = math.sqrt(a)
        self.sqrt_disc = math.sqrt(disc)
        self.normalize = bool(normalize)

        # Anchors so that forward(0) == 0 and forward(1) == 1 when normalising.
        f0 = self._raw(0.0)
        f1 = self._raw(1.0)
        self._off = f0 if normalize else 0.0
        self._gain = (f1 - f0) if normalize else 1.0

    # -- core ---------------------------------------------------------------

    def _raw(self, y: Any) -> Any:
        u = (2.0 * self.a * y + self.b) / self.sqrt_disc
        return _asinh(u) / self.sqrt_a

    def forward(self, y: Any) -> Any:
        """Image space -> stabilised space."""
        y = _as_float32(y)
        return (self._raw(y) - self._off) / self._gain

    def inverse(self, f: Any) -> Any:
        """Stabilised space -> image space. Exact inverse of `forward`."""
        f = _as_float32(f)
        raw = f * self._gain + self._off
        return (self.sqrt_disc * _sinh(raw * self.sqrt_a) - self.b) / (2.0 * self.a)

    # convenience
    __call__ = forward

    # -- reporting ----------------------------------------------------------

    def sigma(self, y: Any) -> Any:
        """Predicted noise std at intensity `y`, in image space."""
        return np.sqrt(np.clip(self.a * y * y + self.b * y + self.c, 0.0, None))

    @property
    def unit_noise_scale(self) -> float:
        """Noise std in transformed space (constant, by construction)."""
        return 1.0 / self._gain

    def range_of(self, lo: float = 0.0, hi: float = 1.0) -> tuple:
        return float(self.forward(lo)), float(self.forward(hi))

    def __repr__(self) -> str:
        lo, hi = self.range_of()
        return (f"VST(a={self.a:.6g}, b={self.b:.6g}, c={self.c:.6g}, "
                f"normalize={self.normalize}, maps [0,1]->[{lo:.3f},{hi:.3f}], "
                f"noise_std={self.unit_noise_scale:.5f})")

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "c": self.c, "normalize": self.normalize}

    @classmethod
    def from_dict(cls, d: dict) -> "VST":
        return cls(d["a"], d["b"], d["c"], d.get("normalize", True))


# --------------------------------------------------------------------- helpers

def _is_torch(x: Any) -> bool:
    return type(x).__module__.startswith("torch")


def _as_float32(x: Any) -> Any:
    """asinh/sinh in fp16 under AMP loses precision; force fp32."""
    if _is_torch(x):
        return x.float() if x.dtype != _torch().float32 else x
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    return x


def _torch():
    import torch
    return torch


def _asinh(x: Any) -> Any:
    if _is_torch(x):
        return _torch().asinh(x)
    if isinstance(x, np.ndarray):
        return np.arcsinh(x)
    return math.asinh(x)


def _sinh(x: Any) -> Any:
    if _is_torch(x):
        return _torch().sinh(x)
    if isinstance(x, np.ndarray):
        return np.sinh(x)
    return math.sinh(x)


def vst_from_stats(stats: dict | None = None, normalize: bool = True) -> VST:
    """Build the VST from the measured constants in artifacts/stats.json."""
    if stats is None:
        from src.io_utils import load_stats
        stats = load_stats()
    fit = stats.get("noise_var_fit")
    if not fit:
        raise KeyError(
            "artifacts/stats.json has no 'noise_var_fit'. Run:\n"
            "  python scripts/build_residual_bank.py --set data.root=<DATA> --refit")
    return VST(fit["a_mult"], fit["b_poisson"], fit["c_additive"], normalize=normalize)


# --------------------------------------------------------------------- selftest

def _selftest() -> int:
    """python -m src.vst  -- verifies the algebra without needing the dataset."""
    a, b, c = 0.023807, 0.010394, 0.0030539
    v = VST(a, b, c)
    print(v)

    # 1. round trip over the full observed NoisyLR range, and beyond it
    y = np.linspace(-0.5, 2.5, 100001).astype(np.float32)
    err = np.abs(v.inverse(v.forward(y)) - y).max()
    print(f"round-trip max abs error       : {err:.3e}")
    assert err < 1e-4, err

    # 2. anchors
    assert abs(float(v.forward(0.0))) < 1e-6
    assert abs(float(v.forward(1.0)) - 1.0) < 1e-6
    print("anchors f(0)=0, f(1)=1         : ok")

    # 3. derivative is 1/sigma  =>  transformed noise std is intensity-independent
    rng = np.random.default_rng(1337)
    print("\n  intensity   sigma(image)   sigma(VST)   ratio-to-mean")
    stds = []
    for mu in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95):
        s = float(v.sigma(mu))
        noisy = mu + s * rng.standard_normal(400_000).astype(np.float32)
        sv = float(np.std(v.forward(noisy)))
        stds.append((mu, s, sv))
    mean_sv = float(np.mean([s for _, _, s in stds]))
    for mu, s, sv in stds:
        print(f"  {mu:8.2f}   {s:12.5f}   {sv:10.5f}   {sv / mean_sv:12.4f}")

    spread_before = max(s for _, s, _ in stds) / min(s for _, s, _ in stds)
    spread_after = max(s for _, _, s in stds) / min(s for _, _, s in stds)
    print(f"\nnoise-std spread across intensity: {spread_before:.2f}x -> "
          f"{spread_after:.2f}x  (1.00x is perfect)")
    assert spread_after < 1.25, f"stabilisation failed: {spread_after:.3f}x"

    # 4. torch parity
    try:
        import torch
        t = torch.linspace(-0.4, 2.3, 5000)
        dt = float((torch.as_tensor(v.forward(t.numpy())) - v.forward(t)).abs().max())
        rt = float((v.inverse(v.forward(t)) - t).abs().max())
        print(f"torch/numpy agreement            : {dt:.3e}")
        print(f"torch round-trip                 : {rt:.3e}")
        assert dt < 1e-4 and rt < 1e-4
    except ImportError:
        print("torch not installed - skipped parity check")

    print("\nall VST checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
