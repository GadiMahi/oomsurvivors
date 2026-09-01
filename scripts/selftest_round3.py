#!/usr/bin/env python3
"""Verify the round-3 additions before spending GPU hours on them.

Run this FIRST on Kaggle. It needs no dataset and no checkpoint - everything is
synthetic - and it takes about ten seconds. If any check fails, a training run
would waste hours producing a result you could not trust.

    python scripts/selftest_round3.py

Checks:
  1. VST round-trips exactly and actually flattens noise variance
  2. VST behaves identically in numpy and torch, including under autocast
  3. MS-SSIM is 0 for identical images, positive otherwise, and differentiable
  4. MS-SSIM degrades monotonically as an image is blurred
  5. TTA n=2 and n=4 average the correct number of orientations
  6. The transform composition used at inference is an exact identity
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS, FAIL = "  PASS", "  FAIL"
failures = []


def check(name, cond, detail=""):
    print(f"{PASS if cond else FAIL}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def main() -> int:
    torch.manual_seed(1337)
    from src.vst import VST, vst_from_stats

    print("\n[1] VST algebra")
    try:
        v = vst_from_stats()
        print(f"       built from artifacts/stats.json: {v}")
    except Exception as e:
        print(f"       stats.json unavailable ({type(e).__name__}), using literals")
        v = VST(0.023807, 0.010394, 0.0030539)

    y = torch.linspace(-0.5, 2.5, 200_001)
    rt = float((v.inverse(v.forward(y)) - y).abs().max())
    check("round-trip is exact", rt < 1e-4, f"max err {rt:.2e}")
    check("f(0) == 0", abs(float(v.forward(0.0))) < 1e-6)
    check("f(1) == 1", abs(float(v.forward(1.0)) - 1.0) < 1e-6)
    check("monotonic", bool((v.forward(y).diff() > 0).all()))

    print("\n[2] VST stabilises variance, and numpy == torch")
    g = torch.Generator().manual_seed(7)
    stds = []
    for mu in (0.05, 0.3, 0.6, 0.95):
        s = float(np.sqrt(v.a * mu * mu + v.b * mu + v.c))
        noisy = mu + s * torch.randn(300_000, generator=g)
        stds.append(float(v.forward(noisy).std()))
    spread = max(stds) / min(stds)
    check("noise std constant across intensity", spread < 1.25, f"{spread:.3f}x spread")

    t = torch.linspace(-0.4, 2.3, 5000)
    d = float((torch.as_tensor(v.forward(t.numpy())) - v.forward(t)).abs().max())
    check("numpy and torch agree", d < 1e-4, f"max diff {d:.2e}")

    if torch.cuda.is_available():
        tc = t.cuda()
        with torch.autocast("cuda", enabled=True):
            rt_amp = float((v.inverse(v.forward(tc)) - tc).abs().max())
        check("survives autocast (fp16)", rt_amp < 1e-3, f"max err {rt_amp:.2e}")
    else:
        print("       no CUDA - skipped autocast check")

    print("\n[3] MS-SSIM loss")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train import MSSSIMLoss

    mss = MSSSIMLoss()
    a = torch.rand(2, 1, 128, 128)
    check("identical images give ~0 loss", float(mss(a, a)) < 1e-3,
          f"{float(mss(a, a)):.2e}")
    b = torch.rand(2, 1, 128, 128)
    check("different images give positive loss", float(mss(a, b)) > 0.1,
          f"{float(mss(a, b)):.4f}")

    a.requires_grad_(True)
    loss = mss(a, b)
    loss.backward()
    check("gradients flow and are finite",
          a.grad is not None and bool(torch.isfinite(a.grad).all()))

    print("\n[4] MS-SSIM tracks degradation monotonically")
    import torch.nn.functional as F
    base = torch.rand(1, 1, 256, 256)
    k = torch.ones(1, 1, 5, 5) / 25.0
    prev, vals, mono = -1.0, [], True
    cur = base.clone()
    for i in range(4):
        val = float(mss(cur, base))
        vals.append(val)
        if val < prev:
            mono = False
        prev = cur_val = val
        cur = F.conv2d(F.pad(cur, (2, 2, 2, 2), mode="reflect"), k)
    check("loss rises as image is blurred", mono,
          " -> ".join(f"{x:.4f}" for x in vals))

    print("\n[5] TTA orientation counts")
    from src.tta import _apply, _invert, tta_forward

    x = torch.rand(1, 1, 32, 32)
    inv_ok = all(bool(torch.allclose(_invert(_apply(x, i), i), x)) for i in range(8))
    check("all 8 transforms invert exactly", inv_ok)

    seen = set()
    for i in range(8):
        seen.add(_apply(x, i).numpy().tobytes())
    check("8 transforms are distinct", len(seen) == 8, f"{len(seen)} unique")

    calls = {"n": 0}

    class Counter(torch.nn.Module):
        def forward(self, z):
            calls["n"] += 1
            return F.interpolate(z, scale_factor=2, mode="nearest")

    for n in (1, 2, 4, 8):
        calls["n"] = 0
        tta_forward(Counter(), x, n)
        check(f"tta={n} runs the model {n}x", calls["n"] == n, f"got {calls['n']}")

    print("\n[6] Inference-time composition is an identity")
    # This is what run.py wraps around a VST-trained model. If the composition
    # is not an exact identity for a perfect model, the wrapper is wrong.
    ident = torch.nn.Identity()
    z = torch.rand(2, 1, 16, 16) * 1.5 - 0.2
    out = v.inverse(ident(v.forward(z)))
    err = float((out - z).abs().max())
    check("inverse(model(forward(x))) == x for identity model", err < 1e-4,
          f"max err {err:.2e}")

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all checks passed - safe to train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
