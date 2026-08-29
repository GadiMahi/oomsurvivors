"""Augmentation, ordered by value for THIS challenge.

KLA states the hidden test set contains unfamiliar image *content* but the same
degradation mechanisms. So content diversity matters more than noise diversity:

  1. scale_jitter - varies feature density. Primary generalisation lever.
  2. d4           - free, and valid for axis-aligned semiconductor layouts.
  3. cutblur      - teaches how much / where to restore; counters blurring.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def scale_jitter(gt: np.ndarray, rng, lo: float = 0.7, hi: float = 1.4) -> np.ndarray:
    """Resize the clean GT before degrading it, changing apparent feature size."""
    f = float(rng.uniform(lo, hi))
    if abs(f - 1.0) < 1e-3:
        return gt
    h, w = gt.shape[-2:]
    size = (max(8, int(round(w * f))), max(8, int(round(h * f))))
    out = Image.fromarray(gt.astype(np.float32), mode="F").resize(size, Image.BICUBIC)
    return np.asarray(out, dtype=np.float32)


def d4(lr: torch.Tensor, hr: torch.Tensor, generator=None):
    """Random element of the dihedral group. Apply on GPU - it is free there."""
    k = int(torch.randint(0, 4, (1,), generator=generator))
    flip = bool(torch.randint(0, 2, (1,), generator=generator))
    lr, hr = torch.rot90(lr, k, (-2, -1)), torch.rot90(hr, k, (-2, -1))
    if flip:
        lr, hr = lr.flip(-1), hr.flip(-1)
    return lr.contiguous(), hr.contiguous()


def d4_batch(lr: torch.Tensor, hr: torch.Tensor, generator=None):
    """Apply one dihedral transform to a whole batch, on GPU. Effectively free.

    Valid here because SEM texture has no preferred global orientation (the 2D
    FFT is close to isotropic), so the transformed images stay in-distribution.
    """
    k = int(torch.randint(0, 4, (1,), generator=generator))
    flip = bool(torch.randint(0, 2, (1,), generator=generator))
    lr, hr = torch.rot90(lr, k, (-2, -1)), torch.rot90(hr, k, (-2, -1))
    if flip:
        lr, hr = lr.flip(-1), hr.flip(-1)
    return lr.contiguous(), hr.contiguous()


def cutblur_sr(lr: torch.Tensor, hr: torch.Tensor, scale: int = 2,
               p: float = 0.5, alpha: float = 0.5, generator=None):
    """CutBlur adapted for a network that upsamples internally.

    The original formulation needs input and target at the same resolution. Our
    model takes 64px and emits 128px, so instead we replace a rectangle of the
    NOISY input with a clean, noise-free downsample of the corresponding target
    region. The model then sees patches where part of the frame needs no
    restoration and must learn where and how much to restore, rather than
    applying a fixed amount everywhere - which is the mechanism that makes
    CutBlur reduce over-smoothing.
    """
    if float(torch.rand(1, generator=generator)) > p:
        return lr, hr

    h, w = lr.shape[-2:]
    cut = float(torch.empty(1).uniform_(0.15, alpha, generator=generator))
    ch, cw = max(1, int(h * cut)), max(1, int(w * cut))
    y = int(torch.randint(0, max(1, h - ch + 1), (1,), generator=generator))
    x = int(torch.randint(0, max(1, w - cw + 1), (1,), generator=generator))

    hr_region = hr[..., y * scale:(y + ch) * scale, x * scale:(x + cw) * scale]
    clean_lr = torch.nn.functional.avg_pool2d(hr_region, scale)   # noise-free
    out = lr.clone()
    out[..., y:y + ch, x:x + cw] = clean_lr
    return out, hr


def cutblur(lr_up: torch.Tensor, hr: torch.Tensor, p: float = 0.7, alpha: float = 0.7,
            generator=None):
    """CutBlur (Yoo et al., CVPR 2020), adapted to a batch of tensors.

    lr_up must already be bicubically upsampled to the HR size.
    """
    if float(torch.rand(1, generator=generator)) > p:
        return lr_up, hr

    h, w = hr.shape[-2:]
    cut = float(torch.empty(1).uniform_(0.0, alpha, generator=generator))
    ch, cw = max(1, int(h * cut)), max(1, int(w * cut))
    y = int(torch.randint(0, max(1, h - ch + 1), (1,), generator=generator))
    x = int(torch.randint(0, max(1, w - cw + 1), (1,), generator=generator))

    out = lr_up.clone()
    if float(torch.rand(1, generator=generator)) < 0.5:
        out[..., y:y + ch, x:x + cw] = hr[..., y:y + ch, x:x + cw]
    else:
        out, keep = hr.clone(), lr_up[..., y:y + ch, x:x + cw]
        out[..., y:y + ch, x:x + cw] = keep
    return out, hr
