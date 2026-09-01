"""Test-time augmentation via the dihedral (D4) group.

Idea: the restoration of a rotated image should equal the rotation of the
restored image. The model does not know that exactly, so its errors differ
between orientations. Averaging over all 8 orientations cancels part of that
error while the consistent signal reinforces - a free accuracy gain requiring
no retraining.

The cost is linear in the number of transforms, and inference throughput is
scored, so `n_transforms=4` is offered as a middle option: it uses the four
rotations only and typically captures most of the benefit.

Valid here because SEM texture has no preferred global orientation, so a rotated
SEM image is still a plausible SEM image.
"""
from __future__ import annotations

import torch


def _apply(x: torch.Tensor, i: int) -> torch.Tensor:
    """Transform i of the dihedral group: rotate by i%4 quarter turns, then
    optionally mirror."""
    y = torch.rot90(x, i % 4, (-2, -1))
    return y.flip(-1) if i // 4 else y


def _invert(y: torch.Tensor, i: int) -> torch.Tensor:
    """Exact inverse of _apply. Mirror first, then rotate back."""
    if i // 4:
        y = y.flip(-1)
    return torch.rot90(y, -(i % 4), (-2, -1))


@torch.no_grad()
def tta_forward(model, x: torch.Tensor, n_transforms: int = 8,
                clamp: tuple[float, float] | None = (0.0, 1.0)) -> torch.Tensor:
    """Average the model's predictions over n_transforms D4 orientations.

    n_transforms:  1 = no TTA, 2 = identity + 180 deg, 4 = rotations only,
                   8 = full group.

    The n=2 option exists because the measured 1 -> 4 gain (+0.18 dB, +0.011
    SSIM) is large while 4 -> 8 is negligible (+0.0003 SSIM), which leaves open
    where between 1 and 4 the benefit actually arrives. Two transforms cost
    roughly 19 ms/image against 34 ms for four.

    Rotations and flips commute with 2x upsampling, so the same transform index
    applies to input and output despite the resolution change.
    """
    if n_transforms <= 1:
        out = model(x)
        return out.clamp(*clamp) if clamp else out

    # 180 deg (index 2) is chosen for n=2 because it shares no axis alignment
    # with the identity, so the two error realisations are maximally different.
    idx = {2: (0, 2), 4: (0, 1, 2, 3)}.get(n_transforms, tuple(range(8)))
    acc = None
    for i in idx:
        pred = _invert(model(_apply(x, i)), i)
        acc = pred if acc is None else acc + pred
    out = acc / len(idx)
    return out.clamp(*clamp) if clamp else out


@torch.no_grad()
def compare_tta(model, loader, device, lpips_fn=None, variants=(1, 2, 4, 8)) -> dict:
    """Measure quality and wall-clock cost for each TTA setting.

    Returns {n: {psnr, ssim, ssim_edge, lpips, seconds, ms_per_image}} so the
    quality/throughput trade-off can be judged on evidence rather than assumed.
    """
    import time

    import torch.nn.functional as F

    from .eval_utils import stratified_ssim

    model.eval()
    results = {}
    for n in variants:
        acc = {"psnr": 0.0, "ssim": 0.0, "ssim_edge": 0.0, "ssim_flat": 0.0, "lpips": 0.0}
        count = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for batch in loader:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            pred = tta_forward(model, lr, n)

            mse = F.mse_loss(pred, hr, reduction="none").mean(dim=(1, 2, 3))
            acc["psnr"] += float((10 * torch.log10(1.0 / mse.clamp_min(1e-12))).sum())
            if lpips_fn is not None:
                acc["lpips"] += float(lpips_fn(pred.repeat(1, 3, 1, 1) * 2 - 1,
                                               hr.repeat(1, 3, 1, 1) * 2 - 1).sum())
            p, h = pred.cpu().numpy(), hr.cpu().numpy()
            for j in range(p.shape[0]):
                s = stratified_ssim(p[j, 0], h[j, 0])
                acc["ssim"] += s["ssim"]
                acc["ssim_edge"] += s["ssim_edge"]
                acc["ssim_flat"] += s["ssim_flat"]
            count += lr.shape[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        results[n] = {k: v / max(count, 1) for k, v in acc.items()}
        results[n]["seconds"] = dt
        results[n]["ms_per_image"] = 1000 * dt / max(count, 1)
    return results
