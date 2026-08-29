"""Training dataset.

Mixed source resolutions (128->256 and 256->512) are unified by cropping every
sample to a fixed LR patch, so both groups are indistinguishable inside a batch.
The network is fully convolutional, so full-size inference is unaffected.

Crops are drawn with gradient-based rejection sampling: wafer imagery is mostly
flat die area and uniform random crops would spend most of training on blank
regions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import scale_jitter
from .degrade import degrade
from .transforms import load_stats, normalize


def degrade_cfg_from_stats(stats=None, width=1.0, jitter=0.30,
                           residual_bank=None) -> dict:
    """Build a degrade() config from the measured constants in stats.json.

    residual_bank : path to a .npz from scripts/build_residual_bank.py, or an
                    already-loaded bank. Strongly recommended for round 2 - the
                    noise is right-skewed with excess kurtosis 2.08, and a
                    Gaussian generator produces ~6x too few 5-sigma outliers.
    """
    s = stats or load_stats()
    bank = residual_bank
    if isinstance(bank, (str, Path)):
        from .noise_fit import load_residual_bank
        bank = load_residual_bank(bank)
    return {
        "residual_bank": bank,
        "width": width,
        "jitter": jitter,
        "kernels": s.get("kernels", ["area", "bicubic_aa", "lanczos", "bilinear_aa"]),
        "kernel_p": s.get("kernel_p", [0.3, 0.3, 0.2, 0.2]),
        "order_mix": s.get("order_mix", [0.5, 0.5]),
        "noise_var_fit": s.get("noise_var_fit"),
        "meas_mult": s.get("meas_mult", 0.0),
        "meas_add": s.get("meas_add", 0.0),
    }


def _grad_energy(patch: np.ndarray) -> float:
    return float(np.abs(np.diff(patch, axis=0)).mean() + np.abs(np.diff(patch, axis=1)).mean())


class RestorationDataset(Dataset):
    def __init__(self, cache_dir, stems=None, lr_patch=64, scale=2,
                 grad_thresh=0.0, crop_tries=8, train=True, seed=1337,
                 synth_p=0.0, degrade_cfg=None, jitter_range=(0.7, 1.4), margin=8,
                 use_gt_only=True, gt_only_stems=None, real_frac=None):
        """
        synth_p       probability a PAIRED sample is synthesised from its GT
                      instead of using the provided degraded counterpart
        degrade_cfg   from degrade_cfg_from_stats(); required for any synthesis
        jitter_range  GT rescaling applied BEFORE degradation. This is the main
                      defence against unfamiliar feature scales - the training
                      set is entirely 256->128, while evaluation may include
                      512x512 content.
        margin        HR pixels of context kept around the crop while degrading,
                      then trimmed, so the resampling kernel never sees the crop
                      boundary. Measured effect on border-row statistics was
                      small (<0.2%), so this is cheap insurance rather than a
                      fix for an observed problem.

        use_gt_only   include the unpaired ground truths (kind="gt_only" in the
                      cache index). Round 2 ships 4,785 GT against 1,325 pairs,
                      so ~72% of clean images are reachable ONLY through
                      synthesis. Leave this on for training, off for validation.
        gt_only_stems restrict the unpaired pool (None = all of it)
        real_frac     target proportion of REAL pairs per epoch. None keeps the
                      dataset's natural ratio (1,325 : 3,460 ~= 28% real).
                      Setting e.g. 0.3 repeats paired items to hit that share.
        """
        self.cache_dir = Path(cache_dir)
        with open(self.cache_dir / "index.json") as f:
            self.index = json.load(f)

        self.lr_patch, self.scale = lr_patch, scale
        self.grad_thresh, self.crop_tries = grad_thresh, crop_tries
        self.train = train
        self.seed = seed
        self._rng = None                       # created per worker, see _ensure_rng
        self.synth_p = float(synth_p)
        self.degrade_cfg = degrade_cfg
        self.jitter_range = tuple(jitter_range)
        self.margin = int(margin) - int(margin) % scale     # keep divisible by scale

        keep = set(stems) if stems is not None else None
        keep_go = set(gt_only_stems) if gt_only_stems is not None else None

        self._groups = []
        paired_items, gtonly_items = [], []
        for gi, g in enumerate(self.index):
            kind = g.get("kind", "paired")
            gt_mm = np.load(self.cache_dir / g["gt_file"], mmap_mode="r")
            lr_mm = (np.load(self.cache_dir / g["lr_file"], mmap_mode="r")
                     if g.get("lr_file") else None)
            self._groups.append((gt_mm, lr_mm, g))

            if kind == "gt_only":
                if not (use_gt_only and train):
                    continue
                for i, stem in enumerate(g["stems"]):
                    if keep_go is None or stem in keep_go:
                        gtonly_items.append((gi, i, stem))
            else:
                for i, stem in enumerate(g["stems"]):
                    if keep is None or stem in keep:
                        paired_items.append((gi, i, stem))

        if gtonly_items and self.degrade_cfg is None:
            raise ValueError(
                "gt_only images require degrade_cfg - they have no real degraded "
                "counterpart and can only be used through degrade(). "
                "Pass degrade_cfg_from_stats() or set use_gt_only=False.")
        if self.synth_p > 0 and self.degrade_cfg is None:
            raise ValueError("synth_p > 0 requires degrade_cfg (see degrade_cfg_from_stats)")

        # Realise the real/synthetic mix as an explicit item list so ordinary
        # samplers and DDP work unchanged, rather than drawing randomly per call.
        if real_frac is not None and paired_items and gtonly_items:
            target = float(real_frac)
            n_slots = max(1, int(round(len(gtonly_items) * target / max(1e-9, 1 - target))))
            reps = int(np.ceil(n_slots / len(paired_items)))
            paired_items = (paired_items * reps)[:n_slots]

        self.paired_items, self.gtonly_items = paired_items, gtonly_items
        self.items = [(gi, i, s, False) for gi, i, s in paired_items] + \
                     [(gi, i, s, True) for gi, i, s in gtonly_items]

    def __len__(self) -> int:
        return len(self.items)

    def composition(self) -> dict:
        n = max(1, len(self.items))
        return {"total": len(self.items),
                "paired_slots": len(self.paired_items),
                "gt_only_slots": len(self.gtonly_items),
                "real_share": round(len(self.paired_items) / n, 4)}

    def _ensure_rng(self):
        """Per-worker RNG.

        A single RNG created in __init__ is COPIED into every dataloader worker,
        so all workers would draw identical crops and identical noise. Seed by
        worker id instead.
        """
        if self._rng is None:
            try:
                import torch.utils.data as _tud

                info = _tud.get_worker_info()
                wid = info.id if info is not None else 0
            except Exception:
                wid = 0
            self._rng = np.random.default_rng([self.seed, wid])
        return self._rng

    def set_width(self, width: float) -> None:
        """Curriculum knob: ramp synthetic degradation variety 0.3 -> 1.0."""
        if self.degrade_cfg is not None:
            self.degrade_cfg = {**self.degrade_cfg, "width": float(width)}

    def _read(self, gi: int, i: int):
        """Return (gt, lr). lr is None for unpaired ground truths."""
        gt_mm, lr_mm, g = self._groups[gi]
        gt = np.asarray(gt_mm[i], dtype=np.float32)
        if g["gt_dtype"] == "uint16":
            gt = gt / 65535.0
        lr = np.asarray(lr_mm[i], dtype=np.float32) if lr_mm is not None else None
        return gt, lr

    def _crop_hr(self, hr: np.ndarray, size: int):
        """Gradient-rejection crop in HR space (used by the synthetic path)."""
        rng = self._ensure_rng()
        H, W = hr.shape
        if H <= size or W <= size:
            pad_h, pad_w = max(0, size - H), max(0, size - W)
            hr = np.pad(hr, ((0, pad_h), (0, pad_w)), mode="reflect")
            H, W = hr.shape
        best = None
        for _ in range(self.crop_tries):
            y = int(rng.integers(0, H - size + 1))
            x = int(rng.integers(0, W - size + 1))
            e = _grad_energy(hr[y:y + size, x:x + size])
            if e >= self.grad_thresh:
                best = (e, y, x)
                break
            if best is None or e > best[0]:
                best = (e, y, x)
        _, y, x = best
        return hr[y:y + size, x:x + size]

    def _synth(self, gt_full: np.ndarray):
        """Generate a fresh (lr, hr) pair from a clean GT image.

        Order matters: rescale -> crop with margin -> degrade -> trim margin.
        Degrading before cropping would waste work; cropping without a margin
        would bake resampling edge artefacts into every training patch.
        """
        rng = self._ensure_rng()
        s, m = self.scale, self.margin
        gt = scale_jitter(gt_full, rng, *self.jitter_range)

        hr_size = self.lr_patch * s
        crop = self._crop_hr(gt, hr_size + 2 * m)
        lr = degrade(crop, rng, self.degrade_cfg)

        lm = m // s
        lr = lr[lm:lm + self.lr_patch, lm:lm + self.lr_patch]
        hr = crop[m:m + hr_size, m:m + hr_size]
        return hr, lr

    def _crop(self, gt: np.ndarray, lr: np.ndarray):
        rng = self._ensure_rng()
        p, s = self.lr_patch, self.scale
        H, W = lr.shape
        if H <= p or W <= p:
            return gt, lr
        best = None
        for _ in range(self.crop_tries):
            y = int(rng.integers(0, H - p + 1))
            x = int(rng.integers(0, W - p + 1))
            # Measure structure on the CLEAN patch. Round-2 analysis showed noise
            # roughly doubles apparent gradient energy on the LR image (median
            # 0.263 vs 0.133 on GT), so selecting on LR selects on the noise
            # realisation rather than on content - which makes rejection
            # sampling equivalent to uniform random sampling.
            e = _grad_energy(gt[y * s:(y + p) * s, x * s:(x + p) * s])
            if e >= self.grad_thresh:
                best = (e, y, x)
                break
            if best is None or e > best[0]:
                best = (e, y, x)
        _, y, x = best  # always terminates
        return (gt[y * s:(y + p) * s, x * s:(x + p) * s], lr[y:y + p, x:x + p])

    def __getitem__(self, idx: int):
        gi, i, stem, is_gt_only = self.items[idx]
        gt, lr = self._read(gi, i)

        if is_gt_only or lr is None:
            # No real degraded counterpart exists: synthesis is the only route.
            gt, lr = self._synth(gt)
            synthetic = True
        elif self.train:
            if self.synth_p > 0 and float(self._ensure_rng().random()) < self.synth_p:
                gt, lr = self._synth(gt)
                synthetic = True
            else:
                gt, lr = self._crop(gt, lr)
                synthetic = False
        else:
            synthetic = False

        return {
            "lr": torch.from_numpy(np.ascontiguousarray(normalize(lr))).unsqueeze(0),
            "hr": torch.from_numpy(np.ascontiguousarray(normalize(gt))).unsqueeze(0),
            "stem": stem,
            "synthetic": synthetic,
        }


def estimate_grad_threshold(cache_dir, percentile=40, lr_patch=64, scale=2,
                            n=2000, seed=0) -> float:
    """Measure the crop-gradient distribution on the CLEAN (GT) patches.

    Must match what _crop measures, otherwise the threshold is on the wrong
    scale entirely. Measuring on noisy LR patches inflates the values (noise
    contributes gradient of its own) and the resulting threshold rejects
    nothing useful.

    Round-2 reference values, GT 128px crops:
        p10 0.052 · p25 0.094 · p40 0.122 · p50 0.140 · p75 0.184 · p90 0.223
    """
    ds = RestorationDataset(cache_dir, lr_patch=lr_patch, train=False, seed=seed)
    rng = np.random.default_rng(seed)
    hr_patch = lr_patch * scale
    vals = []
    for _ in range(n):
        gi, i, _, _ = ds.items[int(rng.integers(0, len(ds.items)))]
        gt, _ = ds._read(gi, i)
        H, W = gt.shape
        if H <= hr_patch or W <= hr_patch:
            continue
        y = int(rng.integers(0, H - hr_patch + 1))
        x = int(rng.integers(0, W - hr_patch + 1))
        vals.append(_grad_energy(gt[y:y + hr_patch, x:x + hr_patch]))
    return float(np.percentile(vals, percentile)) if vals else 0.0
