# Round 2 — Experiment Log

All metrics measured on held-out **real pairs only**, never synthetic.

> **Runs 1 and 2/3 are not directly comparable.** Run 1 used the partial
> 1,325-pair dataset, whose OOD split was cluster 4 of that subset (297 images).
> Runs 2 and 3 use the full 4,785-pair dataset, OOD cluster 3 (1,165 images).
> Different validation sets with different intrinsic difficulty. **Compare gains
> over the matching baseline, not raw scores.**

---

## Baselines (bicubic upsampling)

| validation set | PSNR | SSIM | SSIM edge | SSIM flat | LPIPS (vgg) |
|---|---|---|---|---|---|
| partial data, cluster 4 (297 img) | 20.47 | 0.5079 | 0.5964 | 0.4784 | 0.4839 |
| **full data, val_ood cluster 3 (1,165 img)** | **20.94** | **0.4441** | **0.5269** | **0.4165** | **0.5248** |
| full data, val_id (362 img) | 20.31 | 0.5318 | 0.6239 | 0.5011 | — |

Note bicubic scores 0.5318 on `val_id` against 0.4441 on `val_ood`: **cluster 3 is
intrinsically harder content**, independent of any model. Raw scores across the
two sets are therefore not comparable; only gains over the matching baseline are.

---

## Runs

### Run 1 — partial data, 70% synthetic
`dim=64` · 926 real pairs + 3,460 synthetic · 80 epochs · `real_frac=0.30`

| metric | value | vs baseline |
|---|---|---|
| PSNR | 19.39 | **−1.08 dB** |
| SSIM | 0.5359 | +5.5% |
| LPIPS | 0.3661 | −24% |

Converged by epoch 25; final 30 epochs flat. **PSNR below baseline** — the model
was worse than plain bicubic on pixel accuracy.

### Run 2 — full data, 100% real
`dim=64` · 3,258 real pairs · 50 epochs · `synth_p=0.0`

| metric | value | vs baseline |
|---|---|---|
| PSNR | 23.94 | **+3.00 dB** |
| SSIM | 0.5034 | +13.4% |
| LPIPS | 0.3655 | −30% |
| SSIM edge / flat | 0.5597 / 0.4814 | edge > overall, no over-smoothing |

Epoch time 75s. Converged around epoch 29.

### Run 3 — full data, wider model
`dim=96` (2.14M params) · otherwise identical to Run 2

| metric | value | vs baseline |
|---|---|---|
| PSNR | 23.93 | +2.99 dB |
| SSIM | 0.5064 | +14.0% |
| LPIPS | 0.3577 | −32% |

Epoch time 103s (+37%). **2.2× the parameters bought +0.003 SSIM and no PSNR.**

In-distribution comparison for this checkpoint:

| | PSNR | SSIM | SSIM edge | LPIPS |
|---|---|---|---|---|
| val_id (familiar content) | 23.66 | 0.6281 | 0.6828 | 0.3146 |
| val_ood (unfamiliar) | 23.92 | 0.5064 | 0.5673 | 0.3701 |

PSNR is equal on both; SSIM is 24% higher on familiar content. Most of that raw
difference is content difficulty rather than generalisation — bicubic alone
scores 0.5318 on `val_id` against 0.4441 on `val_ood`. Comparing gains over the
matching baseline:

| | bicubic | model | gain |
|---|---|---|---|
| val_id (familiar) | 0.5318 | 0.6281 | **+18.1%** |
| val_ood (unfamiliar) | 0.4441 | 0.5064 | **+14.0%** |
| val_id PSNR | 20.31 | 23.66 | +3.35 dB |
| val_ood PSNR | 20.94 | 23.92 | +2.98 dB |

**Conclusion: a real but modest generalisation gap of roughly 4 percentage
points.** The model transfers to unfamiliar content, just less well than to
familiar content. Not a failure mode, and not large enough to justify chasing
external data through a synthetic pipeline that has already been shown to hurt.

---

### Runs 4 and 5 — texture-recovery experiments

Lost once to a session restart, then re-run to completion. Histories in
`results/round2_lpips02_history.json` and `results/round2_spectral3_history.json`.
The re-run reproduced the original epoch-for-epoch, confirming determinism.

**Final results (epoch 40):**

| | PSNR | SSIM | edge | flat | LPIPS | HF retained |
|---|---|---|---|---|---|---|
| bicubic | 20.94 | 0.4441 | 0.5269 | 0.4165 | 0.5248 | — |
| **baseline dim96** | **23.93** | **0.5044** | **0.5651** | **0.4842** | 0.3610 | 28% |
| lpips 0.2 | 23.66 | 0.4867 | 0.5473 | 0.4665 | **0.3448** | 48% |
| spectral 3.0 | 23.46 | 0.4899 | 0.5506 | 0.4696 | 0.3699 | 72% |

**Verdict: neither experiment beat the baseline.** `lpips 0.2` buys 4.5% on LPIPS
for 0.27 dB PSNR and 0.018 SSIM. `spectral 3.0` loses on every metric.

### The finding that explains the plateau

Visual inspection showed both variants looking nearly identical to each other
despite 48% vs 72% "high-frequency retention" — which prompted checking what that
metric was actually measuring.

**The ground truth images contain their own white-noise floor.** The GT radial
spectrum flattens above roughly bin 20 (tail flatness std/mean = **0.024**):

| bin | GT power | est. noise floor | real structure |
|---|---|---|---|
| 14 | 13.23 | ~10.3 | 2.9 |
| 20 | 11.26 | ~10.3 | 0.9 |
| 30 | 10.42 | ~10.3 | ~0 |

So above bin 20, essentially **all** of the ground truth's high-frequency energy
is sensor noise, not structure. The "HF retained" metric was largely measuring
*how much noise each model reproduces*, not how much detail it recovers.

This inverts the interpretation: the spectral model's 72% is not superior
recovery, it is emitting more noise-like energy — which cannot correlate with the
GT's actual noise and therefore only hurts pixel accuracy. That is exactly why
its PSNR was the worst of the three. **The baseline at 28% was behaving
correctly**: a good denoiser should decline to reproduce unpredictable noise.

A follow-up metric subtracting the noise floor returned 5.6% / 5.8%, but that
number is unstable — model output falls below the floor across most of the band,
so the subtraction clips to zero and amplifies estimation error. Not trustworthy;
recorded only so nobody repeats it.

**Implication:** a meaningful fraction of the residual error against ground truth
is irreducible. This bounds what any model can achieve on this dataset and
explains why capacity, data volume and loss modifications all plateaued at
similar points.

### Not yet tried (as of version 3)

Reviewing what has actually been explored: data volume (large gain, banked),
model *width* (no gain), and two loss modifications (marginal). Several
well-motivated levers remain untouched:

1. **Test-time augmentation** — `src/tta.py` is implemented and verified but has
   never been measured. No retraining required.
2. **Ensembling** the three existing checkpoints — free, errors are partly
   uncorrelated.
3. **Deeper U-Net.** The current architecture has only **one** downsampling level
   (`enc1 → down → enc2 → middle → up → dec1`). NAFNet normally uses three or
   four. Receptive field is therefore small, which likely explains why increasing
   *width* bought nothing — width does not extend receptive field.
4. **MS-SSIM loss term.** SSIM is scored by KLA but never optimised directly;
   the loss is Charbonnier + Sobel + LPIPS.
5. **Larger training patches** (`lr_patch=128`) — removes the train/validation
   receptive-field mismatch. Proposed twice, never run.

---

### Original (lost) run notes, kept for the record

**Motivation.** Visual inspection of Run 3 showed the model erasing real surface
texture along with the noise. Quantified: it retained only **28% of the ground
truth's high-frequency power**. Note that edge SSIM stayed *above* overall SSIM
throughout, which we had been reading as "no over-smoothing" — that heuristic
works on natural photographs where flat regions are genuinely featureless, but
on SEM images the fine texture lives in exactly those mid-gradient regions. The
metric that reflected the damage was **flat SSIM**, the lowest of the three.

**Run 4 — `loss.lpips` 0.05 -> 0.2** (`dim=64`, 40 epochs)

| epoch | loss | PSNR | SSIM | edge | flat | LPIPS |
|---|---|---|---|---|---|---|
| 1 | 0.3999 | 23.20 | 0.4760 | 0.5399 | 0.4547 | 0.4172 |
| 13 | 0.3364 | 23.60 | 0.4818 | 0.5408 | 0.4621 | 0.3614 |
| 18 | 0.3319 | 23.55 | 0.4864 | 0.5483 | 0.4657 | 0.3587 |
| 20 | 0.3315 | 23.61 | 0.4849 | 0.5459 | 0.4646 | 0.3520 |

Against baseline at matching epochs: PSNR about −0.24 dB, SSIM about −0.014,
LPIPS **5–10% better** and still improving at epoch 20. The expected
perception–distortion trade, behaving as designed.

**Run 5 — `loss.spectral` (`dim=64`, 40 epochs)**

First attempt at weight 0.3 was a silent no-op: the spectral term computes to
~0.032, so 0.3x it contributed 0.0096 to a ~0.29 total loss — **3% of the
objective**, hence 3% of the gradient. Epoch-1 loss matched the baseline to four
decimal places, which is how we caught it.

Relaunched at **weight 3.0**: epoch-1 loss 0.4453 (versus baseline 0.2882),
confirming the term was finally a real fraction of the objective. PSNR 23.17,
SSIM 0.4789 at epoch 1 — nothing destabilised.

**Lesson worth keeping:** loss weights cannot be chosen without knowing the
magnitudes of the terms being combined. LPIPS returns ~0.4 and the spectral term
~0.03, so weights that look comparable differ by an order of magnitude in effect.

**Also flagged:** the spectral loss slices the upper rows of an unshifted FFT to
isolate high frequencies, but negative frequencies wrap to the bottom rows, so
the "high band" accidentally includes low frequencies. Dilutes the effect;
worth fixing before drawing conclusions from a weak result.

---

## Findings

**Synthetic data hurt, despite passing validation.** Run 1 versus Run 2 is a
4.1 dB swing in PSNR relative to baseline. The synthetic pipeline had been
validated to within 1.2% on spatial autocorrelation and 0.1% on local texture
energy — matched second-order statistics were not sufficient for pixel-exact
reconstruction. Confounded with data volume (926 → 3,258 real pairs), so the two
effects cannot be fully separated, but PSNR being the worst-affected metric
points at degradation mismatch rather than volume alone.

**Capacity is not the bottleneck.** Two models differing 2.2× in size converge to
within 0.006 SSIM of each other. The limit lies in the data or the task, not the
network. `dim=64` is the submission candidate: equal quality, 37% faster, and
throughput is scored.

**No over-smoothing in any run.** Edge SSIM stayed above overall SSIM throughout,
which is the failure mode the KLA specification names explicitly.

---

## Configuration for Runs 2 and 3

```
kernel            gauss:[0.5, 0.6, 0.7] weighted [0.25, 0.50, 0.25]
noise var fit     2.3807e-02*mu^2 + 1.0394e-02*mu + 3.0539e-03
residual bank     2,637,598 samples, skew 0.813, excess kurtosis 3.919
grad_thresh       0.1162 (GT p40)
loss              1.0 * edge-weighted Charbonnier + 0.5 * Sobel + 0.05 * LPIPS(vgg)
edge weighting    1 + 4*(grad/max), normalised to mean 1
augmentation      D4, CutBlur p=0.5, scale jitter 0.7-1.4
optimiser         AdamW lr 5e-4, cosine to 1e-6, weight decay 1e-4, grad clip 1.0
precision         AMP
split             6 clusters, OOD = cluster 3, seed 1337
```

---

## Open items

- [x] bicubic on `val_id` — done. Gap is ~4 percentage points, modest.
- [x] ~~loss ablation~~ — done via runs 4 and 5. Neither beat baseline.
- [ ] **test-time augmentation** — implemented in `src/tta.py`, never measured
- [ ] **ensemble** the three existing checkpoints
- [ ] **deeper U-Net** (2-3 levels rather than 1) — most promising untried change
- [ ] **MS-SSIM loss term** — SSIM is scored but never directly optimised
- [ ] `lr_patch=128` — removes the train/validation receptive-field mismatch
- [x] ~~NFFA-EUROPE external data~~ — **dropped.** Usable only through synthetic
      degradation, which cost 4 dB in Run 1. Not worth a 13.6 GB download to
      chase a 4-point gap using the technique that already failed.
- [ ] 512x512 forward pass has still never been tested
- [ ] final retrain on all 4,785 pairs once decisions are locked

## Checkpoint inventory

| run | location | status |
|---|---|---|
| Run 2, dim=64 | `artifacts_full/` | **LOST** — deleted by a repo re-clone |
| Run 3, dim=96 | `artifacts_dim96/` | preserved, saved to Kaggle dataset |
| Round 1 model | `weights/best_nafnet.pt` (in git) | trained on natural photographs, **not** round 2 |

**Warning:** `run.py` defaults to `weights/best_nafnet.pt`, which currently holds
the round-1 model. Always pass `--weights` explicitly until that file is
deliberately replaced with the final round-2 checkpoint.

**Lesson:** write training output to `/kaggle/working/runs/<name>`, outside the
repo directory, so re-cloning cannot delete it.

**Second lesson, learned the hard way:** `/kaggle/working` does not survive a
session restart. Runs 4 and 5 completed 40 epochs each and were lost entirely.
**Download the checkpoint and history the moment a run finishes** — do not leave
them in the working directory. Two files, thirty seconds.

## Post-run checklist (do this every time)

```python
# immediately after training completes
import shutil, os
os.makedirs("/kaggle/working/keep", exist_ok=True)
for run in ["lpips02", "spectral3"]:
    for f in ["best_nafnet.pt", "history.json"]:
        src = f"runs/{run}/{f}"
        if os.path.exists(src):
            shutil.copy(src, f"/kaggle/working/keep/{run}_{f}")
print(os.listdir("/kaggle/working/keep"))
```

Then download those files from the Output panel straight away, and commit the
`history.json` files to git — they are a few KB and hold every metric.
