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
- [ ] test-time augmentation (D4 self-ensemble) — no retraining required
- [ ] `lr_patch=128` — removes the train/validation receptive-field mismatch
- [ ] loss ablation at `edge_weight_alpha=1.0, edge=0.15`
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
