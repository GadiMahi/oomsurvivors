# Round 2 — Dataset Analysis Report

**Dataset:** `semicon_train_data` (SEM micrographs)
**Analysed:** 1,325 verified pairs + 4,785 ground-truth images
**Author:** Mahi · Data / preprocessing

---

## 1. Executive summary

The round-2 dataset is the **same restoration problem in a different image domain, with stronger
noise and a deliberately unbalanced structure**.

Three findings drive every recommendation in this report:

1. **The file format and degradation family are unchanged**, so the existing pipeline —
   loaders, normalisation, inference harness, `degrade.py` — transfers without modification.
   Only fitted constants change.
2. **KLA supplied 4,785 clean images but only 1,326 paired examples.** This is intentional.
   The remaining ~3,459 ground truths are unusable unless we degrade them ourselves, which
   makes synthetic pair generation the intended solution path rather than an optional
   optimisation. Roughly **72% of available training images can only enter training through
   `degrade.py`.**
3. **The noise is heavy-tailed, not Gaussian** (excess kurtosis 2.08; 3σ events at 4× the
   Gaussian rate). Because most training data will now be synthetic, generator fidelity
   directly determines training quality, so this moves from a refinement to a priority.

Round 2 is also a materially **harder** problem: the bicubic baseline drops from
22.70 dB / 0.553 SSIM in round 1 to **20.47 dB / 0.508 SSIM** here. Round-1 model scores are
therefore not comparable to round-2 scores and must not be quoted side by side.

---

## 2. Dataset composition

| Item | Value |
|---|---|
| Ground-truth images | 4,785 |
| NoisyLR images | 1,326 (1,325 usable, 1 corrupt from download truncation) |
| Paired examples | 1,325 |
| GT-only images (synthetic source) | 3,459 |
| Resolution | GT 256×256, LR 128×128 — **all pairs exactly 2×** |
| Format | `.npy`, float32, single channel |
| File size | GT 262,144 B · LR 65,536 B |
| Integrity | 0 NaN, 0 Inf, 0 shape violations |

**Content:** scanning-electron-microscope micrographs of material surfaces — porous networks,
particulate-scattered substrates, fibrous mats, filament structures, fine-grained textures.
This is a complete domain change from round 1, which contained natural photographs.

---

## 3. Intensity statistics

| Statistic | GT | NoisyLR |
|---|---|---|
| Global min | 0.00000 | −0.28342 |
| Global max | 1.00000 | 2.09858 |
| Mean of per-image means | 0.45479 | 0.45477 |
| Mean of per-image stds | 0.16603 | 0.18916 |

* Ground truth is confined to [0,1] exactly as the specification states.
* **Noise adds spread but no brightness shift** — the two means agree to five decimal places.
* Per-image GT means span 0.084 → 0.966; CV = 0.258. Content brightness varies widely, which
  is genuine diversity rather than a calibration issue. **Keep `scale_constant = 1.0`.**

### Out-of-range behaviour

| | Value |
|---|---|
| Pixels above 1.0 | mean 1.838%, median 0.690%, max 41.10% |
| Pixels below 0.0 | mean 0.088%, max 13.07% |
| Largest excursion above 1.0 | +1.0986 |
| Images with zero overshoot | 15 / 1,325 |
| **corr(image brightness, overshoot fraction)** | **0.678** |

That correlation of 0.678 is the signature of **multiplicative noise**: bright pixels receive
proportionally larger perturbations and are the ones pushed past the ceiling, while dark pixels
barely move. Purely additive noise would show no such relationship.

**Implication:** never clip the input. The out-of-range values encode the noise process.
Clip only the output, to [0,1].

---

## 4. Degradation operator

### 4.1 Downsampling kernel — not identifiable

| Kernel | Low-pass MSE |
|---|---|
| area | 1.8016e-04 |
| bicubic_aa | 1.8189e-04 |
| lanczos | 1.8196e-04 |
| bilinear_aa | 1.8296e-04 |
| nearest | 2.7286e-04 |

The four antialiased candidates span **1.55%**, far below the measurement noise floor. `area`
ranks first but by a margin smaller than the uncertainty, so that ranking carries no
information. `nearest` is clearly excluded at a 50% margin.

**Decision: randomise across the four antialiased kernels; exclude `nearest`.** Committing to a
single kernel would mean 100% of synthetic data is systematically biased if the guess is wrong;
randomising guarantees the true kernel appears in the mix and produces a model robust to the
uncertainty. This matches the round-1 conclusion.

### 4.2 Alignment

Median sub-pixel shift **(0.0, 0.0)** — no `align_corners`-style half-pixel mismatch. One risk
ruled out.

### 4.3 Order of operations

Lag-1 spatial autocorrelation of the noise residual: **−0.050 horizontal, −0.052 vertical**.
Essentially zero, meaning noise was applied **after** downsampling (noise applied before would
be smoothed by the resampling and show positive correlation). Same conclusion as round 1.

---

## 5. Noise model

### 5.1 Variance fit

```
var(r | mu) = 2.5071e-02 * mu^2  +  9.5974e-03 * mu  +  0.0
```

| Component | Contribution at μ=0.5 | Share |
|---|---|---|
| Multiplicative (speckle) | 6.268e-03 | 56.6% |
| Signal-proportional (shot-like) | 4.799e-03 | 43.4% |
| Additive (constant) | 0.0 | 0.0% |

σ_mult = **0.1583** (round 1: 0.1431 — about 11% stronger)
Residual std = **0.1081** (round 1: ≈0.083 — about 30% noisier overall)

The additive term fits to exactly zero, so the Gaussian component mentioned in the
specification is either small relative to the speckle or absorbed into the linear term.

### 5.2 Distribution shape — heavy-tailed

| Statistic | Measured | Gaussian expectation |
|---|---|---|
| Skewness | +0.5152 | 0 |
| Excess kurtosis | **+2.0798** | 0 |
| \|r\| > 3σ | **1.1412%** | 0.270% (4.2×) |
| \|r\| > 5σ | **0.05295%** | 0.00006% (≈880×) |

The Q-Q plot tracks the Gaussian line through the centre and departs sharply upward beyond
+1.5σ. The noise is right-skewed with substantially heavier tails than Gaussian.

### 5.3 Independent cross-check

Predicting the fraction of pixels exceeding 1.0 from the fitted variance, evaluated per pixel:

| | Value |
|---|---|
| Predicted (Gaussian) | 1.746% |
| Measured | 1.932% |
| Ratio | **1.11×** |

The two measurements are consistent rather than contradictory: **the Gaussian approximation is
adequate within roughly 2σ and light in the far tails.** Overshoot past 1.0 is dominated by
bright pixels where the threshold is only ~1.4σ away, a region where the distributions agree.
The kurtosis and 3σ/5σ counts probe further out, where they diverge.

**Practical consequence:** a Gaussian generator reproduces the bulk of the noise correctly but
produces roughly a quarter of the real ≥3σ outliers — approximately 190 missing extreme pixels
per 128×128 image. Those are exactly the pixels that survive restoration as visible speckle
artefacts.

---

## 6. Texture and frequency content

| Measurement | Value |
|---|---|
| LR 64px crop gradient energy | p10 0.173 · p40 0.246 · median 0.263 · p90 0.361 |
| GT 128px crop gradient energy | median 0.133 |
| Round-1 threshold (0.1828) percentile here | **12.5th** |

Two findings:

1. **The round-1 crop threshold is effectively inactive** — it rejects only 12.5% of crops.
2. **More importantly, gradient is being measured on the wrong image.** LR crops read a median
   of 0.263 against GT's 0.133: noise roughly *doubles* apparent gradient energy. Because
   `sample_crop` measures on the noisy LR patch, crop selection is driven by noise realisation
   rather than structural content, which degenerates into uniform random sampling. In round 1
   this mattered less because natural photographs had genuinely blank regions; here everything
   is textured.

**Radial power spectrum:** GT power decays steeply with frequency. NoisyLR tracks it to roughly
**0.4 normalised frequency**, then flattens and sits *above* GT. That plateau is the white-noise
floor — independently confirming noise was added after downsampling. Above 0.4 the input
contains essentially no real signal, so everything the model produces in that band is
reconstruction rather than recovery.

**2D FFT:** mildly anisotropic, not strongly oriented. D4 augmentation remains valid.

---

## 7. Difficulty and content diversity

### Bicubic baseline (the number to beat)

| Metric | Round 2 | Round 1 |
|---|---|---|
| PSNR | **20.47 dB** | 22.70 dB |
| SSIM | **0.5079** | 0.553 |
| SSIM (edge) | 0.5964 | 0.681 |
| SSIM (flat) | 0.4784 | 0.511 |
| LPIPS | **0.4839** | 0.427 |

Round 2 is 2.2 dB harder, consistent with the 30% stronger noise. As in round 1, bicubic's edge
SSIM exceeds its flat SSIM, because bicubic passes noise straight through and flat regions are
noise-dominated. **A well-behaved model should invert that relationship; if it does not, it is
over-smoothing.**

### Content clusters (600 GT images, k=6)

| Cluster | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Count | 47 | 183 | 88 | 127 | 71 | 84 |

Reasonably balanced with no dominant cluster, and visual inspection confirms the clusters
correspond to genuinely different SEM sample types. **A held-out-cluster OOD split is therefore
meaningful on this data.** Cluster 2 (88) or 5 (84) are good candidates — large enough for a
stable metric, small enough not to cost significant training data.

---

## 8. Round 1 → Round 2 comparison

| | Round 1 | Round 2 | Carries over? |
|---|---|---|---|
| Format | .npy float32, 256←128 | identical | **Yes, unchanged** |
| Content domain | Natural photographs | SEM micrographs | **No — retrain required** |
| Noise family | Multiplicative speckle | Multiplicative speckle | Yes |
| σ_mult | 0.1431 | 0.1583 | Refit constant |
| Residual std | ≈0.083 | 0.1081 | ~30% noisier |
| Tail behaviour | not measured | kurtosis 2.08 | New finding |
| Kernel | not identifiable | not identifiable | Yes |
| Noise order | after downsample | after downsample | Yes |
| Alignment | clean | clean | Yes |
| Resolution coverage | all 256←128 | all 256←128 | **512 gap persists** |

---

## 9. Required codebase changes

### P0 — correctness

**1. `src/dataset.py` · `RestorationDataset._crop` — gradient measured on the wrong image**

```python
# current (selects on noise, not structure)
e = _grad_energy(lr[y:y + p, x:x + p])

# fixed
e = _grad_energy(gt[y*s:(y+p)*s, x*s:(x+p)*s])
```

**2. `src/dataset.py` · `estimate_grad_threshold`** — same defect; measure on GT and re-derive
the threshold from the GT distribution (p40 ≈ 0.11–0.13, to be confirmed).

**3. `artifacts/stats.json`** — update fitted constants:

```json
{
  "noise_var_fit": {"a_mult": 2.5071e-02, "b_poisson": 9.5974e-03, "c_additive": 0.0},
  "meas_mult": 0.1583,
  "meas_add": 0.0,
  "kernels": ["area", "bicubic_aa", "lanczos", "bilinear_aa"],
  "kernel_p": [0.3, 0.3, 0.2, 0.2],
  "order_mix": [0.1, 0.9]
}
```

`order_mix` shifts toward `noise_last`: the autocorrelation evidence for noise-after-downsampling
is clear, but a small fraction of the alternative is retained for robustness.

### P1 — required to use 72% of the data

**4. GT-only synthetic training pool.** `RestorationDataset` currently requires paired entries
from the cache index. It needs a mode that draws from the 3,459 GT-only images and produces
pairs on the fly via `degrade()`. This is the single highest-value change in the list — without
it, most of the dataset is unusable.

Suggested shape:

```python
RestorationDataset(
    cache_dir,
    paired_stems=...,        # 1,325 real pairs
    gt_only_stems=...,       # 3,459 synthesis-only images
    real_frac=0.3,           # proportion of each batch drawn from real pairs
    degrade_cfg=degrade_cfg_from_stats(),
    jitter_range=(0.7, 1.4),
)
```

**5. `src/cache.py`** — extend to cache GT-only images (currently assumes pairs).

**6. `src/degrade.py`** — add empirical residual resampling to reproduce the heavy tails:

```python
build_residual_bank(pairs, downsample_fn, kernel, a, b, c)   # from the 1,325 real pairs
add_noise_empirical(img, banks, a, b, c, rng, scale=1.0)     # replaces add_noise_varfit
```

This resamples normalised residuals from the real data binned by intensity, reproducing
kurtosis, skew and tail shape exactly without assuming a parametric family. Given that most
training pairs will be synthetic, this is no longer optional.

### P2 — validation and reporting

**7. `src/splits.py`** — rebuild the OOD split on round-2 content clusters. Hold out one full
cluster of **real pairs only** for validation; GT-only images all go to training. Real pairs are
the only ground truth about the actual degradation and should be spent carefully.

**8. Baseline references** — every round-2 result must be quoted against
**20.47 dB / 0.5079 SSIM / 0.4839 LPIPS**, not round-1 numbers.

**9. Retrain from scratch.** The round-1 checkpoint learned to restore natural photographs. A
fine-tuning experiment from it is worth one run as a comparison, but the domain gap is large.

### Unchanged and still required

* **Scale jitter (0.7–1.4×)** — no 512×512 anywhere in training data, so the resolution gap
  flagged in round 1 is entirely unaddressed and this remains the only mitigation.
* **Edge-weighted Charbonnier + SSIM + LPIPS loss** — the case is stronger here, since SEM
  images are wall-to-wall fine texture and over-smoothing destroys nearly everything.
* **Format contract, normalisation, inference harness** — verified working on round-2 data.

---

## 10. Recommended training strategy

1. Fit the degradation model on the 1,325 real pairs (done — constants above).
2. Build the residual bank from the same pairs to capture tail behaviour.
3. Hold out one content cluster of real pairs (~85–130 images) as `val_ood`. Never train on it.
4. Train on: remaining real pairs + synthetic pairs generated from all 3,459 GT-only images
   plus the training-split GT images, mixed at roughly 30% real / 70% synthetic.
5. Apply scale jitter before degradation, D4 and CutBlur after.
6. Monitor `ssim_edge` alongside overall SSIM. Edge SSIM falling below overall SSIM indicates
   over-smoothing.
7. Report against the round-2 bicubic baseline.

---

## 11. Open risks

* **No 512×512 examples.** Evaluation may include them; scale jitter is the only mitigation and
  it is untested at that scale.
* **LPIPS backbone domain mismatch.** LPIPS uses AlexNet/VGG features trained on natural
  photographs, so its perceptual judgements on SEM texture rest on weaker foundations. KLA
  scores it regardless, so optimise it, but treat it cautiously as a quality signal.
* **Synthetic-real gap.** With ~70% of training data synthetic, any systematic mismatch between
  `degrade()` and the real degradation propagates through most of training. The verification
  check (synthetic vs real residual statistics within a few percent) should be re-run after
  every change to the generator.
* **One corrupt LR file** from the download truncation, excluded. File counts otherwise match
  the intended dataset.
