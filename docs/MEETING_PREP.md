# Meeting prep — industry guidance session

Everything below is grounded in measurements you actually made. Nothing here is
padding; if you are asked "how do you know that", there is a number behind it.

---

## 1. The 60-second version

> "We're restoring SEM inspection images that have been downsampled 2× and hit
> with speckle and Gaussian noise. Rather than jumping straight to architecture,
> we spent the first phase characterising the degradation itself — we recovered
> the downsampling kernel and fitted the noise model from the paired data. That
> turned out to matter more than anything we did to the network. We're currently
> about 3 dB PSNR and 14% SSIM over a bicubic baseline, at 12 ms per image."

Then stop. Let them ask.

---

## 2. Numbers to have on the tip of your tongue

| | bicubic | our model |
|---|---|---|
| PSNR | 20.94 dB | **23.93 dB** (+3.0) |
| SSIM | 0.4441 | **0.5064** (+14%) |
| LPIPS | 0.5248 | **0.3577** (−32%) |
| inference | — | **12.2 ms/image** (T4) |

Model: NAFNet U-Net, 0.98M–2.1M parameters depending on variant.
Data: 4,785 paired 256←128 SEM images, validated on a held-out content cluster
of 1,165 images the model never saw.

---

## 3. The four findings worth leading with

These are what will make you sound like you understand the problem rather than
having run a tutorial.

### 3.1 We identified the downsampling kernel that a standard test said was unidentifiable

Round 1 we compared candidate kernels by low-pass filtering both images and
measuring MSE. Everything scored within 1.55% and we concluded it couldn't be
determined.

That conclusion was **an artefact of the metric**. Antialiased and
non-antialiased kernels differ almost entirely in the high-frequency band — and
low-pass filtering deliberately discards exactly that band. The test could not
have succeeded.

Switching to spectral distance plus lag-1 autocorrelation plus local-variance
matching, the kernel resolved cleanly: **Gaussian blur σ≈0.6 followed by
decimation** — partial antialiasing. Two independent statistics converged on the
same σ.

**The transferable point:** a negative result is only meaningful if your
measurement was capable of detecting the positive. Always ask what your metric is
blind to.

### 3.2 The noise is heavy-tailed, and that broke our first synthetic generator

Fitting variance against signal intensity gave
`var = 0.0238·μ² + 0.0104·μ + 0.0031` — multiplicative speckle dominant, with a
real additive term.

But the *shape* mattered more than the variance. Excess kurtosis measured **+3.9**
against a Gaussian's 0. Pixels beyond 5σ occurred at **880× the Gaussian rate**.
Our first generator matched the variance exactly and produced roughly a sixth of
the real extreme outliers.

Fix: we stopped guessing a distribution. We collected 2.6 million real noise
residuals from the paired data, normalised by predicted σ and binned by
intensity, and resample from those. Reproduces skew and kurtosis exactly.

**Cross-check worth mentioning:** when we fitted the noise against the *wrong*
kernel, the additive term pinned to exactly zero — physically implausible, since
the spec says additive Gaussian noise is present. With the correct kernel it came
out positive. A parameter we did not fit to agreeing with documented physics is
much stronger evidence than a metric improving.

### 3.3 Carefully validated synthetic data still hurt

Because the first data drop had 4,785 clean images but only 1,325 pairs, we built
a synthetic degradation pipeline. We validated it hard: autocorrelation within
1.2% of real, local texture energy within 0.1%, noise magnitude within 0.4%.

Training on 70% synthetic gave PSNR **1.08 dB below bicubic**. Retraining on real
pairs only gave **3.0 dB above**. A 4 dB swing.

**The lesson:** matching second-order statistics is a much weaker guarantee than
it looks. Something in the synthetic distribution differed in a way none of our
diagnostics caught, and pixel-exact reconstruction is unforgiving of it.

### 3.4 The ground truth is itself noisy, which bounds what anyone can achieve

We noticed the model was smoothing away fine texture — it retained only 28% of
the ground truth's high-frequency energy. Chasing that with a spectral loss got
it to 72%, but the images looked no better and every metric got worse.

Then we measured the GT's own power spectrum. Above roughly 60% of maximum
frequency it is **flat** (tail flatness std/mean = 0.024) — the signature of a
white-noise floor. So most of the "detail" we were trying to recover is
acquisition noise in the reference images, which is unpredictable by definition.

This explains the plateau: capacity, data volume and loss modifications all
converged to similar performance because a fixed fraction of the residual error
is irreducible.

**This is consistent with how the data was made.** The acknowledgements confirm
KLA extracted patches from the NFFA-EUROPE SEM dataset and added synthetic noise
— so the "ground truth" is a real SEM capture carrying its own acquisition noise,
not a noise-free reference.

---

## 4. Honest limitations — say these before they find them

Being upfront about weaknesses reads as competence, not weakness.

- **Our synthetic pipeline is currently unused.** We built it, validated it,
  measured that it hurt, and turned it off. We can defend that with the 4 dB
  number, but it means we're not exploiting the "synthetic data generation"
  route the problem statement explicitly encourages.
- **We have never tested a 512×512 forward pass.** Every training pair is
  256←128, but the spec says evaluation may include 512 ground truths. Scale
  jitter is our only mitigation and it's untested at that size. This is our
  biggest known risk.
- **Capacity is not our bottleneck and we're not certain what is.** Doubling the
  parameters changed nothing, which is why we're currently testing receptive
  field instead — the architecture has only one downsampling level where NAFNet
  normally uses three or four.
- **LPIPS may be a poor proxy on this data.** Its VGG features were trained on
  natural photographs. In one experiment it penalised the model with visibly more
  realistic texture.

---

## 5. Questions to ask them

Good questions demonstrate more than good answers. These are genuine open
problems in your project.

**On the perception–distortion trade-off**
> "There's a provable trade-off between pixel accuracy and perceptual realism.
> For inspection specifically — is a slightly blurrier but pixel-faithful image
> better than a sharper one with plausible-but-invented texture? In our domain
> hallucinated texture could read as a defect."

This is the best question you have. It's technically substantive and it's a real
decision you face.

**On validating against noisy references**
> "We measured that the ground truth images carry their own acquisition noise
> floor, so a fraction of our residual error is irreducible. How do you handle
> validation when the reference isn't clean?"

**On where restoration sits in the pipeline**
> "Is restoration typically run as a preprocessing step before defect detection,
> or are they trained jointly? We've been optimising SSIM and LPIPS, but if the
> real objective is downstream detection accuracy, those may be the wrong
> targets."

**On latency in practice**
> "We're at 12 ms per image. What's the actual budget in a production inspection
> line — is that comfortable, or an order of magnitude too slow?"

**On the degradation model**
> "We reverse-engineered the downsampling kernel and noise statistics from the
> paired data. In a real deployment, would you expect to characterise the imaging
> system's degradation directly from the instrument instead?"

---

## 6. Traps to avoid

**Don't overclaim the synthetic pipeline.** It's built and it doesn't help right
now. Say that plainly.

**Don't quote round-1 numbers alongside round-2 numbers.** Different dataset,
different validation split, not comparable. If you mention round 1 at all, say
so explicitly.

**Don't say "the model learns to remove noise".** Say what it actually does:
predicts the conditional mean of plausible clean images given the input, which is
why it blurs when the input is ambiguous. That one sentence will do more for your
credibility than any metric.

**If you don't know, say so and say what you'd measure.** "I don't know — I'd
test it by..." is a strong answer. Bluffing to a fab engineer is not.

---

## 7. One-line answers to likely questions

**"Why NAFNet?"**
> Transformer-class restoration quality from plain convolutions — it replaces
> activation functions with a gated multiplication. Inference speed is part of
> our score, so a cheap architecture mattered.

**"Why not a GAN or diffusion model?"**
> Both would produce sharper texture, but by generating plausible detail rather
> than recovering real detail. For defect inspection, invented structure is a
> failure mode, not a feature. We'd want to understand the tolerance for that
> before going there.

**"How do you know you're not overfitting?"**
> We hold out an entire visual content cluster — 1,165 images the model never
> sees, chosen to be a different category of surface. We compare gains over
> bicubic on that split rather than raw scores, since the held-out cluster is
> intrinsically harder content.

**"What would you do with another month?"**
> Test whether receptive field is the bottleneck, which we're running now.
> Investigate why validated synthetic data hurt, because that's the most
> surprising result we have. And measure against downstream defect detection
> rather than SSIM, if that's the real objective.

---

## 8. Bring with you

- The four-panel comparison figure: noisy input / bicubic / our model / ground
  truth, on both a best and a worst case
- The GT power-spectrum plot showing the noise floor — it's your most
  sophisticated single finding
- The kernel sweep table (σ 0.4 through 0.8 against the two convergent metrics)
- `docs/RESULTS.md` on a laptop, in case anyone wants specifics
