# Study Guide — Image Restoration for Semiconductor Inspection

A ground-up explanation of everything in this project: the statistics behind the
EDA, the sampling theory behind the kernels, the architecture layer by layer,
and the losses. Written to be read in order.

---

# Part 1 — The statistics of noise

## 1.1 What a distribution is, and its four moments

Take one pixel in a clean image with true value **μ = 0.5**. Photograph it a
thousand times and you get a thousand slightly different measurements. Plot how
often each value occurs and you have a **distribution**. Everything we did in
the EDA was an attempt to describe that shape with a handful of numbers.

Four numbers, called **moments**, describe most of what matters:

```
  1st moment   MEAN        where is the centre?
  2nd moment   VARIANCE    how wide is it?
  3rd moment   SKEWNESS    is it lopsided?
  4th moment   KURTOSIS    how heavy are the tails?
```

The mean and variance are familiar. The other two are the ones that mattered
here, so let us be precise about them.

## 1.2 Skewness — is the distribution lopsided?

```
   SKEW = 0 (symmetric)          SKEW > 0 (right-skewed)
        ▁▃▅█▅▃▁                       ▁▅█▅▃▂▁▁▁▁
       ───────────                   ─────────────────
   equal weight either side      long tail stretching RIGHT
```

Formally, skewness is the average of `((x − μ)/σ)³`. Cubing preserves sign, so
values far above the mean contribute large positive numbers and values far below
contribute large negative ones. If they balance, skew is zero.

**Your measurement: +0.55.** The noise pushes pixels *up* more often and more
violently than it pushes them down. That is the fingerprint of **multiplicative**
noise: a bright pixel at 0.9 can be scaled up by 20% and gain 0.18, while a dark
pixel at 0.1 scaled by the same 20% gains only 0.02. Bright pixels get bigger
kicks, and since your images have more room to move up than an absolute floor at
zero allows moving down, the tail leans right.

## 1.3 Kurtosis — how heavy are the tails?

This is the one worth understanding properly, because it drove a real decision.

Kurtosis is the average of `((x − μ)/σ)⁴`. The fourth power means values near
the mean contribute almost nothing (0.5⁴ = 0.06) while values far out dominate
(4⁴ = 256). So kurtosis is almost entirely a statement about **rare extreme
events**.

By convention we report **excess kurtosis** = kurtosis − 3, so that a Gaussian
scores exactly 0.

```
  EXCESS KURTOSIS = 0            EXCESS KURTOSIS > 0
  (Gaussian)                     (heavy-tailed)

       ▁▃▅███▅▃▁                      ▁▂▅████▅▂▁
     ▁▁         ▁▁                  ▁▁          ▁▁
   ──────────────────            ▁▁▁──────────────▁▁▁
   tails die off fast            sharper peak, FATTER tails
                                 more mass in the extremes
```

Two distributions can have **identical mean and identical variance** and still
behave completely differently in the tails. That is the entire point.

**Your measurement: +2.08.** Concretely:

| event | Gaussian predicts | you measured | ratio |
|---|---|---|---|
| pixel beyond 3σ | 0.270% | 1.141% | 4.2× |
| pixel beyond 5σ | 0.00006% | 0.053% | ~880× |

A 5σ event should happen once in about 1.7 million pixels. In your data it
happens once in every 1,900. On a 128×128 image that is roughly **9 extreme
pixels per image that a Gaussian model says should essentially never occur.**

### Why this changed what we built

Our first noise generator drew Gaussian noise scaled to match the measured
variance. It got the mean right and the variance right — and produced roughly a
quarter of the real 3σ outliers and a sixth of the 5σ ones. Since 70% of your
training data is synthetic, the model would have trained on a systematically
easier problem than the test set, and would never have learned to handle the
violent isolated pixels that survive restoration as visible artefacts.

The fix avoided guessing a distribution entirely. We collected 2.3 million real
noise samples from the paired data, normalised each by its predicted σ, sorted
them into 20 bins by pixel brightness, and now **resample from those** instead of
generating Gaussian values. The result reproduces skew and kurtosis exactly,
because it is literally reusing the real thing.

## 1.4 The variance model

Noise is not the same everywhere in an image. We modelled how it grows with
brightness:

```
   var(noise | brightness μ)  =  a·μ²  +  b·μ  +  c
                                  ▲       ▲      ▲
                                  │       │      └── additive: constant everywhere
                                  │       └───────── shot: grows with brightness
                                  └───────────────── multiplicative: grows with μ²
```

Each term has a physical meaning.

**a·μ² — multiplicative (speckle).** Noise proportional to signal. If a pixel's
true value doubles, its noise standard deviation doubles too. Written
`measured = true × (1 + ε)`, so the error is `true × ε` and its variance is
`true² × var(ε)`, giving the μ² dependence. This is what coherent imaging
produces: waves interfere constructively and destructively, and the interference
scales with the signal.

**b·μ — shot noise.** Photon counting. If you expect N photons you actually get
roughly N ± √N, so variance ≈ N — linear in brightness.

**c — additive (read noise).** Electronics, thermal noise. Same amount
everywhere regardless of signal. This is the "additive Gaussian" in KLA's spec.

**Your fitted values (against the correct kernel):**

```
   a = 0.01334      multiplicative
   b = 0.02109      shot
   c = 0.000324     additive       →  σ_add ≈ 0.018
```

At mid-brightness (μ = 0.5) this gives σ ≈ 0.119.

### The detail that validated the kernel

When we fitted this against the *wrong* downsampling kernel, `c` came out as
**exactly zero** — pinned to the boundary by the non-negative least squares fit.
That was suspicious, because KLA's specification explicitly says additive
Gaussian noise is present. After identifying the correct kernel, `c` became
positive and physically sensible. A better clean reference produced a
decomposition consistent with the documented physics. That is strong independent
evidence, and it is the kind of cross-check worth looking for: not "my number
improved" but "my number now agrees with something I did not fit to."

## 1.5 Autocorrelation — spatial structure

Everything above treats pixels one at a time. Autocorrelation asks whether
*neighbouring* pixels are related.

```
   Take the image.  Shift a copy right by k pixels.  Correlate the two.

   original:  ▓▓░░▓▓░░▓▓░░
   shift 1:    ▓▓░░▓▓░░▓▓░░
              └── high overlap → correlation near 1

   pure noise (no structure):
   original:  ▓░▓▓░▓░░▓░▓░
   shift 1:    ▓░▓▓░▓░░▓░▓░
              └── no relationship → correlation near 0
```

**Two uses in this project.**

First, to determine *when* the noise was added. Noise applied **before**
downsampling gets averaged by the resampling, which smears each noise sample
across neighbours and creates positive correlation. Noise applied **after**
stays independent. You measured −0.05, essentially zero, so noise came last.

Second, to identify the kernel. Comparing autocorrelation of real versus
synthetic images told us our synthetic images were too smooth — the single most
useful diagnostic we ran.

## 1.6 Power spectrum — the same question in frequency

The Fourier transform re-expresses an image as a sum of wave patterns at
different spatial frequencies. Low frequency = broad gradients. High frequency =
fine detail and noise.

```
   power
     │╲
     │ ╲___              real structure decays with frequency
     │     ╲__
     │        ╲_____
     │              ╲________  ← real image
     │              ┌─────────  ← white noise sits FLAT
     └──────────────────────────► frequency
       low                  high
```

White noise has equal power at every frequency — that is the definition of
"white", by analogy with white light. So the **flat plateau at the high end of a
noisy image's spectrum is the noise floor**, and where the curve stops decaying
and goes flat tells you where noise starts dominating signal.

**In your data that crossover is around 0.4 of maximum frequency.** Above that,
the degraded input contains essentially no real information. Everything the model
produces up there is reconstruction from learned priors, not recovery of
something present in the input. That is a good concrete line for your slides.

## 1.7 Q-Q plot — reading tails visually

A quantile-quantile plot sorts your data and plots it against what a Gaussian
would produce at the same ranks.

```
   observed │                  ╱ ← curves UP: heavier right tail
     value  │                ╱
            │            ╱╱╱
            │        ╱╱╱      ← straight middle = Gaussian-like core
            │    ╱╱╱
            │╱╱╱
            └──────────────────► theoretical Gaussian quantile
```

Perfectly Gaussian data lies on a straight line. Yours tracked the line through
the middle and bent sharply upward past about +1.5σ. That is exactly what excess
kurtosis 2.08 with skew +0.55 looks like: an ordinary core with an
extraordinary right tail.

---

# Part 2 — Sampling, aliasing, and kernels

## 2.1 What downsampling actually is

Going 256×256 → 128×128 means producing one output pixel for every four input
pixels. The **kernel** is the rule for combining them.

```
   input 4×4                    output 2×2

   ┌──┬──┬──┬──┐
   │ a│ b│ c│ d│                ┌────┬────┐
   ├──┼──┼──┼──┤                │ w  │ x  │
   │ e│ f│ g│ h│      ───►      ├────┼────┤
   ├──┼──┼──┼──┤                │ y  │ z  │
   │ i│ j│ k│ l│                └────┴────┘
   ├──┼──┼──┼──┤
   │ m│ n│ o│ p│
   └──┴──┴──┴──┘

   nearest / decimate :  w = a                    (throw three away)
   area / box         :  w = (a+b+e+f)/4          (average the block)
   bilinear           :  weighted average, wider window
   bicubic            :  wider still, with NEGATIVE weights at the edges
```

Bicubic's negative weights are a mild sharpening. They are also why bicubic
output can land slightly outside [0,1] — you measured 0.03% of pixels doing
exactly that.

## 2.2 Aliasing — why the kernel matters at all

This is the central concept, and it is worth getting right.

**Nyquist's theorem:** to represent a wave you need at least two samples per
cycle. Halve your sampling rate and any detail finer than the new limit cannot be
represented. But it does not politely vanish — it **masquerades as a lower
frequency**.

```
   true signal (high frequency):   ∿∿∿∿∿∿∿∿∿∿∿∿
   sample every 2nd point:         •   •   •   •
   what you reconstruct:           ╲___╱‾‾‾╲___╱     ← a DIFFERENT, slower wave
                                   this is ALIASING
```

The classic example is a wagon wheel in a film appearing to spin backwards. The
wheel's true rotation is faster than the frame rate, so it aliases into an
apparent slow reverse rotation.

**Antialiasing** means blurring before you subsample, deliberately destroying the
detail that cannot be represented, so it cannot fold down and corrupt lower
frequencies.

```
   WITH antialiasing              WITHOUT antialiasing
   blur, then subsample           subsample directly

   fine detail is REMOVED         fine detail FOLDS DOWN into
   result is smooth, clean        the image as false structure
                                  result looks crisper but contains
                                  energy that does not belong there
```

Neither is "correct". They are different choices, and they produce
measurably different images.

## 2.3 Why our first kernel test failed

Round 1 compared candidate kernels by low-pass filtering both images and
measuring squared error. Every candidate scored within 1.55% of the others and we
concluded the kernel was unidentifiable.

That conclusion was **an artefact of the metric, not a fact about the data.**

```
   kernels differ HERE ────────────────────┐
                                           ▼
   power │╲                          ░░░░░░░░░░
         │ ╲___                      ░░░░░░░░░░
         │     ╲____                 ░░░░░░░░░░
         │          ╲______________  ░░░░░░░░░░
         └────────────────────────────────────► frequency
          └──── what low-pass KEEPS ──┘└ what it DISCARDS ┘
```

Antialiased and non-antialiased kernels are nearly identical at low frequency and
differ almost entirely at high frequency. By low-pass filtering first, we threw
away the only band containing the answer, then measured what was left and found
no difference. The test could not have succeeded.

**The lesson generalises:** a negative result is only meaningful if your
measurement was capable of detecting the positive. Always ask what your metric is
blind to.

## 2.4 How we actually found it

We switched to three metrics that are sensitive to high-frequency content, and
swept a family of kernels that interpolates between the extremes:

`gauss:σ` = blur with Gaussian of width σ, then decimate. σ = 0 is pure
decimation with maximum aliasing; large σ approaches box averaging.

```
  kernel      autocorr err   local-var ratio   verdict
  decimate      −0.0864          1.159         too sharp  (too much aliasing)
  gauss:0.4     −0.0594          1.116         still sharp
  gauss:0.5     −0.0182          1.045         close
  gauss:0.6     +0.0113          1.004         ✓ BEST
  gauss:0.7     +0.0278          0.971         slightly smooth
  area          +0.0500          0.947         too smooth
```

Two **independent** statistics — pixel-to-pixel correlation and local texture
energy — both cross their target between σ = 0.5 and 0.6. Independence matters:
one metric agreeing with itself proves nothing, two unrelated metrics converging
on the same answer is real evidence.

**Conclusion: KLA used partial antialiasing, σ ≈ 0.6.** Once identified, the
right move changed from "randomise widely because we cannot tell" to "narrow
jitter around the estimate for robustness" — `[gauss:0.5, gauss:0.6, gauss:0.7]`
weighted `[0.25, 0.5, 0.25]`.

---

# Part 3 — The architecture, layer by layer

## 3.1 The shape of the problem

```
   INPUT                                          OUTPUT
   128×128 noisy          ┌─────────┐             256×256 clean
   ───────────────────►   │  MODEL  │   ─────────────────────►
   1 channel              └─────────┘             1 channel
```

Two jobs at once: remove noise, and invent detail that the downsampling
destroyed. Doing them in one network rather than sequentially matters, because a
separate upscaler would amplify whatever noise the denoiser left behind and
reconstruct it as if it were real detail.

## 3.2 Full dataflow of your model

```
  x (1×128×128)
   │
   ├──────────────────────────────────────────┐
   │                                          │  GLOBAL RESIDUAL
   ▼                                          │  bilinear ×2
  intro: Conv3×3, 1→64                        │
   │                                          │
   ▼                                          │
  enc1: NAFBlock(64) × 2  ──────┐             │
   │                            │ SKIP        │
   ▼                            │             │
  down: Conv3×3 stride2, 64→128 │             │
   │                            │             │
   ▼                            │             │
  enc2: NAFBlock(128) × 2       │             │
   │                            │             │
   ▼                            │             │
  middle: NAFBlock(128) × 2     │  ← bottleneck, 64×64 resolution
   │                            │             │
   ▼                            │             │
  up: Conv3×3 128→256           │             │
      + PixelShuffle(2)         │             │
   │  (back to 128×128, 64ch)   │             │
   ▼                            │             │
  concat ◄──────────────────────┘             │
   │  (64 + 64 = 128 channels)                │
   ▼                                          │
  reduce: Conv1×1, 128→64                     │
   │                                          │
   ▼                                          │
  dec1: NAFBlock(64) × 2                      │
   │                                          │
   ▼                                          │
  upsample: Conv3×3 64→4                      │
            + PixelShuffle(2)   ← 1×256×256   │
   │                                          │
   ▼                                          │
   + ◄────────────────────────────────────────┘
   │
   ▼
  output (1×256×256)
```

Total: **0.98M parameters**. Small by modern standards — deliberately, because
inference throughput is part of your score.

## 3.3 The global residual — the single most important design choice

```
  output = network(x) + bilinear_upsample(x)
```

The network never has to produce the image. Bilinear upsampling already gives a
blurry but structurally correct version, and the network only predicts the
**difference** — the sharpening and the noise removal.

Why this matters so much: at initialisation, a network's output is essentially
random. Without the residual it would start from noise and have to discover
"output should resemble input" from scratch. With it, an untrained network
already emits a reasonable blurry upscale, and training begins from a sensible
starting point. Gradients also flow directly to early layers through the addition,
which is the same reason ResNet works.

## 3.4 LayerNorm2d

```python
mu    = x.mean(dim=1, keepdim=True)      # average ACROSS CHANNELS
sigma = x.var (dim=1, keepdim=True)
out   = (x - mu)/sqrt(sigma + eps) * weight + bias
```

For each pixel position independently, look across all 64 channels, and
normalise so they have mean 0 and variance 1. Then rescale by learned per-channel
`weight` and `bias`.

**Why not BatchNorm?** BatchNorm normalises across the batch, so a sample's
output depends on which other samples share its batch. It also behaves
differently at training and inference time. For restoration, where you often use
small batches and care about exact pixel values, that instability hurts. LayerNorm
treats every sample independently — same result at batch size 1 or 128.

## 3.5 SimpleGate — the idea that names the architecture

"NAFNet" stands for **Nonlinear Activation Free** network. It contains no ReLU,
no GELU, no sigmoid. Instead:

```python
def forward(self, x):
    x1, x2 = x.chunk(2, dim=1)   # split channels in half
    return x1 * x2                # multiply them together
```

```
   input: 128 channels
   ┌──────────────┬──────────────┐
   │  x1 (ch 0-63)│ x2 (ch 64-127)│
   └──────┬───────┴───────┬──────┘
          └───── × ───────┘
                 │
                 ▼
          output: 64 channels
```

**Why does multiplication provide non-linearity?** Because it is not a linear
operation. A stack of convolutions without any non-linearity collapses
mathematically into a single convolution, no matter how deep — the network could
learn nothing a one-layer network could not. Something non-linear must break that
collapse. ReLU does it by clipping; SimpleGate does it by multiplying two learned
quantities together.

**Why prefer it?** It is *gating*: `x1` acts as content and `x2` as a learned
mask deciding how much of `x1` passes through, computed from the data rather than
by a fixed rule like ReLU's "negatives become zero". It is also cheap — one
element-wise multiply — and it halves the channel count, so the following
convolution is smaller. The paper's finding was that this matches or beats
conventional activations at lower cost.

Note the cost: `chunk` halves the channels, which is why `conv1` expands to `2c`
first. The expansion exists to feed the gate.

## 3.6 Simplified Channel Attention

```python
self.pool = nn.AdaptiveAvgPool2d(1)         # H×W → 1×1
self.conv = nn.Conv2d(channels, channels, 1)
return x * self.conv(self.pool(x))
```

```
   x: 128 × 64 × 64
        │
        ├──────────────────────────────┐
        ▼                              │
   AdaptiveAvgPool2d(1)                │
   128 × 1 × 1   ← one number per      │
        │          channel: "how much  │
        ▼          is this feature      │
   Conv1×1 (128→128)   present overall?"│
   128 × 1 × 1                         │
        │                              │
        └────────────► × ◄─────────────┘
                       │
                       ▼
               rescaled 128 × 64 × 64
```

Pooling to 1×1 collapses all spatial information, leaving a summary of *what* is
in the image rather than *where*. The 1×1 convolution lets channels talk to each
other, and the result multiplies the original feature map, scaling each channel
up or down.

Concretely: if the pooled summary indicates heavy speckle, the network can
suppress channels that respond to noise and amplify channels that respond to
edges — a **global** decision applied to every pixel. Convolutions are local by
construction; this is one of the few places the network reasons about the image
as a whole.

## 3.7 The NAFBlock

Two sub-blocks, each residual:

```
  x ──────────────────────────────┐
  │                               │
  ▼                               │
 LayerNorm2d                      │
  │                               │
  ▼                               │
 Conv1×1  c → 2c                  │   expand
  │                               │
  ▼                               │
 Conv3×3 DEPTHWISE (groups=2c)    │   spatial mixing, cheap
  │                               │
  ▼                               │
 SimpleGate  2c → c               │   non-linearity
  │                               │
  ▼                               │
 ChannelAttention                 │   global reweighting
  │                               │
  ▼                               │
 Conv1×1  c → c                   │   project back
  │                               │
  ▼                               │
  × beta  ◄── learned scalar, init 0
  │                               │
  ▼                               │
  + ◄─────────────────────────────┘
  │
  │  ── second half ──
  ├───────────────────────────────┐
  ▼                               │
 LayerNorm2d                      │
  │                               │
  ▼                               │
 Conv1×1  c → 2c                  │
  │                               │
  ▼                               │
 SimpleGate  2c → c               │
  │                               │
  ▼                               │
 Conv1×1  c → c                   │
  │                               │
  ▼                               │
  × gamma ◄── learned scalar, init 0
  │                               │
  ▼                               │
  + ◄─────────────────────────────┘
  │
  ▼ output
```

### Depthwise convolution

A normal 3×3 convolution on 128 channels producing 128 channels needs
128 × 128 × 9 = 147,456 weights: every output channel looks at every input
channel. A **depthwise** convolution (`groups=channels`) gives each channel its
own 3×3 filter and no cross-channel mixing: 128 × 9 = 1,152 weights, **128×
fewer**.

The division of labour is deliberate: depthwise 3×3 handles *spatial* patterns,
1×1 convolutions handle *channel* mixing. Doing them separately costs far less
than doing both at once, which is what makes this architecture fast enough to
matter for your throughput score.

### beta and gamma initialised to zero

```python
self.beta  = nn.Parameter(torch.zeros((1, c, 1, 1)))
self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))
```

At step zero, `x + out*0 = x`. **Every NAFBlock starts as an identity function**,
and the whole network starts as a pure bilinear upscale via the global residual.
Training then gradually opens each block up as it finds something useful to
contribute. This makes deep stacks stable from the first step — a trick from
ReZero and used in many modern transformers.

## 3.8 PixelShuffle — how upsampling actually happens

This deserves careful attention because it is the operation that does
super-resolution.

**The problem.** To go from 128×128 to 256×256 you must create pixels. The
obvious approaches have flaws: interpolation invents nothing new, and transposed
convolution produces checkerboard artefacts from unevenly overlapping kernels.

**The idea.** Do not create pixels spatially. Create them in *channels*, where
convolution is cheap, then rearrange.

```
  Conv produces 4 channels at 2×2:        PixelShuffle(2) rearranges to 1×4×4:

   channel 0      channel 1
   ┌────┬────┐   ┌────┬────┐               ┌────┬────┬────┬────┐
   │ A0 │ B0 │   │ A1 │ B1 │               │ A0 │ A1 │ B0 │ B1 │
   ├────┼────┤   ├────┼────┤               ├────┼────┼────┼────┤
   │ C0 │ D0 │   │ C1 │ D1 │      ───►     │ A2 │ A3 │ B2 │ B3 │
   └────┴────┘   └────┴────┘               ├────┼────┼────┼────┤
                                           │ C0 │ C1 │ D0 │ D1 │
   channel 2      channel 3                ├────┼────┼────┼────┤
   ┌────┬────┐   ┌────┬────┐               │ C2 │ C3 │ D2 │ D3 │
   │ A2 │ B2 │   │ A3 │ B3 │               └────┴────┴────┴────┘
   ├────┼────┤   ├────┼────┤
   │ C2 │ D2 │   │ C3 │ D3 │               each input position A became a
   └────┴────┘   └────┴────┘               2×2 output block, filled from
                                           the 4 channels at that position
```

Formally: `(C×r², H, W) → (C, H×r, W×r)`. No parameters, no arithmetic — pure
reshaping. All the learning is in the convolution that produced those `r²`
channels.

**Why no checkerboard?** In transposed convolution, output pixels receive
contributions from different numbers of input positions, so some are
systematically brighter — a periodic artefact. With PixelShuffle every output
pixel comes from exactly one channel at one position. There is no overlap to be
uneven about.

**Your model uses it twice:**

```
  self.up = Conv3×3(128 → 256) + PixelShuffle(2)
      64×64×128  →  64×64×256  →  128×128×64
      (decoder: restoring resolution lost to the encoder's stride-2)

  self.upsample = Conv3×3(64 → 1×2²=4) + PixelShuffle(2)
      128×128×64  →  128×128×4  →  256×256×1
      (the super-resolution tail: the actual 2× upscale)
```

## 3.9 The U-Net skeleton and why the skip matters

```
   128×128, 64ch   enc1 ──────── SKIP ──────────┐  detail preserved here
        │ down (stride 2)                       │
   64×64, 128ch    enc2                         │
        │                                       │
   64×64, 128ch    middle   ← wide receptive    │
        │                     field, sees       │
        │ up (PixelShuffle)   context           │
   128×128, 64ch   ─────── concat ◄─────────────┘
                       reduce, dec1
```

Downsampling to 64×64 means each neuron sees a larger fraction of the image, so
the network can reason about context — is this a fibre, a pore, a flat substrate?
But downsampling destroys fine positional detail.

The skip connection solves that by carrying the full-resolution features from
`enc1` straight across to the decoder. The bottleneck contributes *what* is
there; the skip contributes *exactly where*. Concatenating gives the decoder
both, and `reduce` (a 1×1 convolution) merges 128 channels back to 64.

## 3.10 Parameter budget

```
  per NAFBlock(c) ≈ 7c² weights

  enc1     2 × NAFBlock(64)    ≈   57k
  down     Conv3×3 64→128      ≈   74k
  enc2     2 × NAFBlock(128)   ≈  229k
  middle   2 × NAFBlock(128)   ≈  229k
  up       Conv3×3 128→256     ≈  295k   ← single biggest layer
  reduce   Conv1×1 128→64      ≈    8k
  dec1     2 × NAFBlock(64)    ≈   57k
  intro + upsample tail        ≈    3k
                                 ──────
                                 ≈ 0.95M   (reported: 0.98M)
```

Worth noticing that the `up` convolution alone is 30% of the model. Producing
`r²` channels for PixelShuffle is where the parameters go.

---

# Part 4 — Measuring image quality

Before the losses, the metrics. You have been reading `psnr 19.09 ssim 0.4666
lpips 0.4507` after every epoch without knowing what any of it means. This
section fixes that.

## 4.0 The problem metrics solve

You have a restored image and the true clean image. How close are they? You need
**one number** so you can compare models automatically. Your eye is the real
judge, but you cannot eyeball 297 validation images after every epoch.

The catch is that "close" is ambiguous, and different definitions disagree. This
is why KLA uses three metrics rather than one, and why we do too.

## 4.0.1 MSE and PSNR — average pixel error

Start with the obvious approach: subtract the two images and see how big the
differences are.

```
   MSE = mean( (predicted − true)² )       "mean squared error"
```

For images in the range 0 to 1, MSE is a small number — 0.01 means the typical
pixel is off by about 0.1. Squaring makes it awkward to interpret and gives
huge weight to a few large errors, so people convert it to a log scale:

```
              ⎛   MAX²  ⎞                      for images in [0,1], MAX = 1
   PSNR = 10·log₁₀⎜ ─────── ⎟   decibels        so this is just 10·log₁₀(1/MSE)
              ⎝   MSE   ⎠
```

**PSNR stands for Peak Signal-to-Noise Ratio. Higher is better.** The decibel
scale is logarithmic, so the numbers are compressed:

| PSNR | MSE | typical pixel error | meaning |
|---|---|---|---|
| 10 dB | 0.1 | 0.32 (32%) | badly wrong |
| 20 dB | 0.01 | 0.10 (10%) | recognisable, clearly degraded |
| 25 dB | 0.0032 | 0.056 (5.6%) | decent |
| 30 dB | 0.001 | 0.032 (3.2%) | good |
| 40 dB | 0.0001 | 0.010 (1%) | near-perfect |

**The rule of thumb: every 6 dB halves the pixel error.** So going from 20 dB to
26 dB means your typical error dropped from 10% to 5%. This is why gains of
"only" 0.5 dB are taken seriously — the scale is compressed and small numbers
represent real improvement.

**Your numbers:** bicubic gets 20.47 dB, meaning a typical pixel is off by about
9.5%. Your model at epoch 3 was at 19.09 dB, about 11% error — slightly worse
than bicubic at that point.

**Why PSNR alone is not enough.** It treats every pixel independently and has no
concept of structure. Consider two corrupted versions of the same image, both
with identical MSE:

```
   A: every pixel shifted           B: half the image is perfect,
      up by 0.1 uniformly              half is scrambled noise

   ┌────────────┐                   ┌──────┬─────┐
   │ ░░░░░░░░░░ │  slightly         │ ▓▓▓▓ │▒█░▓ │  half destroyed
   │ ░░░░░░░░░░ │  too bright       │ ▓▓▓▓ │░▓█▒ │
   │ ░░░░░░░░░░ │  everywhere       │ ▓▓▓▓ │█░▒▓ │
   └────────────┘                   └──────┴─────┘

   SAME PSNR. Wildly different usefulness.
```

Image A is perfectly usable — every structure is intact, it is just a bit bright.
Image B is half garbage. PSNR cannot tell them apart. For defect inspection that
distinction is everything, which is why we need a structural metric.

## 4.0.2 SSIM — structural similarity

SSIM was designed to match human judgement better than PSNR. Instead of comparing
pixels individually, it slides a small window (typically 11×11) across both
images and, at each position, asks three questions about the two patches.

```
   ┌─────────────────┐         ┌─────────────────┐
   │      ┌───┐      │         │      ┌───┐      │
   │      │win│      │         │      │win│      │      compare these
   │      └───┘      │         │      └───┘      │      two windows
   │   predicted     │         │      true       │
   └─────────────────┘         └─────────────────┘
```

**Question 1 — Luminance.** Are the two windows equally bright on average?
Compares μx against μy.

**Question 2 — Contrast.** Do they have the same amount of variation? Compares
σx against σy. A flat grey patch and a high-contrast patch have very different σ.

**Question 3 — Structure.** Ignoring brightness and contrast, do they vary
*in the same places*? This is the covariance σxy — when one window goes bright,
does the other go bright at the same spot?

```
            (2·μx·μy + c1)     (2·σxy + c2)
   SSIM  =  ──────────────  ·  ──────────────
            (μx² + μy² + c1)   (σx² + σy² + c2)
             └─ luminance ─┘    └─ contrast × structure ─┘
```

The `c1` and `c2` are tiny constants that stop the fractions blowing up when
both denominators approach zero on flat patches. Each fraction equals 1 when the
two quantities match and less than 1 otherwise.

**Range: −1 to 1, where 1 means identical.** In practice for restoration you see
values from about 0.3 (poor) to 0.95 (excellent).

**Worked intuition:** brighten an entire image by 10%. Every pixel changes, so
PSNR collapses. But μx and μy both shift, the luminance term absorbs most of it,
and σ and σxy are unchanged because *relative* variation is identical. SSIM
barely moves. That is the behaviour you want: the structure survived, and so
does the score.

**Your numbers:** bicubic 0.5079, your model 0.4666 at epoch 3, climbing. Round 1
on easier data reached 0.718.

### Edge-stratified SSIM — your over-smoothing alarm

We compute SSIM separately on the top 25% highest-gradient pixels (edges) and
the rest (flat regions). This is not standard, and it exists for a specific
reason.

```
   BICUBIC (passes noise straight through):
      edge SSIM 0.596  >  flat SSIM 0.478
      flat regions are pure noise, so they score badly
      edges at least have real structure to match

   AN OVER-SMOOTHING MODEL:
      edge SSIM LOW   <  flat SSIM HIGH
      it wiped the noise from flat areas (easy win)
      but blurred the edges away too (the actual damage)
```

**So if edge SSIM ever falls below overall SSIM, the model is blurring.** The
training script prints a warning when this happens. KLA's specification calls
this failure out by name: *"Do not blur the image to remove noise, that destroys
useful information."* This metric is how you detect it happening.

At epoch 3 you had edge 0.5118 versus overall 0.4666 — edges scoring *higher*,
which is exactly right.

## 4.0.3 LPIPS — perceptual distance

Both metrics so far are hand-designed formulas. LPIPS takes a different approach:
it asks a neural network.

```
   predicted ──► VGG ──► feature maps at 5 depths ──┐
                                                     ├─► weighted distance
   true      ──► VGG ──► feature maps at 5 depths ──┘
```

VGG is an image classifier trained on millions of photographs. To classify well
it had to learn internal representations of edges, textures, patterns and shapes.
LPIPS discards the classification output and compares those **internal feature
maps** instead of raw pixels.

The reasoning: two images that produce similar activations in a network trained
to understand images are probably perceptually similar to a human, even if their
pixel values differ. It was calibrated against human judgements — people were
shown image pairs and asked which looked more similar — and it matches those
judgements better than PSNR or SSIM.

**LOWER IS BETTER — this is the opposite of the other two.** It is a *distance*,
not a similarity. Rough scale:

| LPIPS | meaning |
|---|---|
| 0.0 | identical |
| 0.1 | very close, hard to tell apart |
| 0.3 | noticeably different |
| 0.5 | clearly different images |

**Your numbers:** bicubic 0.4839, your model 0.4507 at epoch 3 and dropping.
This was your strongest axis in round 1 too — the model reached 0.313 against
bicubic's 0.427.

**The caveat worth knowing.** VGG was trained on natural photographs — cats,
cars, landscapes. Your images are electron micrographs of material surfaces.
Whether VGG's learned notion of "perceptually similar" transfers cleanly to SEM
texture is genuinely uncertain. KLA scores it, so you optimise it, but treat it
as a scored objective rather than as truth about how good your images look.

## 4.0.4 Why three metrics, and how they disagree

Each one is blind to something the others catch:

```
   PSNR   sees: exact pixel values
          blind to: whether structure survived
          fooled by: a uniform brightness shift

   SSIM   sees: local structure, contrast, correlation
          blind to: fine texture realism
          fooled by: plausible-looking smoothing

   LPIPS  sees: perceptual texture and pattern
          blind to: exact intensity accuracy
          fooled by: hallucinated detail that looks right but is invented
```

That last point matters for your problem specifically. A model can score well on
LPIPS by *inventing* convincing texture that was never in the true image. For
photographs that is often fine. For defect inspection it is dangerous — an
invented feature could look like a defect, or hide one. This is why the LPIPS
weight in the loss is only 0.05 rather than something larger.

**Round 1 gave you a direct demonstration of the disagreement.** Two models:

| | model A | model B | winner |
|---|---|---|---|
| PSNR | 22.88 | 23.36 | B |
| SSIM | 0.7182 | 0.7322 | B |
| edge SSIM | 0.7256 | 0.7179 | **A** |
| LPIPS | 0.3127 | 0.3608 | **A** |

B won on the two headline metrics by smoothing harder — and paid for it with 15%
worse perceptual quality and edge SSIM dropping below overall SSIM. A was the
better model despite losing on PSNR and SSIM. Reading only one metric would have
selected the wrong one.

**KLA combines all three with undisclosed weights**, which is precisely to stop
you gaming any single one.

## 4.0.5 Reading your training log

```
ep 003/80 | loss 0.2938 | OOD psnr 19.09 ssim 0.4666 edge 0.5118 flat 0.4515 lpips 0.4507
             ▲            ▲    ▲          ▲          ▲          ▲          ▲
             │            │    │          │          │          │          └ perceptual distance, LOWER better
             │            │    │          │          │          └ SSIM on flat regions
             │            │    │          │          └ SSIM on edges — should stay ABOVE overall
             │            │    │          └ overall structural similarity, HIGHER better
             │            │    └ pixel accuracy in dB, HIGHER better
             │            └ measured on the held-out OOD cluster, never trained on
             └ training loss — only comparable to itself, not to metrics
```

`OOD` means the 297 real image pairs from the held-out content cluster. The model
never sees them during training, so these numbers estimate how it will do on
KLA's unfamiliar test images.

The training loss is a different quantity entirely — a weighted blend of three
loss terms on training data. It tells you optimisation is progressing. It does
not tell you the model is getting better at the real task; only the OOD metrics
do.

---

# Part 5 — The loss functions

Metrics measure quality. **Losses** are what the model actually optimises. They
are not the same thing: a loss must be differentiable so gradients can flow
backwards, which rules out some metrics directly, and it can encode priorities a
metric does not.

KLA scores a weighted blend of PSNR, SSIM and LPIPS, so the loss targets all
three.

## 5.1 Charbonnier — robust pixel accuracy

```
   L = mean( sqrt( (pred − target)² + ε² ) )        ε = 1e-3
```

```
   loss │                                    ╱  MSE: grows as error²
        │                                  ╱
        │                              ╱ ╱
        │                          ╱  ╱
        │                     ╱   ╱      Charbonnier ≈ |error|
        │                ╱  ╱             (linear, robust)
        │           ╱ ╱
        │      ╱╱╱
        └──────────────────────────► error
```

MSE squares errors, so a single pixel with error 0.5 contributes as much as 100
pixels with error 0.05. With heavy-tailed speckle producing exactly those rare
huge errors, MSE-trained models spend their capacity chasing outliers and
smooth everything else to hedge. Charbonnier behaves like absolute error for
large deviations, so outliers do not dominate.

The `ε²` inside the square root exists purely so the function is differentiable
at zero — plain `|x|` has an undefined gradient there.

## 5.2 Edge weighting

```python
w = 1 + alpha * (gradient_magnitude / max_gradient)      # alpha = 4
w = w / w.mean()                                          # normalise to mean 1
loss = (charbonnier_per_pixel * w).mean()
```

SEM images are wall-to-wall texture, and the pixels that matter most are the
ones on structural boundaries. This weights the loss up to about 2.6× on edges
and down to 0.45× on flat areas.

**The normalisation line was a real bug fix.** Without dividing by the mean, the
weight map averaged 2.12, so the loss was silently 2.12× larger than plain
Charbonnier — which rescales the effective learning rate on that term without
anyone intending it. Dividing by the mean preserves the relative emphasis
(still ~5.8× edge-versus-flat) while restoring the scale.

## 5.3 Sobel edge loss

```
   Sobel_x = [-1  0  1]      Sobel_y = [-1 -2 -1]
             [-2  0  2]                [ 0  0  0]
             [-1  0  1]                [ 1  2  1]
```

These convolutions approximate spatial derivatives — they respond to
brightness changes. The loss compares gradients of prediction and target rather
than the values themselves, so it directly penalises blurred edges even when
absolute intensities are close. A blurred edge has a *smaller* gradient, and this
loss notices.

## 5.4 SSIM as a loss

PSNR treats every pixel independently and correlates poorly with what humans see.
SSIM compares local statistics in a sliding window:

```
            (2·μx·μy + c1)(2·σxy + c2)
   SSIM = ────────────────────────────────
           (μx² + μy² + c1)(σx² + σy² + c2)
             └── luminance ──┘└── contrast + structure ──┘
```

Three comparisons: are the local **means** similar, are the local **variances**
similar, and do the two windows **co-vary** — do they go up and down together?
Range −1 to 1, with 1 being identical.

The key property: shift an entire image's brightness slightly and PSNR collapses
while SSIM barely moves, because the luminance term normalises it away. SSIM
cares about structure, which is what matters for detecting defects.

**Edge-stratified SSIM** — which we compute separately on high-gradient and
flat regions — is your over-smoothing detector. For bicubic, edge SSIM (0.596)
exceeds flat SSIM (0.478), because bicubic passes noise straight through and flat
regions are dominated by it. A model that *smooths* would invert this: flat
regions would score well, edges would suffer. **If edge SSIM falls below overall
SSIM, the model is blurring** — which is exactly the failure KLA's specification
warns against by name.

## 5.5 LPIPS as a loss

Push both images through a pretrained VGG network, take the internal feature
maps, and measure distance in that feature space rather than pixel space.

```
   pred  ──► VGG ──► features at several depths ──┐
                                                   ├──► weighted L2 distance
   target ─► VGG ──► features at several depths ──┘
```

The intuition: VGG was trained on millions of natural images, so its internal
representations encode something about what makes images look similar to humans —
textures, edges, patterns — rather than raw pixel values. Two images can differ
substantially per-pixel yet be perceptually near-identical, and LPIPS captures
that where PSNR cannot.

**Lower is better**, unlike the other two.

**One honest caveat for this project:** VGG was trained on natural photographs,
not electron micrographs. Its notion of perceptual similarity on SEM texture
rests on shakier foundations than it would on photos. KLA scores it regardless,
so optimise it — but do not treat it as ground truth about visual quality here.

## 5.6 The combination

```python
loss = 1.0 * charbonnier_edge_weighted   # pixel accuracy → PSNR
     + 0.5 * sobel_edge                  # sharpness      → SSIM
     + 0.05 * lpips                      # perceptual     → LPIPS
```

The weights are not arbitrary. LPIPS is small because it operates on a different
scale and, pushed too hard, will happily trade real pixel accuracy for
texture that merely *looks* plausible — hallucination, which in defect inspection
is worse than blur. Round 1 provided direct evidence: the run that dropped LPIPS
entirely gained 0.48 dB PSNR but lost 15% on LPIPS and flipped edge SSIM below
overall SSIM. That model was smoother and worse.

---

# Part 6 — Training mechanics

## 6.1 Mixed precision (AMP)

Most operations run in 16-bit instead of 32-bit. On a T4, tensor cores make
half-precision matrix multiplies roughly twice as fast, and memory traffic halves.

The catch is that fp16's smallest representable number is about 6e-8, so small
gradients underflow to zero and vanish. `GradScaler` fixes this by multiplying the
loss by a large constant before backpropagation — scaling gradients up into
representable range — then dividing them back out before the optimiser step.

## 6.2 Cosine annealing

```
   lr │╲
      │ ╲___
      │     ╲___
      │         ╲____
      │              ╲______
      │                     ╲________
      └──────────────────────────────► epoch
      5e-4                        1e-6
```

High learning rate early to explore broadly; low late to settle into a minimum
precisely. Smooth decay avoids the disruption of sudden step drops.

**Important subtlety you hit already:** `T_max` is set to the total epoch count,
so `--epochs 10` is not a truncated 80-epoch run — it is a *complete* 10-epoch
schedule that anneals fully. The two are not comparable, and a short run's final
number is not a preview of the long run's trajectory.

## 6.3 Curriculum on degradation width

```python
w = 0.3 + 0.7 * (epoch - 1) / (epochs - 1)
train_ds.set_width(w)
```

Early training sees a narrow range of noise levels — an easier, more consistent
problem. As training progresses the range widens to ±30%, forcing robustness.
The same principle as teaching easy cases first.

## 6.4 The augmentations

**D4 (dihedral group).** The 8 symmetries of a square: 4 rotations × optional
flip. Free, and valid here because SEM texture has no preferred global
orientation — the 2D FFT is close to isotropic. It would be questionable on
images with a strong "up".

**Scale jitter (0.7–1.4×).** Resize the ground truth *before* degrading it, so
the model sees structures at varying apparent sizes. This is your only defence
against evaluation images at 512×512, since every training pair is 256←128 and
the model has never seen that scale.

**CutBlur (adapted).** The original assumes input and output are the same size.
Yours upsamples internally, so instead a rectangle of the noisy input is replaced
with a clean, noise-free downsample of the corresponding target region.

```
   noisy LR input              after CutBlur
   ┌──────────────┐            ┌──────────────┐
   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│            │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│    ───►    │▓▓▓┌──────┐▓▓▓│
   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│            │▓▓▓│ CLEAN│▓▓▓│
   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│            │▓▓▓└──────┘▓▓▓│
   └──────────────┘            └──────────────┘
```

The model must now decide *where* restoration is needed rather than applying a
fixed amount everywhere. That is precisely the mechanism by which CutBlur reduces
over-smoothing.

---

# Part 7 — How the pieces connect

```
  ┌─ EDA ─────────────────────────────────────────────────────────┐
  │  kurtosis 2.08  ──► Gaussian noise generator is inadequate    │
  │                      ──► resample real residuals instead      │
  │                                                                │
  │  autocorr + local variance ──► kernel is gauss:0.6            │
  │                      ──► refit noise ──► additive term appears│
  │                          matching KLA's stated physics        │
  │                                                                │
  │  spectrum flattens at 0.4 ──► above that the model is         │
  │                               inventing, not recovering        │
  └────────────────────────┬───────────────────────────────────────┘
                           ▼
  ┌─ SYNTHETIC DATA ──────────────────────────────────────────────┐
  │  3,460 unpaired clean images become usable                    │
  │  validated: autocorr within 0.012, texture energy within 0.1% │
  └────────────────────────┬───────────────────────────────────────┘
                           ▼
  ┌─ MODEL ───────────────────────────────────────────────────────┐
  │  global residual ──► only learn the correction                │
  │  NAFBlocks ──► cheap non-linearity, identity at init          │
  │  U-Net skip ──► context from bottleneck, detail from skip     │
  │  PixelShuffle ──► upsample without checkerboard               │
  └────────────────────────┬───────────────────────────────────────┘
                           ▼
  ┌─ LOSS ────────────────────────────────────────────────────────┐
  │  Charbonnier (robust to the heavy tails EDA found)            │
  │  + edge weighting (SEM is all texture)                        │
  │  + Sobel (penalise blur directly)                             │
  │  + LPIPS (perceptual, because KLA scores it)                  │
  └────────────────────────┬───────────────────────────────────────┘
                           ▼
  ┌─ VALIDATION ──────────────────────────────────────────────────┐
  │  held-out content cluster, real pairs only                    │
  │  edge SSIM vs overall SSIM = the over-smoothing alarm         │
  └───────────────────────────────────────────────────────────────┘
```

---

# Questions worth being able to answer

1. Two noise distributions have identical mean and variance. What could still
   make one much harder to model, and how would you detect it?
2. Why did the low-pass MSE kernel test fail, and what does that tell you about
   designing measurements generally?
3. What is aliasing, and why does antialiasing change the high-frequency content
   of a downsampled image?
4. Why does multiplying two halves of a feature map provide a usable
   non-linearity, and why might it be preferable to ReLU?
5. If `beta` and `gamma` start at zero, what does the network compute on its very
   first forward pass?
6. Why does PixelShuffle avoid checkerboard artefacts when transposed
   convolution does not?
7. What does it mean if edge SSIM falls below overall SSIM, and why does that
   matter for this specific problem?
8. Why is MSE a poor choice given the noise you measured?
9. What does the flattening of the power spectrum at 0.4 imply about what the
   model is doing at high frequencies?
10. Why can you not validate on synthetically degraded images?
