# Non-local attention — what we measured and what to build

Handoff note. Everything below is implemented on branch `v2` and switched off by
default, so pulling this changes nothing until you set a flag.

---

## The finding

Our model has been stuck at the same ceiling regardless of what we throw at it.
Width (dim 64 → 96), depth (levels 1 → 2), spectral loss, sharpening — every one
of them landed within 0.05 dB and 0.004 SSIM of the others. We eventually worked
out why: the ground-truth images carry their own acquisition noise. Their radial
power spectrum goes flat above roughly 60% of maximum frequency (tail flatness
std/mean = 0.024), which is the signature of a white-noise floor. A fixed share
of the residual error is unpredictable by definition, so no amount of capacity
touches it.

That framing suggested the ceiling was information-theoretic: one noisy
observation, no second measurement to average against. The physically correct
fix is frame averaging at acquisition — scan the field several times, noise
falls as √N, structure reinforces — but that costs scan time, which is the exact
constraint the whole problem exists to work around.

**There is one way around that argument, and we tested it.** If the same texture
patch recurs many times *inside a single frame*, and the noise on each instance
is independent, you can average those instances and get the same √N benefit with
no second acquisition. It is frame averaging performed in space instead of in
time. This is what BM3D does classically and what non-local attention does in
networks.

`scripts/measure_recurrence.py` measures it directly. On all 4,785 pairs, 8×8
patches, k=16 neighbours, matches required to come from ≥12 px away:

| what we averaged | PSNR vs clean | change |
|---|---|---|
| single noisy patch | 20.08 dB | — |
| 16 spatially nearest patches | 21.54 dB | +1.46 |
| **16 matched patches, same image** | **23.42 dB** | **+3.34** |
| 16 matched patches, *oracle* matching | 24.22 dB | +4.13 |
| 16 patches from a **different image** | 18.09 dB | −1.99 |

Read the last row first — it is the control. Any two patches share global
brightness and contrast statistics, so "patches look alike" is trivially true
and proves nothing. Averaging patches from a *different* image makes things
**worse** by 2 dB, while averaging matched patches from the same image makes
them better by 3.34 dB. That 5.3 dB spread is the evidence: the matching is
finding real recurring structure.

Two more numbers that matter:

- **Oracle vs practical is only 0.79 dB.** "Oracle" finds matches using the
  clean image; "practical" finds them using the noisy image, as you would at
  test time. The small gap means noise does *not* prevent us from locating the
  matches. Most non-local methods die here — ours doesn't.
- **The noise reduction is 2.16×, not 16×.** Sixteen perfectly independent
  copies of identical structure would give 16×. We get 2.16×, because SEM
  texture recurs *approximately* rather than identically. Set expectations
  accordingly: this is worth maybe a few tenths of a dB end-to-end, not several.

We also ran this on the earlier partial dataset (1,325 pairs) and got the same
verdict with smaller numbers (+2.23 practical, +2.66 oracle), so it replicates
across two different content mixes.

---

## Why it should work when nothing else did

Sort every change we have tried by mechanism and a clean pattern falls out.

**Information-extraction methods all failed.** Wider model, deeper model,
spectral loss, unsharp masking — each tried to pull more out of the input, and
each returned nothing, because the information wasn't there to pull.

**Error-reduction methods all worked.** More real training data (+3 dB over
bicubic, and the single biggest gain we have), test-time augmentation (+0.18 dB,
+0.011 SSIM), model ensembling (+0.21 dB without TTA). Each reduced the model's
own random error rather than trying to invent signal.

Non-local attention is the first idea that is information-extraction **with
information actually available to extract** — not from the input pixel's
neighbourhood, but from everywhere else in the frame. Our model physically
cannot reach it. It is convolutional with a ~60 px receptive field at levels=2,
so a pixel in one corner has no mechanism to know that a matching patch sits in
the other corner. That is an architectural limit, not a capacity limit, which is
precisely why making the model bigger did nothing.

---

## What is implemented and where

Branch `v2`. All of it is inert until `model.non_local=true`.

**`src/model.py` — `class NonLocalBlock`.** Multi-head self-attention over the
whole feature map. Four things about it are deliberate:

1. **It sits at the bottleneck**, in the middle of the `middle` block stack —
   after some local processing has built features worth matching on, and with
   blocks after it to integrate what it gathered. Resolution is lowest there, so
   it is affordable: at `levels=2`, a 128 px input is 32×32 at the bottleneck,
   which is 1,024 tokens.
2. **`gamma` is zero-initialised**, so the block is an exact identity at
   startup and training decides how much to use it. Same convention NAFNet uses
   for its residual scales. An untrained block cannot make the model worse — the
   self-test asserts this.
3. **`nl_kv_stride` pools keys and values** (default 2). Cost then grows as
   HW × HW/4 instead of HW², which is what keeps a 512 px input feasible.
   Queries stay full-resolution, so every position still gets its own answer.
4. **It uses `F.scaled_dot_product_attention`**, which selects a memory-efficient
   kernel and never materialises the full attention matrix. Without it a 512 px
   input would need roughly a gigabyte per head just for the scores.

**`configs/default.yaml`** — `model.non_local`, `model.nl_heads`,
`model.nl_kv_stride`, all documented inline with the measurement table.

**`train.py`** — passes the three flags into `build_model`.

**`run.py` and `scripts/check_sizes.py`** — detect attention from the checkpoint
(`.to_kv.` keys) and rebuild the model accordingly, so **the evaluator never has
to pass a flag**. This matters: KLA runs `run.py` unmodified.

**`scripts/measure_recurrence.py`** — the measurement above. Re-run it if you
want to check a different patch size or k.

**`scripts/selftest_round3.py` — section [7]** — verifies the block adds
parameters, outputs 2× the input, is finite, is an *exact* identity at zero-init,
and runs at 32/64/128 px inputs.

---

## How to test it

**1. Self-test first.** Ten seconds, no data, no GPU. If it fails, stop.

```bash
python scripts/selftest_round3.py
```

**2. Confirm the finding on your own machine** before spending GPU time:

```bash
python scripts/measure_recurrence.py --set data.root=$DATA --images 25 --k 16
```

Header should read `images=25 (from splits.json[val_ood])` and `kernel=gauss:0.6`.
If it says `directory scan`, run `scripts/make_splits.py` first — and check the
split reports **4,785 paired images** and **val_ood 1,165**. There is an older
1,325-pair dataset floating around; training on it scores *below* bicubic.

**3. Train the A/B.** Same config as our best model, one flag apart:

```bash
python train.py --epochs 50 --set data.root=$DATA cache.dir=$CACHE \
    dataset.synth_p=0.0 model.dim=48 model.levels=2 \
    train.output_dir=/kaggle/working/runs/nonlocal \
    model.non_local=true
```

**Baseline to beat: SSIM 0.5151, PSNR 23.68**, measured on val_ood (cluster 3,
1,165 images) with dim48/levels=2 trained on the full data for 50 epochs. Check
`docs/RESULTS.md` for the full table.

**4. Check it still runs at 512.** Attention memory grows faster than
convolution, and this is the one failure that would invalidate our submission:

```bash
python scripts/check_sizes.py --weights /kaggle/working/runs/nonlocal/best_nafnet.pt \
    --sizes 128 256 512 1024
```

If 512 runs out of memory, raise `nl_kv_stride` to 4 before doing anything more
elaborate.

---

## What to watch, and when to stop

- **Epoch 1 should look normal** (~23.2 PSNR). Zero-init means the block starts
  as a no-op, so a wildly different first epoch means something is wired wrong.
- **Watch `flat` SSIM specifically.** SEM texture lives in the mid-gradient
  regions that our stratification calls "flat", so that is where a texture gain
  should appear first. Edge SSIM is the wrong indicator here — we got burned by
  that earlier.
- **Watch ms/image.** Inference time is scored. Our current model is 9.7 ms; if
  attention pushes it past roughly 15 ms the trade needs justifying.
- **Expect a small gain.** The 2.16× patch-level noise reduction does not
  translate one-for-one into end-to-end dB. If it comes back within noise of
  0.5151, that is a real answer and it goes in the report — it would mean our
  ceiling is genuinely information-theoretic rather than an artefact of choosing
  a convolutional architecture, which is a stronger claim than the block itself
  would have bought us.

**Don't combine this with `vst.enabled` or `loss.ms_ssim` on the first run.** Two
changes at once and the result cannot be attributed.
