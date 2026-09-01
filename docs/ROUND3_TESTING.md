# Round 3 — testing procedure

Run these in order. Steps 1–3 are the submission blockers and take about twenty
minutes; steps 4–7 are the experiments.

Everything defaults to **off**, so before you change any config the pipeline
behaves exactly as it did for the final run.

---

## Step 0 — pull the branch

```bash
cd /kaggle/working/oomsurvivors && git pull origin v2
```

If you have a running Kaggle session, **restart the kernel** afterwards. Python
caches imported modules, and `from src.vst import ...` will not pick up a file
that did not exist when the interpreter started. This has bitten us four times.

---

## Step 1 — self-test (10 seconds, no data, no GPU needed)

```bash
python scripts/selftest_round3.py
```

Expect `all checks passed - safe to train`. It verifies the VST round-trips
exactly, actually flattens the noise variance, matches between numpy and torch
including under AMP, that MS-SSIM is zero for identical images and rises
monotonically under blur, and that TTA runs the model exactly *n* times.

**If anything fails, stop.** A training run on top of a broken transform costs
hours and produces a number you cannot interpret.

You can also check the transform on its own:

```bash
python -m src.vst
```

which prints the measured noise-std spread before and after stabilisation —
should read `3.08x -> 1.02x`.

---

## Step 2 — the 512×512 check (2 minutes) ← **highest priority**

This is the one outstanding risk that could invalidate the submission.

```bash
python scripts/check_sizes.py --weights /kaggle/working/runs/final/best_nafnet.pt \
    --sizes 128 256 512 1024
```

You want `PASS` on every row, output exactly 2× input, and peak memory well
under the T4's 15 GB. Then repeat with TTA on, since that is how you would
actually submit:

```bash
python scripts/check_sizes.py --weights ... --sizes 256 512 --tta 4
```

**If 512 fails on memory**, the fix is tiled inference — process overlapping
crops and blend. Tell me and I will write it. **If it fails on shape**, the
input is not divisible by `2**levels`; `run.py` already reflect-pads for this,
so verify through `run.py` before assuming it is broken.

---

## Step 3 — end-to-end from a clean clone (5 minutes)

The specification requires `run.py` to work with no manual edits.

```bash
cd /kaggle/working && rm -rf clean_test
git clone -b v2 <your-repo-url> clean_test && cd clean_test
mkdir -p weights && cp /kaggle/working/runs/final/best_nafnet.pt weights/

mkdir -p /tmp/in && python -c "
import numpy as np, pathlib
for i in range(5):
    np.save(f'/tmp/in/{i:06d}.npy', np.random.rand(128,128).astype('float32'))
"
python run.py /tmp/in /tmp/out
python -c "
import numpy as np, glob
fs = sorted(glob.glob('/tmp/out/*.npy'))
print(len(fs), 'files')
a = np.load(fs[0]); print(a.shape, a.dtype, a.min(), a.max())
assert a.shape == (256,256) and np.isfinite(a).all()
print('clean-clone run: ok')
"
```

---

## Step 4 — measure TTA2 (10 minutes)

You measured 1, 4 and 8 but never 2. All the gain is in the 1→4 jump, so it
matters where between them it arrives — 2 transforms cost ~19 ms against 34 ms
for 4.

In a notebook cell:

```python
import torch
from src.tta import compare_tta

res = compare_tta(model, val_loader, device, lpips_fn, variants=(1, 2, 4, 8))
for n, r in res.items():
    print(f"tta={n}  psnr {r['psnr']:.3f}  ssim {r['ssim']:.4f}  "
          f"{r['ms_per_image']:.2f} ms/img")
```

This also gives you the **TTA4 number on the final checkpoint**, which you
currently do not have — the headline SSIM in the handover report assumes the
+0.011 gain transfers from the earlier run's weights.

---

## Step 5 — VST training run (~45 minutes)

```bash
python train.py --epochs 50 \
    --set data.root=$DATA cache.dir=$CACHE \
          dataset.synth_p=0.0 model.dim=48 model.levels=2 \
          vst.enabled=true \
          train.output_dir=/kaggle/working/runs/vst
```

**Sanity checks in the first two epochs:**

- The log prints `VST enabled: VST(a=0.023807, ...)` at startup. If it does
  not, the config override did not take.
- Epoch-1 loss will be a **different absolute number** from previous runs —
  Charbonnier is now measured in stabilised space, where noise std is 0.106
  rather than varying with intensity. Do not compare loss values across the
  flag; compare the OOD metrics, which are still in image space.
- Epoch-1 PSNR should land in the same ballpark as before (~23.2). If it comes
  out at 10 dB, the inverse transform is misapplied — stop and tell me.

**What success looks like:** SSIM above 0.5151 at 50 epochs. The mechanism is
that dark regions stop being drowned out by bright ones in the loss, so watch
`flat` in particular — that is where the effect should show first.

**What failure looks like:** metrics within noise of the baseline. That is a
real result and goes in the report either way, because "we derived the
principled transform from our measured noise model and it did not help" is a
finding, not a wasted run.

---

## Step 6 — MS-SSIM run (~45 minutes)

```bash
python train.py --epochs 50 \
    --set data.root=$DATA cache.dir=$CACHE \
          dataset.synth_p=0.0 model.dim=48 model.levels=2 \
          loss.ms_ssim=0.3 \
          train.output_dir=/kaggle/working/runs/msssim
```

Expect SSIM up and PSNR down — that is the trade every loss experiment has
made, and this time SSIM is the metric being scored. `0.3` is a starting guess;
if SSIM barely moves the weight is too low, and if PSNR falls more than about
0.5 dB it is too high.

Do **not** combine this with `vst.enabled=true` on the first attempt. Two
changes at once and you cannot attribute the result.

---

## Step 7 — internal patch recurrence (~3 minutes, CPU)

Run this **before** deciding whether to build a non-local block. It answers, for
this dataset, whether the same texture repeats inside a frame often enough to
denoise by averaging distant patches — the one mechanism that could beat the
single-observation ceiling without a second acquisition.

```bash
python scripts/measure_recurrence.py --set data.root=$DATA \
    --images 25 --k 16 --exclude 12
```

It prints a table and then a verdict. The number that decides it is
**`practical - local`**: non-local averaging has to beat *local* averaging,
because local averaging is just a blur, which your model already does for free
with a convolution.

Two guards are built in, and both matter:

- **Cross-image null.** Patches from a *different* image are averaged as a
  control. Any two patches share brightness and contrast statistics, so this is
  the floor a real match has to clear.
- **Spatial exclusion.** Matches must come from at least `--exclude` pixels
  away. Without it the nearest neighbours are the overlapping patches around the
  query, and you are measuring blur wearing a disguise.

The verdict logic was validated against three synthetic images with known
answers — a tiled motif (`WORTH BUILDING`, +21.4 dB over local), a smooth
random field (`MARGINAL`), and pure noise (`NOT WORTH IT`). The noise case is
instructive: it scores +3.07 dB against local averaging while being 6.35 dB
*worse* than doing nothing at all, so the script requires non-local to beat
both.

Sensitivity worth checking if the result is borderline:

```bash
for k in 8 16 32; do
  python scripts/measure_recurrence.py --set data.root=$DATA --k $k --images 15
done
```

More neighbours means more noise cancellation but looser matches. If the gain
grows with `k`, the redundancy is real; if it peaks and falls, you are averaging
in structure that does not belong.

**If the verdict is NOT WORTH IT, that is a result, not a failure.** It says
your ceiling is genuinely information-theoretic rather than an artefact of
choosing a convolutional architecture — which is a considerably stronger claim
to make in the report than anything you would gain from the block itself.

---

## Step 8 — Wiener baseline (~10 minutes, CPU)

```bash
python scripts/wiener_baseline.py --split val_ood --limit 200 \
    --set data.root=$DATA
```

No training. This is for the report, not the model: it gives you the optimal
*linear* estimator built from your own measured noise and signal spectra, which
is a far more honest comparator than bicubic. The gap between it and your
network measures exactly what nonlinearity buys.

It also makes the theoretical point for you — the Wiener gain goes to zero at
high frequencies where the ground truth is a noise floor, so the classical
theory independently derives the same low-pass behaviour you measured
empirically in the network.

---

## Save your checkpoints immediately

`/kaggle/working` does not survive a session restart. It has already cost this
project one checkpoint permanently. As soon as a run finishes:

```bash
cp /kaggle/working/runs/vst/best_nafnet.pt /kaggle/working/vst_best.pt
```

then download it through the Kaggle file browser before doing anything else.

---

## Recording results

Add a row to the table in `docs/RESULTS.md` for each run, then regenerate the
handover PDF:

```bash
python docs/build_handover_pdf.py docs/Round2_Handover_Report.pdf
```
