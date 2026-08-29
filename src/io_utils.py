"""Image I/O — the highest-risk component in this repo.

KLA scores the images exactly as saved by our pipeline and performs no
clipping or renormalisation of its own (problem statement, section 4A).
If we save in a lossier format than the ground truth, we quietly throw away
score no matter how good the model is.

Rules enforced here:
  1. Outputs are written in the SAME format and dtype as the ground truth.
  2. Loading is lossless: values are returned as float32 on the GT's own scale.
  3. round_trip_check() must pass before any training starts.

End-to-end runtime includes disk read and write, so these functions are also
on the timed path. Keep them allocation-light.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".npy", ".tif", ".tiff", ".png")


@dataclass(frozen=True)
class ImageFormat:
    """Everything needed to write a file indistinguishable from the GT files."""

    kind: str          # 'npy' | 'tiff32' | 'tiff16' | 'png16' | 'png8'
    suffix: str        # '.npy' | '.tif' | '.png'
    dtype: str         # numpy dtype name of the stored data
    scale: float       # stored_value = float_value * scale

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ImageFormat":
        return ImageFormat(**d)


def detect_format(path: str | Path) -> ImageFormat:
    """Infer the on-disk format of a single reference image."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        return ImageFormat("npy", ".npy", str(arr.dtype), 1.0)

    if suffix in (".tif", ".tiff"):
        import tifffile

        arr = tifffile.imread(path)
        if arr.dtype == np.uint16:
            return ImageFormat("tiff16", suffix, "uint16", 65535.0)
        if arr.dtype == np.uint8:
            return ImageFormat("tiff8", suffix, "uint8", 255.0)
        return ImageFormat("tiff32", suffix, str(arr.dtype), 1.0)

    if suffix == ".png":
        from PIL import Image

        with Image.open(path) as im:
            mode = im.mode
        if mode in ("I;16", "I;16B", "I", "I;16L"):
            return ImageFormat("png16", ".png", "uint16", 65535.0)
        return ImageFormat("png8", ".png", "uint8", 255.0)

    raise ValueError(f"Unsupported image suffix: {suffix!r} ({path})")


def load_image(path: str | Path) -> np.ndarray:
    """Load as float32 on the ground-truth scale (GT in [0,1], LR may exceed).

    Never clips. The out-of-range values in NoisyLR are intentional signal
    about the speckle process and must survive loading.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
    elif suffix in (".tif", ".tiff"):
        import tifffile

        arr = tifffile.imread(path)
    elif suffix == ".png":
        from PIL import Image

        with Image.open(path) as im:
            arr = np.array(im)
    else:
        raise ValueError(f"Unsupported image suffix: {suffix!r} ({path})")

    if arr.ndim == 3:
        # Grayscale challenge: collapse only if the channels really are identical.
        if arr.shape[-1] in (3, 4) and np.all(arr[..., 0] == arr[..., 1]):
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0]

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return arr.astype(np.float32) / np.float32(info.max)
    return arr.astype(np.float32, copy=False)


def save_image(arr: np.ndarray, path: str | Path, fmt: ImageFormat) -> None:
    """Write `arr` (float32, GT scale) in exactly the ground-truth format.

    Clipping to [0,1] happens here because KLA does not do it for us and GT is
    guaranteed to lie in [0,1].
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, dtype=np.float32)

    if fmt.kind == "npy":
        np.save(path, np.clip(arr, 0.0, 1.0).astype(fmt.dtype))
        return

    if fmt.kind in ("tiff32",):
        import tifffile

        tifffile.imwrite(path, np.clip(arr, 0.0, 1.0).astype(fmt.dtype))
        return

    # Integer-backed formats: round-half-up then cast.
    maxval = np.float32(fmt.scale)
    q = np.clip(arr, 0.0, 1.0) * maxval
    q = np.rint(q).astype(fmt.dtype)

    if fmt.kind in ("tiff16", "tiff8"):
        import tifffile

        tifffile.imwrite(path, q)
        return

    from PIL import Image

    if fmt.kind == "png16":
        Image.fromarray(q.astype(np.uint16), mode="I;16").save(path)
    else:
        Image.fromarray(q.astype(np.uint8), mode="L").save(path)


def output_path(input_path: str | Path, out_dir: str | Path, fmt: ImageFormat) -> Path:
    """Preserve the input filename stem; use the ground-truth suffix."""
    return Path(out_dir) / (Path(input_path).stem + fmt.suffix)


def round_trip_check(reference: str | Path, tol: float = 0.0) -> dict:
    """Assert save(load(x)) reproduces the reference file's values.

    Run this BEFORE anything else. If it fails, every downstream metric is
    capped by an I/O bug rather than by the model.
    """
    reference = Path(reference)
    fmt = detect_format(reference)
    original = load_image(reference)

    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / ("rt" + fmt.suffix)
        save_image(original, dst, fmt)
        restored = load_image(dst)

    max_abs = float(np.max(np.abs(original - restored)))
    # Integer formats quantise; one LSB of round-off is the floor.
    allowed = tol if tol > 0 else (0.0 if fmt.scale == 1.0 else 0.5 / fmt.scale + 1e-9)
    return {
        "format": fmt.to_dict(),
        "max_abs_error": max_abs,
        "allowed": allowed,
        "passed": max_abs <= allowed,
        "shape": list(original.shape),
        "min": float(original.min()),
        "max": float(original.max()),
    }


def is_readable(path: str | Path) -> bool:
    """Cheap check that a file is complete and loadable.

    Datasets delivered as archives can contain zero-byte or half-written files
    when the transfer was interrupted - the round-2 download left exactly one
    such NoisyLR file. A single bad file crashes any script that iterates the
    whole set, so every entry point filters through this.

    Uses mmap for .npy (reads the header plus one element) rather than a full
    load, so scanning thousands of files stays fast.
    """
    p = Path(path)
    try:
        if p.stat().st_size == 0:
            return False
        if p.suffix.lower() == ".npy":
            a = np.load(p, mmap_mode="r")
            if a.size == 0:
                return False
            _ = a.reshape(-1)[0]        # force a real read of the data region
            return True
        return load_image(p).size > 0
    except Exception:
        return False


def list_images(directory: str | Path, validate: bool = False) -> list[Path]:
    directory = Path(directory)
    files = [p for p in sorted(directory.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    if validate:
        good = [p for p in files if is_readable(p)]
        if len(good) != len(files):
            bad = [p.name for p in files if p not in set(good)]
            print(f"!! skipping {len(files) - len(good)} unreadable file(s) in "
                  f"{directory.name}: {bad[:5]}{' ...' if len(bad) > 5 else ''}")
        return good
    return files


def pair_by_stem(gt_dir: str | Path, lr_dir: str | Path,
                 validate: bool = True) -> list[tuple[Path, Path]]:
    """Match GT and NoisyLR files on filename stem.

    Returns only complete, readable pairs. Validation is ON by default: one
    truncated file is enough to crash a full pass over the dataset, and the
    round-2 delivery contains exactly that. Pass validate=False to skip the
    check when you know the data is clean and want the extra speed.

    Note the GT and LR counts are NOT expected to match in round 2 - KLA ships
    4,785 clean images against 1,326 pairs deliberately, so the unpaired clean
    images can be degraded synthetically.
    """
    gt = {p.stem: p for p in list_images(gt_dir, validate=validate)}
    lr = {p.stem: p for p in list_images(lr_dir, validate=validate)}
    common = sorted(set(gt) & set(lr))
    return [(gt[k], lr[k]) for k in common]
