"""Model registry — self-contained copy of the NAFNet architecture used for
training, embedded here so the submission has no dependency on any
project-internal `src/` package.

NOTE: the original registry had `build_model()` looking up `_REGISTRY`
while `@register` populated `MODELS` — those were never the same dict.
Fixed here so `build_model("nafnet", ...)` actually resolves.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS: dict = {}


def register(name):
    """Decorator to register models for easy access via config strings."""
    def decorator(cls_or_func):
        MODELS[name] = cls_or_func
        return cls_or_func
    return decorator


def build_model(name: str = "nafnet", **kwargs) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(MODELS)}")
    return MODELS[name](**kwargs)


class BicubicUpsample(nn.Module):
    """Zero-parameter baseline. Same interface as the real model."""

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        dw_channel = c * 2
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channel // 2)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0)

        ffn_channel = c * 2
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = self.conv3(self.sca(self.sg1(self.conv2(self.conv1(out)))))
        x = identity + out * self.beta

        identity2 = x
        out = self.norm2(x)
        out = self.conv5(self.sg2(self.conv4(out)))
        return identity2 + out * self.gamma


class NAFNet_UNet(nn.Module):
    """NAFNet U-Net with a configurable number of encoder/decoder levels.

    The original round-2 model used ONE downsampling level, which caps the
    receptive field at roughly 30 pixels. That is almost certainly why widening
    the network (dim 64 -> 96) bought nothing: width adds representational
    capacity but does not let the model see any further. Depth does.

    Receptive field roughly doubles per level:
        levels=1  ->  ~30 px   (round-2 baseline)
        levels=2  ->  ~70 px
        levels=3  -> ~150 px   (exceeds the 128px training patch)

    Channels double and resolution halves at each level, so the parameter count
    grows quickly with depth. `blocks` sets how many NAFBlocks sit at each
    encoder/decoder level.
    """

    def __init__(self, in_channels=1, out_channels=1, dim=64, scale=2,
                 levels=1, blocks=2, middle_blocks=2):
        super().__init__()
        self.scale = scale
        self.levels = levels

        self.intro = nn.Conv2d(in_channels, dim, 3, 1, 1)

        # ---- encoder ----------------------------------------------------
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        c = dim
        for _ in range(levels):
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(blocks)]))
            self.downs.append(nn.Conv2d(c, c * 2, 3, 2, 1))
            c *= 2

        # ---- bottleneck -------------------------------------------------
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(middle_blocks)])

        # ---- decoder ----------------------------------------------------
        # Each level: PixelShuffle back up, concat the skip, 1x1 to merge.
        self.ups = nn.ModuleList()
        self.reduces = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(levels):
            self.ups.append(nn.Sequential(
                nn.Conv2d(c, c * 2, 3, 1, 1),
                nn.PixelShuffle(2)))           # c*2 channels at 2x -> c//2
            c //= 2
            self.reduces.append(nn.Conv2d(c * 2, c, 1, 1, 0))   # merge skip
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(blocks)]))

        # ---- super-resolution tail --------------------------------------
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, out_channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale))

    def forward(self, x):
        # Global residual: the network only learns the CORRECTION to a plain
        # bilinear upscale, so an untrained model already emits something sane.
        shortcut = F.interpolate(x, scale_factor=self.scale, mode="bilinear",
                                 align_corners=False)

        out = self.intro(x)

        skips = []
        for enc, down in zip(self.encoders, self.downs):
            out = enc(out)
            skips.append(out)
            out = down(out)

        out = self.middle(out)

        for up, reduce, dec, skip in zip(self.ups, self.reduces,
                                         self.decoders, reversed(skips)):
            out = up(out)
            out = torch.cat([out, skip], dim=1)
            out = reduce(out)
            out = dec(out)

        return self.upsample(out) + shortcut


def remap_legacy_state_dict(sd: dict) -> dict:
    """Convert a pre-`levels` checkpoint to the current layer naming.

    The old fixed architecture was:
        intro, enc1(2 blocks @dim), down, enc2(2 @2dim),
        middle(2 @2dim), up, reduce, dec1(2 @dim), upsample

    `enc2` and `middle` both sit at 2*dim channels and the same resolution, so
    together they are simply four blocks in the bottleneck. The old model is
    therefore EXACTLY  levels=1, blocks=2, middle_blocks=4  under the new naming.

        enc1.{i}   -> encoders.0.{i}
        down       -> downs.0
        enc2.{i}   -> middle.{i}          (first two bottleneck blocks)
        middle.{i} -> middle.{i+2}        (last two)
        up.0       -> ups.0.0
        reduce     -> reduces.0
        dec1.{i}   -> decoders.0.{i}
    """
    out = {}
    for k, v in sd.items():
        if k.startswith("enc1."):
            out["encoders.0." + k[len("enc1."):]] = v
        elif k.startswith("down."):
            out["downs.0." + k[len("down."):]] = v
        elif k.startswith("enc2."):
            out["middle." + k[len("enc2."):]] = v
        elif k.startswith("middle."):
            rest = k[len("middle."):]
            idx, tail = rest.split(".", 1)
            out[f"middle.{int(idx) + 2}.{tail}"] = v
        elif k.startswith("up."):
            out["ups.0." + k[len("up."):]] = v
        elif k.startswith("reduce."):
            out["reduces.0." + k[len("reduce."):]] = v
        elif k.startswith("dec1."):
            out["decoders.0." + k[len("dec1."):]] = v
        else:
            out[k] = v                     # intro, upsample unchanged
    return out


def is_legacy_state_dict(sd: dict) -> bool:
    return any(k.startswith(("enc1.", "dec1.", "reduce.")) for k in sd)


@register("nafnet")
def _nafnet(scale=2, dim=64, levels=1, blocks=2, middle_blocks=2, **kwargs):
    """Width AND depth are configurable so both can be swept.

    Round-2 finding: dim=96 (2.2x the parameters of dim=64) gained +0.003 SSIM
    and no PSNR. Capacity was not the constraint. `levels` is the more promising
    knob - it controls receptive field, which width does not affect at all.

    Parameters scale roughly as dim^2 (each NAFBlock is ~7c^2 weights):
        dim=48  -> ~0.55M      dim=64  -> ~0.98M   (round-2 baseline)
        dim=96  -> ~2.2M       dim=128 -> ~3.9M

    Inference throughput is scored, so measure runtime alongside quality
    rather than assuming a bigger model is a better submission.
    """
    return NAFNet_UNet(in_channels=1, out_channels=1, dim=int(dim), scale=scale,
                       levels=int(levels), blocks=int(blocks),
                       middle_blocks=int(middle_blocks))


@register("bicubic")
def _bicubic(scale: int = 2, **_) -> nn.Module:
    return BicubicUpsample(scale)