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


class GlobalNonLocalBlock(nn.Module):
    """Full self-attention over the feature map, with pooled keys/values.

    Every position attends to EVERY other, which is the point. The redundancy
    we measured is spread across the whole frame:

        single noisy patch                    20.08 dB
        average of 16 spatial neighbours      21.54 dB   (+1.46)
        average of 16 MATCHED patches         23.42 dB   (+3.34)
        average of 16 patches, OTHER image    18.09 dB   (-1.99)   <- control

    and `scripts/measure_recurrence.py` required matches to sit at least 12px
    away, so the gain is specifically NON-local. See docs/NONLOCAL_HANDOFF.md.

    `kv_stride` pools keys and values so cost grows as HW x HW/stride^2 rather
    than (HW)^2, and SDPA never materialises the attention matrix - together
    that keeps a 512px input affordable without giving up global reach.

    `gamma` is zero-initialised, so the block starts as an exact identity.
    """

    def __init__(self, channels: int, heads: int = 4, kv_stride: int = 2):
        super().__init__()
        if channels % heads:
            heads = 1
        self.heads = heads
        self.kv_stride = kv_stride
        self.norm = LayerNorm2d(channels)
        self.to_q = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_kv = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x)
        q = self.to_q(y)
        src = (F.avg_pool2d(y, self.kv_stride)
               if self.kv_stride > 1 and min(h, w) >= 2 * self.kv_stride else y)
        k, v = self.to_kv(src).chunk(2, dim=1)

        def heads_first(t):
            bb, cc, hh, ww = t.shape
            return t.reshape(bb, self.heads, cc // self.heads,
                             hh * ww).transpose(-2, -1)

        o = F.scaled_dot_product_attention(heads_first(q), heads_first(k),
                                           heads_first(v))
        o = o.transpose(-2, -1).reshape(b, c, h, w)
        return x + self.gamma * self.proj(o)


class WindowedNonLocalBlock(nn.Module):
    """Swin-Style Windowed Multi-Head Self-Attention.

    Partitions the feature map into non-overlapping windows (e.g. 8x8), so
    complexity drops from O((HW)^2) to O(HW * M^2).

    CAVEAT worth measuring before preferring this to the global variant: at
    levels=2 the bottleneck is 32x32 for a 128px input, so an 8x8 window spans
    about 32px in input coordinates - comparable to the ~60px the convolutions
    already reach. The recurrence measurement that motivated attention required
    matches at least 12px away and found the benefit spread across the whole
    frame, which a window this size cannot see. The cyclic shift propagates
    information across windows only over DEPTH, and there is a single non-local
    block in the stack, so with one block the shift has nothing to alternate
    with. Run both and compare rather than assuming the cheaper one is equivalent.
    """

    def __init__(self, channels: int, heads: int = 4, window_size: int = 8, shift_size: int = 0):
        super().__init__()
        if channels % heads:
            heads = 1
        self.heads = heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm = LayerNorm2d(channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        # Zero-init: block is an identity until training gives it a reason.
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x)

        # Pad to ensure divisibility by window_size
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            y = F.pad(y, (0, pad_r, 0, pad_b))
        
        hp, wp = y.shape[2], y.shape[3]

        # Cyclic shift (Toroidal wrap is acceptable for stationary SEM texture)
        if self.shift_size > 0:
            y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        qkv = self.to_qkv(y)
        q, k, v = qkv.chunk(3, dim=1)

        def window_partition(t):
            bb, cc, hh, ww = t.shape
            t = t.view(bb, self.heads, cc // self.heads, hh // self.window_size, self.window_size, ww // self.window_size, self.window_size)
            t = t.permute(0, 3, 5, 1, 4, 6, 2).contiguous()
            return t.view(-1, self.heads, self.window_size * self.window_size, cc // self.heads)

        q_w, k_w, v_w = window_partition(q), window_partition(k), window_partition(v)

        # SDPA efficiently handles the windowed attention matrices
        o_w = F.scaled_dot_product_attention(q_w, k_w, v_w)

        def window_reverse(t, bb, cc, hh, ww):
            t = t.view(bb, hh // self.window_size, ww // self.window_size, self.heads, self.window_size, self.window_size, cc // self.heads)
            t = t.permute(0, 3, 6, 1, 4, 2, 5).contiguous()
            return t.view(bb, cc, hh, ww)
        
        o = window_reverse(o_w, b, c, hp, wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            o = torch.roll(o, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        
        # Remove padding
        if pad_r > 0 or pad_b > 0:
            o = o[:, :, :h, :w]
        
        return x + self.gamma * self.proj(o)


def NonLocalBlock(channels, heads=4, mode="global", kv_stride=2,
                  window_size=8, shift_size=0):
    """Dispatch to the global or windowed attention block.

    Both are kept so the choice can be measured rather than assumed. `global`
    is the default because it is what the recurrence measurement justifies;
    `window` is cheaper and worth checking against it.
    """
    if mode == "window":
        return WindowedNonLocalBlock(channels, heads=heads,
                                     window_size=window_size,
                                     shift_size=shift_size)
    if mode != "global":
        raise ValueError(f"nl_mode must be 'global' or 'window', got {mode!r}")
    return GlobalNonLocalBlock(channels, heads=heads, kv_stride=kv_stride)


class NAFNet_UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dim=64, scale=2,
                 levels=1, blocks=2, middle_blocks=2,
                 non_local=False, nl_heads=4, nl_mode='global',
                 nl_kv_stride=2, nl_window_size=8):
        super().__init__()
        self.scale = scale
        self.levels = levels
        self.non_local = bool(non_local)

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
        mid = []
        for i in range(middle_blocks):
            mid.append(NAFBlock(c))
            if self.non_local:
                # Alternate shift sizes (0 then window_size // 2) for cross-window matching
                shift = (nl_window_size // 2) if (i % 2 == 1) else 0
                mid.append(NonLocalBlock(c, heads=nl_heads, mode=nl_mode,
                                         kv_stride=nl_kv_stride,
                                         window_size=nl_window_size,
                                         shift_size=shift))
        self.middle = nn.Sequential(*mid)

        # ---- decoder ----------------------------------------------------
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
    """Convert a pre-`levels` checkpoint to the current layer naming."""
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
            out[k] = v
    return out


def is_legacy_state_dict(sd: dict) -> bool:
    return any(k.startswith(("enc1.", "dec1.", "reduce.")) for k in sd)


@register("nafnet")
def _nafnet(scale=2, dim=64, levels=1, blocks=2, middle_blocks=2,
            non_local=False, nl_heads=4, nl_mode='global',
            nl_kv_stride=2, nl_window_size=8, **kwargs):
    """Width AND depth are configurable so both can be swept."""
    return NAFNet_UNet(in_channels=1, out_channels=1, dim=int(dim), scale=scale,
                       levels=int(levels), blocks=int(blocks),
                       middle_blocks=int(middle_blocks),
                       non_local=bool(non_local), nl_heads=int(nl_heads),
                       nl_mode=str(nl_mode), nl_kv_stride=int(nl_kv_stride),
                       nl_window_size=int(nl_window_size))


@register("bicubic")
def _bicubic(scale: int = 2, **_) -> nn.Module:
    return BicubicUpsample(scale)