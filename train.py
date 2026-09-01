#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision.transforms.functional as TF
from torch.optim.swa_utils import AveragedModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lpips

from src.augment import cutblur_sr, d4_batch
from src.config import add_config_args, load_config
from src.dataset import RestorationDataset, degrade_cfg_from_stats
from src.model import build_model
from src.splits import load_splits
from src.transforms import load_stats

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, weight=None):
        d = torch.sqrt((pred - target) ** 2 + self.eps ** 2)
        return (d * weight).mean() if weight is not None else d.mean()

class SpectralLoss(nn.Module):
    def __init__(self, hi_from: float = 0.25):
        super().__init__()
        self.hi_from = hi_from

    def forward(self, pred, target):
        with torch.autocast(pred.device.type, enabled=False):
            p = torch.fft.rfft2(pred.float(), norm="ortho").abs()
            t = torch.fft.rfft2(target.float(), norm="ortho").abs()
            if self.hi_from > 0:
                k = int(p.shape[-2] * self.hi_from)
                p, t = p[..., k:, :], t[..., k:, :]
            return F.l1_loss(torch.log1p(p), torch.log1p(t))

class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("wx", kx)
        self.register_buffer("wy", ky)
        self.l1 = nn.L1Loss()

    def _grad(self, x):
        return F.conv2d(x, self.wx, padding=1), F.conv2d(x, self.wy, padding=1)

    def forward(self, pred, target):
        px, py = self._grad(pred)
        tx, ty = self._grad(target)
        return self.l1(px, tx) + self.l1(py, ty)

    def edge_weight(self, target, alpha: float = 4.0):
        gx, gy = self._grad(target)
        g = torch.sqrt(gx ** 2 + gy ** 2)
        m = g.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        w = 1.0 + alpha * (g / m)
        return w / w.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

@torch.no_grad()
def evaluate(model, loader, device, lpips_fn, max_batches=None):
    from src.eval_utils import stratified_ssim
    model.eval()
    acc = {"psnr": 0.0, "ssim": 0.0, "ssim_edge": 0.0, "ssim_flat": 0.0, "lpips": 0.0}
    n = 0
    for bi, batch in enumerate(tqdm(loader, desc="val", leave=False)):
        if max_batches and bi >= max_batches:
            break
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        pred = model(lr).clamp(0.0, 1.0)
        mse = F.mse_loss(pred, hr, reduction="none").mean(dim=(1, 2, 3))
        acc["psnr"] += float((10 * torch.log10(1.0 / mse.clamp_min(1e-12))).sum())
        acc["lpips"] += float(lpips_fn(pred.repeat(1, 3, 1, 1) * 2 - 1,
                                       hr.repeat(1, 3, 1, 1) * 2 - 1).sum())
        p_np, h_np = pred.cpu().numpy(), hr.cpu().numpy()
        for i in range(p_np.shape[0]):
            s = stratified_ssim(p_np[i, 0], h_np[i, 0])
            acc["ssim"] += s["ssim"]
            acc["ssim_edge"] += s["ssim_edge"]
            acc["ssim_flat"] += s["ssim_flat"]
        n += lr.shape[0]
    return {k: v / max(n, 1) for k, v in acc.items()}

def build_datasets(cfg):
    sp = load_splits()
    cache_dir = cfg.get_path("cache.dir", "/kaggle/working/cache")
    bank = cfg.get_path("degrade.residual_bank", "artifacts/residual_bank.npz")
    dcfg = degrade_cfg_from_stats(width=cfg.get_path("degrade.width", 1.0),
                                  jitter=cfg.get_path("degrade.jitter", 0.30),
                                  residual_bank=bank)

    train_ds = RestorationDataset(
        cache_dir, stems=sp["train"], gt_only_stems=sp.get("train_gt_only"),
        lr_patch=cfg.get_path("dataset.lr_patch", 64), scale=cfg.get_path("dataset.scale", 2),
        grad_thresh=cfg.get_path("dataset.grad_thresh", 0.0) or 0.0,
        crop_tries=cfg.get_path("dataset.crop_tries", 8), real_frac=cfg.get_path("dataset.real_frac"),
        degrade_cfg=dcfg, jitter_range=tuple(cfg.get_path("augment.scale_jitter", [0.7, 1.4])),
        seed=cfg.get_path("train.seed", 1337))

    val_ds = RestorationDataset(cache_dir, stems=sp["val_ood"], train=False, scale=cfg.get_path("dataset.scale", 2))
    val_id = RestorationDataset(cache_dir, stems=sp["val_id"], train=False, scale=cfg.get_path("dataset.scale", 2))
    return train_ds, val_ds, val_id, sp

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main() -> int:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.get_path("train.seed", 1337))

    out_dir = Path(cfg.get_path("train.output_dir", "artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, val_id_ds, sp = build_datasets(cfg)
    bs = cfg.get_path("train.batch_size", 32)
    nw = cfg.get_path("train.num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              pin_memory=True, drop_last=True, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.get_path("train.val_batch_size", 8),
                            shuffle=False, num_workers=nw, pin_memory=True)

    model = build_model(cfg.get_path("model.name", "nafnet"), scale=cfg.get_path("dataset.scale", 2),
                        dim=cfg.get_path("model.dim", 64), levels=cfg.get_path("model.levels", 1),
                        blocks=cfg.get_path("model.blocks", 2), middle_blocks=cfg.get_path("model.middle_blocks", 2)).to(device)

    epochs = args.epochs or cfg.get_path("train.epochs", 80)
    lr0 = cfg.get_path("train.lr", 5e-4)
    opt = optim.AdamW(model.parameters(), lr=lr0, weight_decay=cfg.get_path("train.weight_decay", 1e-4))
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    amp = bool(cfg.get_path("train.amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    charb = CharbonnierLoss().to(device)
    sobel = SobelEdgeLoss().to(device)
    spectral = SpectralLoss(cfg.get_path("loss.spectral_hi_from", 0.25)).to(device)
    lpips_fn = lpips.LPIPS(net=cfg.get_path("train.lpips_net", "vgg")).to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    w_char = cfg.get_path("loss.charbonnier", 1.0)
    w_lpips = cfg.get_path("loss.lpips", 0.05)
    w_edge = cfg.get_path("loss.edge", 0.5)
    w_spec = cfg.get_path("loss.spectral", 0.0)
    edge_alpha = cfg.get_path("loss.edge_weight_alpha", 4.0)
    clip = cfg.get_path("train.grad_clip", 1.0)
    cutblur_p = cfg.get_path("augment.cutblur_p", 0.5)
    use_d4 = bool(cfg.get_path("augment.d4", True))
    w_lo, w_hi = cfg.get_path("degrade.curriculum", [0.3, 1.0])

    start_ep, best = 1, -1.0
    
    # Initialize SWA Model
    swa_model = AveragedModel(model)
    swa_start = 36 # Start averaging weights during the plateau phase

    history = []
    print(f"\n--- training {epochs} epochs on {device} (amp={amp}) ---")
    for ep in range(start_ep, epochs + 1):
        w = w_lo + (w_hi - w_lo) * (ep - 1) / max(1, epochs - 1)
        train_ds.set_width(w)

        model.train()
        tot, t0 = 0.0, time.perf_counter()
        pbar = tqdm(train_loader, desc=f"ep {ep}/{epochs}", leave=False)
        for batch in pbar:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)

            if use_d4:
                lr, hr = d4_batch(lr, hr)
            if cutblur_p > 0:
                lr, hr = cutblur_sr(lr, hr, scale=cfg.get_path("dataset.scale", 2), p=cutblur_p)

            opt.zero_grad(set_to_none=True)
            
            # --- Ground Truth Filtering ---
            # Applying a slight blur prevents the model from hallucinating texture to match the GT noise floor.
            hr_smooth = TF.gaussian_blur(hr, kernel_size=5, sigma=1.0)

            with torch.autocast("cuda", enabled=amp):
                pred = model(lr)
                wmap = sobel.edge_weight(hr_smooth, edge_alpha)
                l_char = charb(pred, hr_smooth, wmap)
                l_edge = sobel(pred, hr_smooth)
                l_perc = lpips_fn(pred.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1,
                                  hr_smooth.repeat(1, 3, 1, 1) * 2 - 1).mean()
                l_spec = spectral(pred, hr_smooth) if w_spec > 0 else pred.new_zeros(())
                loss = (w_char * l_char + w_edge * l_edge + w_lpips * l_perc + w_spec * l_spec)

            scaler.scale(loss).backward()
            if clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())

        sched.step()
        train_loss = tot / max(1, len(train_loader))
        m = evaluate(model, val_loader, device, lpips_fn)

        print(f"ep {ep:03d}/{epochs} | loss {train_loss:.4f} | "
              f"OOD psnr {m['psnr']:.2f} ssim {m['ssim']:.4f} "
              f"edge {m['ssim_edge']:.4f} flat {m['ssim_flat']:.4f} "
              f"lpips {m['lpips']:.4f} | {time.perf_counter() - t0:.0f}s")

        history.append({"epoch": ep, "loss": train_loss, "width": w, **m})
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        score = m[cfg.get_path("train.select_metric", "ssim")]
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": ep, "best": best}, out_dir / "best_nafnet.pt")
            print(f"   -> saved (best={best:.4f})")
            
        # --- Update SWA weights ---
        if ep >= swa_start:
            swa_model.update_parameters(model)
            print(f"   -> SWA model updated (epoch {ep})")

    # Save the final SWA model
    swa_path = out_dir / "swa_best_nafnet.pt"
    torch.save({"model": swa_model.module.state_dict(), "config": dict(cfg)}, swa_path)
    print(f"\n✅ Saved SWA checkpoint to: {swa_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())