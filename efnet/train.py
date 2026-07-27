"""
efnet/train.py
==============
Training script for EFNet on CASIA-WebFace (folder format).

Fixes vs previous version
──────────────────────────
1. NaN loss (root cause: AMP fp16 + ArcFace scale overflow)
   ▸ Loss now always computed in float32 (handled in losses.py).
   ▸ LR warmup: ramps from 0 → base_lr over warmup_epochs (default 1).
     Without warmup, large random gradients in epoch 1 blow up the model.
   ▸ NaN guard: NaN batches are skipped and logged, never propagated.

2. TensorBoard  — loss/lr per step, LFW accuracy per epoch
   ▸ %tensorboard --logdir /content/tb_logs  in any Colab cell.

3. GDrive checkpoints  — every epoch → MyDrive/EFNet/checkpoints/
"""

import os
import time
import json
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .model    import EFNet
from .losses   import build_loss
from .dataset  import FolderFaceDataset, LFWVerificationDataset
from .dataset  import get_train_transform, get_val_transform
from .evaluate import evaluate_lfw


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Paths
    data_root      : str  = '/content/data/casia-webface'
    lfw_root       : str  = '/content/data/lfw'
    lfw_split      : str  = 'DevTest'
    save_dir       : str  = '/content/checkpoints/efnet'
    gdrive_ckpt_dir: str  = '/content/gdrive/MyDrive/EFNet/checkpoints'
    log_file       : str  = '/content/training_log.json'
    tb_log_dir     : str  = '/content/tb_logs'

    # Model
    embed_dim      : int   = 512
    dropout        : float = 0.0

    # Loss
    loss_type      : str   = 'arcface'
    arcface_s      : float = 64.0
    arcface_m      : float = 0.5

    # Optimiser
    optimizer      : str   = 'sgd'
    lr             : float = 0.1
    momentum       : float = 0.9
    weight_decay   : float = 5e-4
    adamw_lr       : float = 1e-3

    # LR schedule
    warmup_epochs  : int   = 1
    scheduler      : str   = 'step'
    lr_steps       : list  = field(default_factory=lambda: [10, 18, 22])
    lr_gamma       : float = 0.1
    total_epochs   : int   = 25

    # Training
    batch_size     : int   = 128
    num_workers    : int   = 2
    pin_memory     : bool  = True
    use_amp        : bool  = True
    grad_clip      : float = 5.0
    resume_ckpt    : str   = ''

    # Evaluation
    eval_every     : int   = 1
    save_every     : int   = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / (self.count + 1e-8)


def build_optimizer(model: EFNet, criterion: nn.Module,
                    cfg: TrainConfig) -> optim.Optimizer:
    params = list(model.parameters()) + list(criterion.parameters())
    if cfg.optimizer == 'sgd':
        return optim.SGD(params, lr=1e-9,           # actual LR set by scheduler
                         momentum=cfg.momentum,
                         weight_decay=cfg.weight_decay,
                         nesterov=True)
    return optim.AdamW(params, lr=cfg.adamw_lr, weight_decay=cfg.weight_decay)


class WarmupMultiStepLR:
    """
    Per-step scheduler: linear warmup for warmup_epochs, then MultiStep decay.

    Starts at LR=0, ramps linearly to base_lr over warmup_steps.
    After warmup, multiplies LR by gamma each time a milestone (in epochs)
    is reached. Called once per batch (not once per epoch).
    """
    def __init__(self, optimizer, warmup_epochs: int, steps_per_epoch: int,
                 milestones: list, gamma: float, base_lr: float):
        self.optimizer      = optimizer
        self.warmup_steps   = max(warmup_epochs * steps_per_epoch, 1)
        self.milestones_s   = sorted(m * steps_per_epoch for m in milestones)
        self.gamma          = gamma
        self.base_lr        = base_lr
        self.current_step   = 0
        self._update_lr()

    def _current_lr(self) -> float:
        s = self.current_step
        if s < self.warmup_steps:
            return self.base_lr * max(s, 1) / self.warmup_steps
        passed = sum(1 for m in self.milestones_s if s >= m)
        return self.base_lr * (self.gamma ** passed)

    def _update_lr(self):
        lr = self._current_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def step(self):
        self.current_step += 1
        self._update_lr()

    def get_lr(self) -> float:
        return self._current_lr()

    def state_dict(self):
        return {'current_step': self.current_step}

    def load_state_dict(self, d: dict):
        self.current_step = d['current_step']
        self._update_lr()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, criterion, optimizer, scheduler,
                    epoch: int, lfw_acc: float,
                    cfg: TrainConfig, is_best: bool = False):
    state = {
        'epoch'    : epoch,
        'lfw_acc'  : lfw_acc,
        'model'    : model.state_dict(),
        'criterion': criterion.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    fname = f'ckpt_ep{epoch:03d}.pth'

    # Local (fast access within session)
    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.save(state, os.path.join(cfg.save_dir, fname))

    # GDrive (persistent across sessions)
    try:
        os.makedirs(cfg.gdrive_ckpt_dir, exist_ok=True)
        gdrive_path = os.path.join(cfg.gdrive_ckpt_dir, fname)
        torch.save(state, gdrive_path)
        tag = " ★ BEST" if is_best else ""
        print(f"  [Ckpt{tag}] ep{epoch:03d} → {gdrive_path}  LFW={lfw_acc:.4f}")
        if is_best:
            best_path = os.path.join(cfg.gdrive_ckpt_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            print(f"  [Ckpt ★] best_model.pth → {best_path}")
    except Exception as e:
        print(f"  [Ckpt] GDrive save failed: {e}  (local copy kept)")


def load_checkpoint(path: str, model, criterion, optimizer,
                    scheduler, device) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    criterion.load_state_dict(ckpt['criterion'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    print(f"[Resume] '{path}' | epoch={ckpt['epoch']} | LFW={ckpt['lfw_acc']:.4f}")
    return ckpt['epoch'] + 1


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, criterion, loader, optimizer,
                    scaler, scheduler, device, cfg,
                    epoch: int, writer: SummaryWriter,
                    global_step: int) -> tuple:
    model.train()
    criterion.train()

    loss_meter = AverageMeter()
    nan_count  = 0
    t0         = time.time()

    for i, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Backbone in fp16 (AMP); embeddings cast to fp32 inside loss
        with torch.amp.autocast('cuda', enabled=cfg.use_amp):
            embeddings = model(imgs)

        # Loss always fp32 (losses.py casts internally)
        loss = criterion(embeddings, labels)

        if not torch.isfinite(loss):
            nan_count += 1
            if nan_count <= 10:
                print(f"  [NaN] skipping batch {i+1}/{len(loader)} "
                      f"(epoch {epoch}, total skipped: {nan_count})")
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(criterion.parameters()),
            cfg.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1

        loss_meter.update(loss.item(), imgs.size(0))
        lr = scheduler.get_lr()

        writer.add_scalar('train/loss_step', loss.item(), global_step)
        writer.add_scalar('train/lr',        lr,          global_step)

        if (i + 1) % 100 == 0 or (i + 1) == len(loader):
            elapsed = time.time() - t0
            nan_str = f" | ⚠ {nan_count} NaN" if nan_count else ""
            print(f"  Ep {epoch:02d} | {i+1:05d}/{len(loader):05d} | "
                  f"loss {loss_meter.avg:.4f} | lr {lr:.2e} | "
                  f"{elapsed:.0f}s{nan_str}")

    if nan_count:
        pct = 100 * nan_count / len(loader)
        print(f"  ⚠ Epoch {epoch}: {nan_count} NaN batches ({pct:.1f}%). "
              "Should drop to 0 after warmup. "
              "If it persists past epoch 2, lower arcface_s (try 32).")

    return loss_meter.avg, global_step


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  EFNet Training")
    print(f"  Device   : {device}")
    print(f"  AMP      : {cfg.use_amp}")
    print(f"  Epochs   : {cfg.total_epochs}  (warmup: {cfg.warmup_epochs} ep)")
    print(f"  Batch    : {cfg.batch_size}")
    print(f"  GDrive   : {cfg.gdrive_ckpt_dir}")
    print(f"{'='*60}\n")

    # Dataset & loader
    train_ds = FolderFaceDataset(cfg.data_root, transform=get_train_transform())
    num_ids  = train_ds.num_classes
    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        drop_last=True, persistent_workers=(cfg.num_workers > 0)
    )
    steps_per_epoch = len(train_dl)
    print(f"Dataset  : {len(train_ds):,} images | {num_ids:,} ids | "
          f"{steps_per_epoch} steps/epoch")

    # LFW evaluator
    lfw_dl = None
    if os.path.exists(cfg.lfw_root):
        try:
            lfw_ds = LFWVerificationDataset(
                cfg.lfw_root, split=cfg.lfw_split,
                transform=get_val_transform()
            )
            lfw_dl = DataLoader(lfw_ds, batch_size=256, shuffle=False,
                                num_workers=2, pin_memory=True)
            print(f"LFW eval : {len(lfw_ds)} pairs")
        except Exception as e:
            print(f"[Warn] LFW failed: {e}")

    # Model / loss / optimiser / scheduler
    model     = EFNet(embed_dim=cfg.embed_dim, dropout=cfg.dropout).to(device)
    criterion = build_loss(cfg.loss_type, cfg.embed_dim, num_ids,
                           s=cfg.arcface_s, m=cfg.arcface_m).to(device)
    optimizer = build_optimizer(model, criterion, cfg)
    scheduler = WarmupMultiStepLR(
        optimizer,
        warmup_epochs   = cfg.warmup_epochs,
        steps_per_epoch = steps_per_epoch,
        milestones      = cfg.lr_steps,
        gamma           = cfg.lr_gamma,
        base_lr         = cfg.lr,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.use_amp)

    print(f"Params   : {model.count_parameters():,}")
    print(f"Loss     : {cfg.loss_type}  s={cfg.arcface_s}  m={cfg.arcface_m}")
    print(f"LR sched : warmup {cfg.warmup_epochs} ep → "
          f"MultiStep {cfg.lr_steps} ×{cfg.lr_gamma}")

    # TensorBoard
    os.makedirs(cfg.tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.tb_log_dir)
    print(f"\nTensorBoard: paste in a NEW Colab cell and run:")
    print(f"  %load_ext tensorboard")
    print(f"  %tensorboard --logdir {cfg.tb_log_dir}\n")

    # Resume
    start_epoch = 1
    best_lfw    = 0.0
    global_step = 0
    log_records = []

    if cfg.resume_ckpt and os.path.exists(cfg.resume_ckpt):
        start_epoch = load_checkpoint(
            cfg.resume_ckpt, model, criterion, optimizer, scheduler, device
        )
        global_step = (start_epoch - 1) * steps_per_epoch

    # Training loop
    for epoch in range(start_epoch, cfg.total_epochs + 1):
        print(f"\n── Epoch {epoch}/{cfg.total_epochs}  "
              f"LR={scheduler.get_lr():.4e} ─────────────────")

        train_loss, global_step = train_one_epoch(
            model, criterion, train_dl, optimizer,
            scaler, scheduler, device, cfg,
            epoch, writer, global_step
        )

        # Evaluate
        lfw_acc = 0.0
        if lfw_dl is not None and epoch % cfg.eval_every == 0:
            lfw_acc = evaluate_lfw(model, lfw_dl, device)
            writer.add_scalar('eval/lfw_accuracy', lfw_acc, epoch)

        # Checkpoint (every epoch → GDrive)
        is_best = lfw_acc > best_lfw
        if is_best:
            best_lfw = lfw_acc
        if epoch % cfg.save_every == 0:
            save_checkpoint(model, criterion, optimizer, scheduler,
                            epoch, lfw_acc, cfg, is_best)

        # TensorBoard epoch scalars
        writer.add_scalar('epoch/train_loss',   train_loss, epoch)
        writer.add_scalar('epoch/lfw_accuracy', lfw_acc,    epoch)
        writer.add_scalar('epoch/best_lfw',     best_lfw,   epoch)

        # JSON log
        log_records.append({'epoch': epoch, 'loss': round(train_loss, 5),
                            'lfw_acc': round(lfw_acc, 5),
                            'best_lfw': round(best_lfw, 5),
                            'lr': scheduler.get_lr()})
        with open(cfg.log_file, 'w') as f:
            json.dump(log_records, f, indent=2)

    writer.close()
    print(f"\n{'='*60}")
    print(f"  Done.  Best LFW={best_lfw:.4f}")
    print(f"  Model  → {cfg.gdrive_ckpt_dir}/best_model.pth")
    print(f"{'='*60}")
