# -*- coding: utf-8 -*-
"""
Generic single-device Trainer for Pascal VOC semantic segmentation.

Checkpoints
-----------
  <checkpoint_dir>/best.pth          – lowest val-loss state
  <checkpoint_dir>/last.pth          – most recent epoch state
  <checkpoint_dir>/training_log.json – per-epoch loss, time, and summary
                                       (read by evaluation/compare.py)
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

IGNORE_INDEX = 255


class Trainer:
    """
    Parameters
    ----------
    model           : nn.Module  (outputs (B, C, H, W) logits)
    train_loader    : DataLoader returning (images, masks)
    val_loader      : DataLoader returning (images, masks)
    optimizer       : any torch optimizer
    loss_fn         : callable(logits, targets) → scalar loss
    device          : torch.device
    checkpoint_dir  : directory to save best.pth / last.pth
    num_epochs      : total training epochs
    scheduler       : optional LR scheduler
    scheduler_mode  : 'epoch'   — scheduler.step() once per epoch
                      'plateau' — scheduler.step(val_loss) once per epoch
    grad_clip       : max gradient norm (0 to disable)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        checkpoint_dir: str,
        num_epochs: int,
        scheduler=None,
        scheduler_mode: str = "epoch",
        grad_clip: float = 1.0,
        start_epoch: int = 1,
    ):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.optimizer      = optimizer
        self.loss_fn        = loss_fn
        self.device         = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_epochs     = num_epochs
        self.start_epoch    = start_epoch
        self.scheduler      = scheduler
        self.scheduler_mode = scheduler_mode
        self.grad_clip      = grad_clip

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self._epoch_log: list = []          # accumulates per-epoch records
        self._train_start: float = 0.0

    # ------------------------------------------------------------------ #
    # Batch utilities
    # ------------------------------------------------------------------ #

    def _prepare_batch(self, images, masks):
        """
        Move batch to device; convert masks from (B,1,H,W) uint8
        to (B,H,W) long with ignore label 255.
        """
        images = images.to(self.device, non_blocking=True)
        masks  = masks.squeeze(1).long().to(self.device, non_blocking=True)
        masks[masks > 20] = IGNORE_INDEX    # clamp out-of-range values
        return images, masks

    def _forward(self, images, masks):
        """Run forward pass; handle HF model outputs and spatial mismatch."""
        logits = self.model(images)

        # HuggingFace models may return an object with a .logits attribute
        if hasattr(logits, "logits"):
            logits = logits.logits

        # Some decoders output at reduced resolution (e.g. SegFormer H/4)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits

    # ------------------------------------------------------------------ #
    # Training / validation loops
    # ------------------------------------------------------------------ #

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = len(self.train_loader)
        log_every  = max(1, n_batches // 5)

        for i, (images, masks) in enumerate(self.train_loader):
            images, masks = self._prepare_batch(images, masks)

            self.optimizer.zero_grad()
            logits = self._forward(images, masks)
            loss   = self.loss_fn(logits, masks)
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_loss += loss.item()

            if (i + 1) % log_every == 0:
                print(f"  [{epoch}] step {i+1}/{n_batches}  loss={loss.item():.4f}")

        return total_loss / n_batches

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0

        for images, masks in self.val_loader:
            images, masks = self._prepare_batch(images, masks)
            logits        = self._forward(images, masks)
            total_loss   += self.loss_fn(logits, masks).item()

        return total_loss / len(self.val_loader)

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        state = {
            "epoch":      epoch,
            "state_dict": self.model.state_dict(),
            "val_loss":   val_loss,
            "optimizer":  self.optimizer.state_dict(),
        }
        torch.save(state, self.checkpoint_dir / "last.pth")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best.pth")
            print(f"  *** new best val_loss={val_loss:.4f} — saved best.pth")

    def save_training_log(self, total_time: float):
        """
        Write training_log.json to the checkpoint directory.

        Schema
        ------
        {
          "total_epochs"     : int,
          "total_time_sec"   : float,
          "avg_epoch_time_sec": float,
          "best_val_loss"    : float,
          "final_train_loss" : float,
          "generalization_gap": float,   // final_val_loss - final_train_loss
          "epochs": [
            {"epoch": 1, "train_loss": x, "val_loss": y, "time_sec": z},
            ...
          ]
        }
        """
        if not self._epoch_log:
            return
        final = self._epoch_log[-1]
        log = {
            "total_epochs":        self.num_epochs,
            "total_time_sec":      round(total_time, 1),
            "avg_epoch_time_sec":  round(total_time / len(self._epoch_log), 1),
            "best_val_loss":       round(self.best_val_loss, 6),
            "final_train_loss":    round(final["train_loss"], 6),
            "final_val_loss":      round(final["val_loss"], 6),
            "generalization_gap":  round(final["val_loss"] - final["train_loss"], 6),
            "epochs":              self._epoch_log,
        }
        path = self.checkpoint_dir / "training_log.json"
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Training log saved -> {path}")

    def _append_time_summary(self, total_time: float):
        """
        Append this run's timing entry to results/training_times.json so that
        training times from multiple models accumulate in one place and can be
        compared without running compare.py.

        Schema: list of {"checkpoint_dir", "total_time_min", "avg_epoch_time_sec",
                          "best_val_loss", "epochs_completed"}
        """
        summary_path = Path("results") / "training_times.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "checkpoint_dir":     str(self.checkpoint_dir),
            "total_time_min":     round(total_time / 60, 1),
            "avg_epoch_time_sec": round(total_time / max(len(self._epoch_log), 1), 1),
            "best_val_loss":      round(self.best_val_loss, 6),
            "epochs_completed":   len(self._epoch_log),
        }

        records = []
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    records = json.load(f)
                # Replace existing entry for the same checkpoint_dir
                records = [r for r in records
                           if r.get("checkpoint_dir") != entry["checkpoint_dir"]]
            except (json.JSONDecodeError, KeyError):
                records = []

        records.append(entry)
        with open(summary_path, "w") as f:
            json.dump(records, f, indent=2)

        # Print a quick cross-model table whenever there are multiple entries
        if len(records) > 1:
            print("\n  -- Cross-model training time summary --")
            print(f"  {'Checkpoint':<35} {'Time(m)':>8} {'AvgEpoch':>9} {'BestVal':>9}")
            print(f"  {'-'*65}")
            for r in records:
                ckpt = r["checkpoint_dir"].replace("checkpoints/", "")
                print(f"  {ckpt:<35} {r['total_time_min']:>8.1f} "
                      f"{r['avg_epoch_time_sec']:>9.1f}s "
                      f"{r['best_val_loss']:>9.4f}")
            print(f"  (full comparison: python evaluation/compare.py ...)\n")

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self):
        if self.start_epoch > self.num_epochs:
            print(f"start_epoch ({self.start_epoch}) > num_epochs ({self.num_epochs}). Nothing to train.")
            return

        if self.start_epoch > 1:
            print(f"\nResuming training from epoch {self.start_epoch} -> {self.num_epochs} on {self.device}")
        else:
            print(f"\nTraining for {self.num_epochs} epochs on {self.device}")
        print(f"Checkpoints -> {self.checkpoint_dir}\n")

        total_start = time.time()

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            t0 = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss   = self.val_epoch(epoch)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, val_loss, is_best)

            if self.scheduler is not None:
                if self.scheduler_mode == "plateau":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            elapsed = time.time() - t0
            self._epoch_log.append({
                "epoch":      epoch,
                "train_loss": round(train_loss, 6),
                "val_loss":   round(val_loss, 6),
                "time_sec":   round(elapsed, 1),
            })

            print(
                f"Epoch {epoch:3d}/{self.num_epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"{'[BEST]' if is_best else '      '}  "
                f"({elapsed:.0f}s)"
            )

        total_time = time.time() - total_start
        self.save_training_log(total_time)
        self._append_time_summary(total_time)

        sep = "=" * 52
        print(f"\n{sep}")
        print("  Training complete")
        print(sep)
        print(f"  Total time      : {total_time / 60:.1f} min  ({total_time:.0f}s)")
        print(f"  Avg epoch time  : {total_time / len(self._epoch_log):.1f}s")
        print(f"  Best val loss   : {self.best_val_loss:.4f}")
        print(f"  Best checkpoint : {self.checkpoint_dir / 'best.pth'}")
        print(f"  Training log    : {self.checkpoint_dir / 'training_log.json'}")
        print(sep)
