"""trainer.py - Model initialization, training loop and validation loop."""
import json
import os

import torch

from model import UNet
from utils import AverageMeter, DiceCELoss, predict, seg_metrics


class Trainer:
    def __init__(self, args, train_loader, val_loader, device):
        self.args, self.device = args, device
        self.train_loader, self.val_loader = train_loader, val_loader

        # --- model initialization
        self.model = UNet(args.in_channels, args.num_classes, args.base_channels, args.dropout).to(device)
        if args.resume:
            ckpt = torch.load(args.resume, map_location=device)
            self.model.load_state_dict(ckpt["model"])
            print(f"Loaded weights from {args.resume}")
        if args.freeze_encoder:
            self.model.freeze_encoder()

        self.criterion = DiceCELoss(args.num_classes)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=args.lr_patience)
        self.history = {k: [] for k in ("train_loss", "val_loss", "train_dice", "val_dice", "lr")}

    def _run_epoch(self, loader, train):
        self.model.train(train)
        loss_m, dice_m = AverageMeter(), AverageMeter()
        with torch.set_grad_enabled(train):
            for images, masks, _ in loader:
                images, masks = images.to(self.device), masks.to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                pred = predict(logits, self.args.num_classes)
                dice = seg_metrics(pred, masks, self.args.num_classes)["dice"].mean().item()
                loss_m.update(loss.item(), images.size(0))
                dice_m.update(dice, images.size(0))
        return loss_m.avg, dice_m.avg

    def train_epoch(self):      # training loop
        return self._run_epoch(self.train_loader, train=True)

    def validate_epoch(self):   # validation loop
        return self._run_epoch(self.val_loader, train=False)

    def fit(self):
        a = self.args
        best, bad_epochs = -1.0, 0
        ckpt_path = os.path.join(a.output_dir, "best_model.pt")
        for epoch in range(1, a.epochs + 1):
            tr_loss, tr_dice = self.train_epoch()
            va_loss, va_dice = self.validate_epoch()
            self.scheduler.step(va_dice)
            lr = self.optimizer.param_groups[0]["lr"]
            for k, v in zip(self.history, (tr_loss, va_loss, tr_dice, va_dice, lr)):
                self.history[k].append(v)
            print(f"Epoch {epoch:3d}/{a.epochs} | train loss {tr_loss:.4f} dice {tr_dice:.4f} | "
                  f"val loss {va_loss:.4f} dice {va_dice:.4f} | lr {lr:.2e}")

            if va_dice > best:
                best, bad_epochs = va_dice, 0
                torch.save({"model": self.model.state_dict(), "args": vars(a),
                            "epoch": epoch, "val_dice": va_dice}, ckpt_path)
                print(f"  -> saved best model (val dice {best:.4f})")
            else:
                bad_epochs += 1
                if a.early_stop_patience and bad_epochs >= a.early_stop_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        with open(os.path.join(a.output_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        return best
