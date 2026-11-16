"""evaluate.py - Final evaluation on the test images.

* test images WITH masks    -> metrics (Dice, IoU, ...) + per-image CSV
* test images WITHOUT masks -> predicted masks are saved (no score possible)

Usage:  python evaluate.py --output_dir outputs
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from args import get_args
from dataset import SegmentationDataset
from model import UNet
from utils import get_device, predict, seg_metrics


def plot_curves(output_dir):
    path = os.path.join(output_dir, "history.json")
    if not os.path.exists(path):
        return
    h = json.load(open(path))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["train_loss"], label="train"); ax[0].plot(h["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h["train_dice"], label="train"); ax[1].plot(h["val_dice"], label="val")
    ax[1].set_title("Dice"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150); plt.close(fig)


def plot_predictions(samples, output_dir):
    n = len(samples)
    fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for r, (img, gt, pr, name, dice) in enumerate(samples):
        ax[r, 0].imshow(img, cmap="gray" if img.ndim == 2 else None); ax[r, 0].set_title(name, fontsize=8)
        if gt is None:
            ax[r, 1].text(0.5, 0.5, "no ground-truth mask", ha="center", va="center"); ax[r, 1].set_xlim(0, 1)
            ax[r, 1].set_ylim(0, 1)
            ax[r, 2].set_title("Prediction", fontsize=8)
        else:
            ax[r, 1].imshow(gt, cmap="gray"); ax[r, 1].set_title("Ground truth", fontsize=8)
            ax[r, 2].set_title(f"Prediction (Dice {dice:.3f})", fontsize=8)
        ax[r, 2].imshow(pr, cmap="gray")
        for a in ax[r]: a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, "test_predictions.png"), dpi=120); plt.close(fig)


def main():
    args = get_args()
    device = get_device()
    ckpt_path = args.checkpoint or os.path.join(args.output_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt["args"]  # use the settings the model was trained with

    model = UNet(a["in_channels"], a["num_classes"], a["base_channels"], a["dropout"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    split = json.load(open(os.path.join(args.output_dir, "split.json")))
    test_pairs = split["test"]
    ds = SegmentationDataset(test_pairs, a["img_size"], a["in_channels"], a["num_classes"], augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    pred_dir = os.path.join(args.output_dir, "test_predictions")
    os.makedirs(pred_dir, exist_ok=True)
    keys = ["dice", "iou", "precision", "recall", "pixel_acc"]
    rows, samples, k = [], [], 0
    with torch.no_grad():
        for images, masks, names in loader:
            logits = model(images.to(device))
            logits = logits.cpu()
            pred = predict(logits, a["num_classes"], args.threshold)
            m = seg_metrics(pred, masks, a["num_classes"])
            for j in range(images.size(0)):
                labeled = test_pairs[k + j][1] is not None
                name = os.path.basename(names[j])
                arr = pred[j].numpy().astype(np.uint8)
                full = arr
                if args.pred_size == "original":    # upsample the logits to the original image size
                    w, h = Image.open(names[j]).size
                    up = F.interpolate(logits[j:j + 1], size=(h, w), mode="bilinear", align_corners=False)
                    full = predict(up, a["num_classes"], args.threshold)[0].numpy().astype(np.uint8)
                Image.fromarray(full * 255 if a["num_classes"] == 1 else full).save(
                    os.path.join(pred_dir, os.path.splitext(name)[0] + ".png"))
                rows.append({"image": name, "labeled": labeled,
                             **{key: float(m[key][j]) if labeled else "" for key in keys}})
                if len(samples) < args.n_vis:
                    img = (images[j] * 0.5 + 0.5).permute(1, 2, 0).numpy().squeeze()
                    samples.append((img, masks[j].numpy() if labeled else None, arr, name,
                                    rows[-1]["dice"] if labeled else 0.0))
            k += images.size(0)

    with open(os.path.join(args.output_dir, "test_per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "labeled"] + keys); w.writeheader(); w.writerows(rows)
    plot_predictions(samples, args.output_dir)
    plot_curves(args.output_dir)

    scored = [r for r in rows if r["labeled"]]
    size_txt = "original image size" if args.pred_size == "original" else f"{a['img_size']}x{a['img_size']}"
    print(f"Saved {len(rows)} predicted masks to {pred_dir}/ ({size_txt}; binary masks are 0/255)")
    if scored:
        summary = {key: {"mean": float(np.mean([r[key] for r in scored])),
                         "std": float(np.std([r[key] for r in scored]))} for key in keys}
        summary["n_scored_images"] = len(scored)
        json.dump(summary, open(os.path.join(args.output_dir, "test_metrics.json"), "w"), indent=2)
        print(f"Test results on {len(scored)} labeled images:")
        for key in keys:
            print(f"  {key:10s} {summary[key]['mean']:.4f} +/- {summary[key]['std']:.4f}")
    else:
        print("The test images have no masks, so no test score can be computed.")
        print(f"Best validation Dice (on labeled data): {ckpt.get('val_dice', float('nan')):.4f}")
    print(f"Plots and tables saved to {args.output_dir}/")


if __name__ == "__main__":
    main()