"""main.py - Manages data loaders and starts the training.

Usage:  python main.py --image_dir data/xray --mask_dir data/mask --csv_dir data/CSV
"""
import json
import os

from torch.utils.data import DataLoader

from args import get_args
from dataset import SegmentationDataset
from trainer import Trainer
from utils import get_device, list_pairs, load_or_create_split, seed_everything, split_from_csvs


def main():
    args = get_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    split_path = os.path.join(args.output_dir, "split.json")
    split = split_from_csvs(args)
    if split is not None:                       # split defined by your CSV files
        with open(split_path, "w") as f:
            json.dump(split, f, indent=2)
    else:                                       # fallback: random split (100/50 by default)
        print("CSV split files not found -> using a random split")
        pairs = list_pairs(args.image_dir, args.mask_dir)
        split = load_or_create_split(split_path, pairs, args.n_train, args.n_test, args.val_frac, args.seed)
    print(f"Split -> train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])}")

    def make(name, augment):
        return SegmentationDataset(split[name], args.img_size, args.in_channels, args.num_classes, augment)

    train_ds, val_ds = make("train", True), make("val", False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=len(train_ds) > args.batch_size,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")

    best = Trainer(args, train_loader, val_loader, device).fit()
    print(f"\nTraining done. Best val dice: {best:.4f}")
    print(f"Next: python evaluate.py --output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
