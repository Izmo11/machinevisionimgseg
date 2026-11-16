"""args.py - Defines arguments for configuring the pipeline."""
import argparse


def get_args(argv=None):
    p = argparse.ArgumentParser(description="U-Net segmentation pipeline")

    # Directories
    p.add_argument("--image_dir", default="data/xray", help="folder with X-ray images")
    p.add_argument("--mask_dir", default="data/mask", help="folder with masks")
    p.add_argument("--csv_dir", default="data/CSVs", help="folder with train/validation/test CSV files")
    p.add_argument("--train_csv", default="train.csv")
    p.add_argument("--val_csv", default="validation.csv")
    p.add_argument("--test_csv", default="test.csv")
    p.add_argument("--image_col", default=None, help="CSV column with image names (auto-detected if omitted)")
    p.add_argument("--mask_col", default=None,
                   help="CSV column with mask names (auto-detected; if none, mask is looked up by image name)")
    p.add_argument("--output_dir", default="outputs", help="checkpoints, logs, plots go here")

    # Data / split (only used if the CSV split files are not found)
    p.add_argument("--n_train", type=int, default=100, help="fallback random split: train+val images")
    p.add_argument("--n_test", type=int, default=50, help="fallback random split: test images")
    p.add_argument("--val_frac", type=float, default=0.2, help="fallback random split: val fraction of n_train")
    p.add_argument("--img_size", type=int, default=256, help="images are resized to img_size x img_size (multiple of 16)")
    p.add_argument("--in_channels", type=int, default=3, help="3 = RGB, 1 = grayscale")
    p.add_argument("--num_classes", type=int, default=1,
                   help="1 = binary segmentation (mask>0 is foreground); >1 = multi-class, mask pixel values are class ids")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--early_stop_patience", type=int, default=20, help="0 disables early stopping")

    # Fine-tuning parameters
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_patience", type=int, default=5, help="epochs without val improvement before lr is halved")
    p.add_argument("--base_channels", type=int, default=32, help="U-Net width (16/32/64)")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--resume", default=None, help="checkpoint to start from (fine-tuning)")
    p.add_argument("--freeze_encoder", action="store_true", help="freeze encoder when fine-tuning from --resume")

    # Evaluation
    p.add_argument("--checkpoint", default=None, help="model to evaluate (default: <output_dir>/best_model.pt)")
    p.add_argument("--threshold", type=float, default=0.5, help="binary decision threshold")
    p.add_argument("--n_vis", type=int, default=8, help="number of test predictions to visualize")
    p.add_argument("--pred_size", choices=["original", "train"], default="original",
                   help="size of the saved prediction PNGs: original image size, or the training size (img_size)")

    return p.parse_args(argv)