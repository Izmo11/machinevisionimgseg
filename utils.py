import csv
import io
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_pairs(image_dir, mask_dir):
    masks = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir)
             if os.path.splitext(f)[1].lower() in IMG_EXT}
    pairs, skipped = [], []
    for f in sorted(os.listdir(image_dir), key=lambda n: (len(n), n)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in IMG_EXT:
            continue
        if stem not in masks:
            skipped.append(f)
            continue
        pairs.append((os.path.join(image_dir, f), os.path.join(mask_dir, masks[stem])))
    if skipped:
        print(f"WARNING: {len(skipped)} images have no mask and are skipped "
              f"(first: {skipped[:3]}). Using {len(pairs)} image/mask pairs.")
    return pairs


def make_split(pairs, n_train, n_test, val_frac, seed):
    if len(pairs) < n_train + n_test:
        raise ValueError(f"Need {n_train + n_test} pairs, found {len(pairs)}")
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    train_all, test = idx[:n_train], idx[n_train:n_train + n_test]
    n_val = int(round(len(train_all) * val_frac))
    val, train = train_all[:n_val], train_all[n_val:]
    pick = lambda ids: [list(pairs[i]) for i in ids]
    return {"train": pick(train), "val": pick(val), "test": pick(test)}


def load_or_create_split(path, pairs, n_train, n_test, val_frac, seed):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    split = make_split(pairs, n_train, n_test, val_frac, seed)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
    return split


# ---------------------------------------------------------------- CSV split
def _index(directory):
    return {os.path.splitext(f)[0]: os.path.join(directory, f)
            for f in sorted(os.listdir(directory)) if os.path.splitext(f)[1].lower() in IMG_EXT}


def _stem(value):
    base = os.path.basename(str(value).strip().replace("\\", "/"))
    root, ext = os.path.splitext(base)
    return root if ext.lower() in IMG_EXT else base


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        text = f.read()
    try:
        dialect = csv.Sniffer().sniff(text[:2048], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    header = rows[0]
    if any(os.path.splitext(c.strip())[1].lower() in IMG_EXT for c in header):  # no header row
        return [f"col{i}" for i in range(len(header))], rows
    return [c.strip() for c in header], rows[1:]


def _resolve(value, index, csv_path):
    v = str(value).strip().replace("\\", "/")
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    for cand in (v, os.path.join(csv_dir, v), os.path.join(os.path.dirname(csv_dir), v)):
        if os.path.isfile(cand):
            return cand
    return index.get(_stem(v))


def pairs_from_csv(path, image_index, mask_index, image_col=None, mask_col=None, allow_no_mask=False):
    names, rows = read_csv_rows(path)
    low = [n.lower() for n in names]

    def stem_hits(ci, index):
        vals = [r[ci] for r in rows if ci < len(r)]
        return sum(_stem(v) in index for v in vals) / max(len(vals), 1)

    def file_hits(ci):
        vals = [r[ci] for r in rows if ci < len(r)]
        return sum(os.path.isfile(str(v).strip().replace("\\", "/")) for v in vals) / max(len(vals), 1)

    def hint(words, exclude=None, skip=()):
        for ci, n in enumerate(low):
            if ci != exclude and any(w in n for w in words) and not any(w in n for w in skip):
                return ci
        return None

    if image_col:
        ic = names.index(image_col)
    else:
        ic = hint(("image", "xray", "x-ray", "x_ray", "img"), skip=("mask",))
        if ic is None:
            ic = max(range(len(names)), key=lambda c: stem_hits(c, image_index))
            if stem_hits(ic, image_index) < 0.5 and file_hits(ic) < 0.5:
                raise ValueError(f"{path}: cannot tell which column holds the images. Columns: {names}. "
                                 f"Use --image_col.")
    if mask_col:
        mc = names.index(mask_col)
    else:
        mc = hint(("mask", "seg", "annot"), exclude=ic)
        if mc is None:
            others = [c for c in range(len(names)) if c != ic and (stem_hits(c, mask_index) > 0.5 or file_hits(c) > 0.5)]
            mc = others[0] if others else None
    print(f"  {os.path.basename(path)}: {len(rows)} rows | image column = '{names[ic]}' | "
          f"mask column = {repr(names[mc]) if mc is not None else 'none (matched by image name)'}")

    pairs, missing = [], []
    for r in rows:
        img = _resolve(r[ic], image_index, path)
        if img is None:
            missing.append(r[ic]); continue
        if mc is not None:
            raw = r[mc].strip() if mc < len(r) else ""
            if raw == "" and allow_no_mask:     
                pairs.append([img, None]); continue
            m = _resolve(raw, mask_index, path)
        else:
            stem = _stem(r[ic])
            m = next((mask_index[c] for c in (stem, stem + "_mask", stem + "-mask", "mask_" + stem,
                                              stem.replace("_mask", ""), stem.replace("_xray", "_mask"))
                      if c in mask_index), None)
        if m is None and allow_no_mask:
            pairs.append([img, None]); continue
        if m is None:
            missing.append(f"{r[ic]} -> mask not found" + (f" ({r[mc]})" if mc is not None else ""))
            continue
        pairs.append([img, m])
    if missing:
        raise FileNotFoundError(f"{path}: {len(missing)} of {len(rows)} entries have no matching file, "
                                f"e.g. {missing[:5]}")
    return pairs


def _find_csv(csv_dir, preferred, aliases):
    files = {f.lower(): f for f in os.listdir(csv_dir) if f.lower().endswith(".csv")}
    if preferred.lower() in files:
        return os.path.join(csv_dir, files[preferred.lower()])
    for alias in aliases:
        for low, f in files.items():
            if os.path.splitext(low)[0] == alias:
                return os.path.join(csv_dir, f)
    return None


def split_from_csvs(args):

    if not os.path.isdir(args.csv_dir):
        return None
    paths = {
        "train": _find_csv(args.csv_dir, args.train_csv, ["train", "training"]),
        "val": _find_csv(args.csv_dir, args.val_csv, ["validation", "valid", "val"]),
        "test": _find_csv(args.csv_dir, args.test_csv, ["test", "testing"]),
    }
    missing = [k for k, p in paths.items() if p is None]
    if missing:
        raise FileNotFoundError(
            f"Could not find the CSV for {missing} in '{args.csv_dir}'. Files there: "
            f"{sorted(os.listdir(args.csv_dir))}. Use --train_csv / --val_csv / --test_csv.")
    image_index, mask_index = _index(args.image_dir), _index(args.mask_dir)
    print(f"Found {len(image_index)} images in {args.image_dir} and {len(mask_index)} masks in {args.mask_dir}")
    split = {k: pairs_from_csv(p, image_index, mask_index, args.image_col, args.mask_col,
                               allow_no_mask=(k == "test")) for k, p in paths.items()}
    n_unlabeled = sum(m is None for _, m in split["test"])
    if n_unlabeled:
        print(f"NOTE: {n_unlabeled} of {len(split['test'])} test images have no mask -> test is predict-only for those")
    return split


#  loss
class DiceCELoss(nn.Module):
    """Cross-entropy (BCE for binary) + soft Dice. Targets are (B,H,W) long class ids."""

    def __init__(self, num_classes, eps=1e-6):
        super().__init__()
        self.nc, self.eps = num_classes, eps

    def forward(self, logits, target):
        if self.nc == 1:
            t = target.float().unsqueeze(1)
            ce = F.binary_cross_entropy_with_logits(logits, t)
            p = torch.sigmoid(logits)
        else:
            ce = F.cross_entropy(logits, target)
            p = torch.softmax(logits, 1)
            t = F.one_hot(target, self.nc).permute(0, 3, 1, 2).float()
        inter = (p * t).sum((0, 2, 3))
        den = p.sum((0, 2, 3)) + t.sum((0, 2, 3))
        dice = 1 - ((2 * inter + self.eps) / (den + self.eps)).mean()
        return ce + dice


# metrics
def predict(logits, num_classes, threshold=0.5):
    if num_classes == 1:
        return (torch.sigmoid(logits) > threshold).long().squeeze(1)
    return logits.argmax(1)


def seg_metrics(pred, target, num_classes, eps=1e-7):
    classes = [1] if num_classes == 1 else list(range(1, num_classes))
    d, i, pr, rc = [], [], [], []
    for c in classes:
        p = (pred == c).flatten(1).float()
        t = (target == c).flatten(1).float()
        inter, ps, ts = (p * t).sum(1), p.sum(1), t.sum(1)
        d.append((2 * inter + eps) / (ps + ts + eps))
        i.append((inter + eps) / (ps + ts - inter + eps))
        pr.append((inter + eps) / (ps + eps))
        rc.append((inter + eps) / (ts + eps))
    mean = lambda x: torch.stack(x).mean(0)
    return {"dice": mean(d), "iou": mean(i), "precision": mean(pr), "recall": mean(rc),
            "pixel_acc": (pred == target).flatten(1).float().mean(1)}


class AverageMeter:
    def __init__(self):
        self.sum = self.n = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.n += n

    @property
    def avg(self):
        return self.sum / max(self.n, 1)