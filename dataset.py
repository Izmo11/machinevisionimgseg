import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_image(path, size, mode="RGB"):
    if mode == "mask":
        img = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
        return np.array(img, dtype=np.int64)
    img = Image.open(path).convert(mode).resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[..., None] if arr.ndim == 2 else arr


class SegmentationDataset(Dataset):
    def __init__(self, pairs, img_size=256, in_channels=3, num_classes=1, augment=False):
        self.pairs = pairs  # list of (image_path, mask_path)
        self.size = img_size
        self.mode = "RGB" if in_channels == 3 else "L"
        self.num_classes = num_classes
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _augment(self, img, mask):
        if np.random.rand() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if np.random.rand() < 0.5:
            img, mask = img[::-1], mask[::-1]
        k = np.random.randint(4)
        img, mask = np.rot90(img, k), np.rot90(mask, k)
        img = np.clip(img * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.05, 0.05), 0, 1)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)

    def __getitem__(self, i):
        img_path, mask_path = self.pairs[i]
        img = read_image(img_path, self.size, self.mode)
        if mask_path is None:                       
            mask = np.zeros((self.size, self.size), dtype=np.int64)
        else:
            mask = read_image(mask_path, self.size, "mask")
            if self.num_classes == 1:
                # works for both {0,1} and {0,255} masks
                thr = 0 if mask.max() <= 1 else 127
                mask = (mask > thr).astype(np.int64)
        if self.augment:
            img, mask = self._augment(img, mask)
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        return img, torch.from_numpy(mask.astype(np.int64)), img_path