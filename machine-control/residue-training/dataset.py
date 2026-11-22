import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A

def list_images(img_dir: Path):
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]
    files = []
    for ext in exts:
        files.extend(sorted(Path(img_dir).glob(f"*{ext}")))
    return files

class SimpleSegDataset(Dataset):
    def __init__(self, image_paths, mask_dir, size=256, augment=False):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.size = size
        self.augment = augment

        # 弱化 CLAHE 強化力道，保留更多自然的漸層邊界，避免模型對特徵過於挑剔
        self.clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))

        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-15, 15), p=0.5),
                A.ColorJitter(p=0.3),
                A.GaussianBlur(p=0.2),
                A.GaussNoise(p=0.2),
                A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 15), hole_width_range=(8, 15), p=0.2),
                A.RandomBrightnessContrast(p=0.3),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise RuntimeError(f"Read error: {img_path}")

        # --- 影像預處理：CLAHE 增強對比度 ---
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype("float32")

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).float(), torch.from_numpy(mask).unsqueeze(0).float()