from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def get_train_transform(image_size=512):
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.6, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.ElasticTransform(alpha=60, sigma=6, p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


def get_val_transform(image_size=512):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


class OrganoidCystDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_paths = sorted(self.masks_dir.glob("*.npy"))
        if not self.mask_paths:
            raise FileNotFoundError(f"No masks found in {masks_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.mask_paths)

    def _find_image(self, stem):
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            candidate = self.images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No image found for {stem} in {self.images_dir}")

    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]
        image_path = self._find_image(mask_path.stem)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        raw_mask = np.load(mask_path)
        organoid, cyst = raw_mask[..., 0] > 0, raw_mask[..., 1] > 0
        label = np.zeros(organoid.shape, dtype=np.uint8)
        label[organoid] = 1
        label[cyst] = 2
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented["image"], augmented["mask"]
        return image, label.long()
