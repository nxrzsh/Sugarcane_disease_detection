"""
Dataset and Data Preprocessing Utilities
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import config


class SugarcaneDataset(Dataset):
    """Dataset for Sugarcane Disease Detection"""

    def __init__(self, data_dir, transform=None, is_training=True):
        """
        Args:
            data_dir: Directory with class subdirectories
            transform: Albumentations transform pipeline
            is_training: Whether this is training data (for augmentation)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_training = is_training
        self.samples = []
        self.classes = config.CLASSES

        # Load all image paths and labels
        self._load_samples()

    def _load_samples(self):
        """Load all samples from the directory structure"""
        if not os.path.exists(self.data_dir):
            print(f"Warning: Directory {self.data_dir} does not exist")
            return

        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            class_idx = config.CLASS_TO_IDX[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default transformation if none provided
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


def get_train_transforms():
    """Get training data augmentation pipeline"""
    return A.Compose([
        A.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.OneOf([
            A.MotionBlur(p=0.3),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.GaussianBlur(p=0.3),
        ], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Get validation/test data transformation pipeline"""
    return A.Compose([
        A.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_dataloaders(train_dir=None, val_dir=None, batch_size=None):
    """
    Create train and validation dataloaders

    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        batch_size: Batch size for dataloaders

    Returns:
        train_loader, val_loader
    """
    train_dir = train_dir or config.TRAIN_DIR
    val_dir = val_dir or config.VAL_DIR
    batch_size = batch_size or config.BATCH_SIZE

    # Create datasets
    train_dataset = SugarcaneDataset(
        train_dir,
        transform=get_train_transforms(),
        is_training=True
    )

    val_dataset = SugarcaneDataset(
        val_dir,
        transform=get_val_transforms(),
        is_training=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader


def get_inference_transforms():
    """Get transformation pipeline for inference"""
    return A.Compose([
        A.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
