import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms():
    """
    Augmentation pipeline for training only.
    Simulates real-world conditions: low bandwidth, different devices, lighting.
    """
    return A.Compose([
        A.RandomRotate90(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        A.Resize(224, 224),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Validation/Test: only resize and convert. No augmentation."""
    return A.Compose([
        A.Resize(224, 224),
        ToTensorV2(),
    ])
