"""
feature_extractor.py — Phase 4: Visual Feature Extraction
Owner: Koshala | Tech: PyTorch, timm (ResNet-50 / ViT)
"""

import torch
import torch.nn as nn
import timm
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor(nn.Module):
    """
    Wraps a pre-trained ResNet-50 or ViT and removes the final
    classification head to output a raw feature vector.

    Output feature sizes:
      - resnet50:               2048-dimensional vector
      - vit_base_patch16_224:   768-dimensional vector
    """

    def __init__(self, model_name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        # Load model without classification head (num_classes=0)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,     # Remove classifier head
            global_pool="avg"  # Global average pooling
        )
        self.feature_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 3, 224, 224) — normalized images
        Returns:
            features: Tensor of shape (batch, feature_dim)
        """
        return self.backbone(x)


def extract_features_from_image(
    image_array: np.ndarray,
    extractor: FeatureExtractor,
) -> np.ndarray:
    """
    Helper to extract features from a single preprocessed image.

    Args:
        image_array: float32 NumPy array (224, 224, 3) — normalized
        extractor:   FeatureExtractor model

    Returns:
        Feature vector as NumPy array of shape (feature_dim,)
    """
    # HWC → CHW, add batch dimension
    tensor = (
        torch.from_numpy(image_array.transpose(2, 0, 1))
        .unsqueeze(0)
        .to(DEVICE)
    )
    extractor.eval()
    with torch.no_grad():
        features = extractor(tensor)
    return features.cpu().numpy().squeeze()
