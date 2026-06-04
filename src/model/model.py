"""
model.py — Phase 6: Full End-to-End ClickbaitDetector Model
Owner: Koshala + Kumara | Tech: PyTorch

Pipeline: Screenshot → FeatureExtractor → MCPPipeline → ClassifierHead → logits
"""

import torch
import torch.nn as nn
from src.model.feature_extractor import FeatureExtractor
from src.model.mcp_pipeline import MCPPipeline
from src.training.config import CONFIG, CLASS_NAMES

NUM_CLASSES = CONFIG["num_classes"]  # 2 now; 3 when clickbait class is added


class ClickbaitDetector(nn.Module):
    """
    Full end-to-end model:
    Screenshot → Feature Extraction → MCP Reasoning → Classification
    """

    def __init__(self, backbone: str = "resnet50"):
        super().__init__()
        self.extractor = FeatureExtractor(
            model_name=backbone,
            pretrained=CONFIG.get("pretrained", True)
        )
        self.mcp = MCPPipeline(
            input_dim=self.extractor.feature_dim,
            hidden_dim=512,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.mcp.output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, NUM_CLASSES),
            # NOTE: No Softmax — CrossEntropyLoss applies it internally
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Batch of images (batch, 3, 224, 224)
        Returns:
            logits: (batch, NUM_CLASSES)
        """
        features = self.extractor(x)     # (batch, 2048)
        reasoned = self.mcp(features)    # (batch, 256)
        logits   = self.classifier(reasoned)  # (batch, 3)
        return logits

    def predict(self, x: torch.Tensor) -> tuple:
        """
        Returns (predicted_class_indices, probabilities).
        Args:
            x: (batch, 3, 224, 224)
        Returns:
            predicted: (batch,) int tensor
            probs:     (batch, NUM_CLASSES) float tensor
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            predicted = torch.argmax(probs, dim=-1)
        return predicted, probs
