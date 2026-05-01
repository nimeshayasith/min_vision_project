"""
test_model.py — Phase 4 & 5 Tests
Verifies FeatureExtractor and MCPPipeline output shapes.
"""

import torch
import numpy as np
import pytest
from src.model.feature_extractor import FeatureExtractor, extract_features_from_image
from src.model.mcp_pipeline import MCPPipeline, SelfAttentionLayer
from src.model.model import ClickbaitDetector


# ── Feature Extractor ─────────────────────────────────────────────────────────

def test_feature_extractor_output_shape():
    extractor = FeatureExtractor(model_name="resnet50", pretrained=False)
    dummy_input = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        output = extractor(dummy_input)
    assert output.shape == (1, 2048), f"Expected (1, 2048), got {output.shape}"


def test_feature_extraction_from_numpy():
    extractor = FeatureExtractor(model_name="resnet50", pretrained=False)
    dummy_image = np.zeros((224, 224, 3), dtype=np.float32)
    features = extract_features_from_image(dummy_image, extractor)
    assert features.shape == (2048,)
    assert features.dtype == np.float32


def test_feature_extractor_batch():
    extractor = FeatureExtractor(model_name="resnet50", pretrained=False)
    batch = torch.zeros(4, 3, 224, 224)
    with torch.no_grad():
        out = extractor(batch)
    assert out.shape == (4, 2048)


# ── MCP Pipeline ──────────────────────────────────────────────────────────────

def test_self_attention_output_shape():
    attn = SelfAttentionLayer(feature_dim=2048)
    x = torch.ones(2, 2048)
    out = attn(x)
    assert out.shape == (2, 2048)


def test_mcp_pipeline_output_shape():
    mcp = MCPPipeline(input_dim=2048, hidden_dim=512)
    x = torch.ones(2, 2048)
    out = mcp(x)
    assert out.shape == (2, 256), f"Expected (2, 256), got {out.shape}"


# ── Full Model ────────────────────────────────────────────────────────────────

def test_full_model_forward():
    model = ClickbaitDetector(backbone="resnet50")
    # Temporarily set to 2 classes (matching Kaggle dataset)
    dummy_input = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape[0] == 2
    assert logits.shape[1] >= 2   # At least 2 output classes


def test_full_model_predict():
    model = ClickbaitDetector(backbone="resnet50")
    dummy_input = torch.zeros(1, 3, 224, 224)
    predicted, probs = model.predict(dummy_input)
    assert predicted.shape == (1,)
    assert probs.shape[1] >= 2
    assert abs(probs.sum().item() - 1.0) < 1e-4  # Probabilities sum to 1
