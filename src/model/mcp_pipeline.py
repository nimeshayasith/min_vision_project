import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    """
    Self-attention to weight different feature dimensions by importance.
    Helps the model focus on features most associated with deceptive elements.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(x)
        return x * attention_weights


class MCPPipeline(nn.Module):
    """
    Multi-Component Reasoning Pipeline.
    Takes raw feature vectors from the extractor and applies:
      1. Self-attention weighting
      2. Two-layer MLP for non-linear reasoning
      3. Dropout regularization
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512):
        super().__init__()
        self.attention = SelfAttentionLayer(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.output_dim = hidden_dim // 2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attended = self.attention(features)
        return self.mlp(attended)


class TemporalAnalyzer(nn.Module):
    """
    (Optional) Analyses a sequence of screenshots from the same website.
    Detects behavioral patterns: frequent pop-ups, aggressive UI changes.

    Input:  Sequence of feature vectors (batch, time_steps, feature_dim)
    Output: Context-aware feature vector (batch, hidden_dim)
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hidden, _) = self.lstm(x)
        return hidden[-1]  # Last layer's hidden state
