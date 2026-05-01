"""
config.py — All hyperparameters and configuration live here.
Owner: All | Tech: Python dict

Rule: NEVER hardcode learning rate, batch size, or paths inside other scripts.
Everything goes here.

Current dataset: Kaggle phishing screenshots (2 classes)
  0 = legitimate  (genuine_site_0 folder)
  1 = phishing    (phishing_site_1 folder)
  2 = clickbait   (reserved — add self-collected data later)
"""

CONFIG = {
    # Model
    "backbone":     "resnet50",   # Options: "resnet50", "vit_base_patch16_224"
    "num_classes":  2,            # Currently 2: legitimate + phishing
                                  # Change to 3 when clickbait class is added

    # Training
    "epochs":       1,
    "batch_size":   32,
    "lr":           1e-4,
    "weight_decay": 1e-4,
    "seed":         42,

    # Image
    "img_size":     224,

    # Data paths
    "data_root":    "data/processed",
    "checkpoint_dir": "models/checkpoints",

    # Freeze backbone for first N epochs (fine-tuning strategy)
    "freeze_backbone_epochs": 1,

    # Logging
    "log_interval": 5,   # Print every N batches
}

CLASS_NAMES = ["legitimate", "phishing", "clickbait"]
