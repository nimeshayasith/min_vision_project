"""
train.py — Phase 6: Training Loop
Owner: All | Tech: PyTorch

Run:
    python -m src.training.train
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
import numpy as np

from src.model.model import ClickbaitDetector
from src.training.config import CONFIG, CLASS_NAMES
from src.training.dataset import ClickbaitDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_weighted_sampler(dataset: ClickbaitDataset) -> WeightedRandomSampler:
    """
    Handle class imbalance via WeightedRandomSampler.
    The Kaggle dataset has far more legitimate than phishing images.
    """
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(labels, minlength=CONFIG["num_classes"])
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[lbl] for lbl in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> tuple:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

        if batch_idx % CONFIG["log_interval"] == 0:
            print(f"    Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader), correct / total


# ─────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────

def train() -> None:
    torch.manual_seed(CONFIG["seed"])
    print(f"[Device] {DEVICE}")
    print(f"[Config] Backbone: {CONFIG['backbone']} | Epochs: {CONFIG['epochs']} | BS: {CONFIG['batch_size']}")

    # Datasets
    data_root = CONFIG["data_root"]
    train_ds = ClickbaitDataset(f"{data_root}/train")
    val_ds   = ClickbaitDataset(f"{data_root}/val")

    if len(train_ds) == 0:
        raise RuntimeError(
            "Training dataset is empty! Run data preparation scripts first:\n"
            "  python scripts/data_cleaning.py ...\n"
            "  python scripts/prepare_labels.py\n"
            "  python scripts/split_dataset.py"
        )

    # Weighted sampler for class imbalance
    sampler = make_weighted_sampler(train_ds)
    _pin = torch.cuda.is_available()   # pin_memory only helps with GPU
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], sampler=sampler,  num_workers=0, pin_memory=_pin)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    # Model
    model = ClickbaitDetector(backbone=CONFIG["backbone"]).to(DEVICE)

    # ── Fine-tuning strategy: freeze backbone for first N epochs ──
    for param in model.extractor.parameters():
        param.requires_grad = False
    print(f"[Frozen] Backbone frozen for first {CONFIG['freeze_backbone_epochs']} epoch(s)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # Checkpoint directory
    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(1, CONFIG["epochs"] + 1):

        # Unfreeze backbone after N epochs
        if epoch == CONFIG["freeze_backbone_epochs"] + 1:
            for param in model.extractor.parameters():
                param.requires_grad = True
            # Re-create optimizer with full model parameters
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=CONFIG["lr"] * 0.1,   # Lower LR for unfrozen backbone
                weight_decay=CONFIG["weight_decay"],
            )
            print(f"[Unfreeze] Backbone unfrozen at epoch {epoch} (LR -> {CONFIG['lr'] * 0.1:.2e})")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc     = validate(model, val_loader, criterion)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{CONFIG['epochs']} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = ckpt_dir / "best_model.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [SAVED] Best model -> {ckpt_path}  (Val Acc: {best_val_acc:.4f})")

    print(f"\n[DONE] Training complete. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
