"""
test_evaluation.py — Phase 8 Tests
Verifies IoU computation and evaluation utilities.
"""

import pytest
import numpy as np
from src.evaluation.evaluate import compute_iou


def test_iou_perfect_overlap():
    box = [0, 0, 100, 100]
    assert compute_iou(box, box) == pytest.approx(1.0)


def test_iou_no_overlap():
    box_a = [0,   0,  50,  50]
    box_b = [100, 100, 150, 150]
    assert compute_iou(box_a, box_b) == pytest.approx(0.0)


def test_iou_threshold():
    """IoU > 0.5 should be a successful detection (COCO standard)."""
    box_pred = [0,  0,  100, 100]
    box_true = [10, 10, 110, 110]
    iou = compute_iou(box_pred, box_true)
    assert iou > 0.5, f"Expected IoU > 0.5, got {iou:.4f}"


def test_iou_partial_overlap():
    box_pred = [0,  0,  100, 100]
    box_true = [50, 50, 150, 150]
    iou = compute_iou(box_pred, box_true)
    assert 0.0 < iou < 1.0


def test_iou_zero_area_box():
    """Degenerate boxes should not crash."""
    box_a = [10, 10, 10, 10]   # zero area
    box_b = [0,  0,  50, 50]
    iou = compute_iou(box_a, box_b)
    assert iou == pytest.approx(0.0)


def test_iou_symmetry():
    box_a = [0, 0, 80, 80]
    box_b = [20, 20, 100, 100]
    assert compute_iou(box_a, box_b) == pytest.approx(compute_iou(box_b, box_a))
