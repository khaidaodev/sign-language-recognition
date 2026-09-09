"""
Tests for demo/webcam_demo.py. Only covers the pure, testable pieces: the bounding box maths,
the crop/resize/normalize step, and turning model output into a letter. run_demo() itself needs
a real webcam and a window to draw to, so that's left for a human (with a webcam) to check.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webcam_demo import (  # noqa: E402
    CLASS_LABELS,
    INDEX_TO_LETTER,
    crop_to_model_input,
    hand_bounding_box,
    predict_letter,
)
from cnn_baseline import SmallCNN  # noqa: E402


class FakeLandmark:
    def __init__(self, x, y):
        self.x, self.y = x, y


def test_hand_bounding_box_pads_around_the_landmarks():
    landmarks = [FakeLandmark(0.5, 0.5), FakeLandmark(0.6, 0.4)]
    # frame is 200x100, so landmark pixels span x: 100-120, y: 40-50, +/-10 padding gives a
    # 40x30 box, which then gets squared up (extra padding added to the shorter side, split
    # evenly) to 40x40 before clamping
    box = hand_bounding_box(landmarks, frame_width=200, frame_height=100, padding=10)
    assert box == (90, 25, 130, 65)


def test_hand_bounding_box_is_always_square():
    # a wide, short hand span (fingers spread sideways) shouldn't come out as a wide rectangle,
    # squashing it to 28x28 later would distort the shape compared to the roughly-square
    # training photos
    landmarks = [FakeLandmark(0.2, 0.5), FakeLandmark(0.8, 0.55)]
    left, top, right, bottom = hand_bounding_box(
        landmarks, frame_width=1000, frame_height=1000, padding=20
    )
    assert (right - left) == (bottom - top)


def test_hand_bounding_box_clamps_to_the_frame_edges():
    # landmark right at the top-left corner, padding would push the box off-frame without clamping
    landmarks = [FakeLandmark(0.0, 0.0)]
    box = hand_bounding_box(landmarks, frame_width=200, frame_height=100, padding=40)
    left, top, right, bottom = box
    assert left == 0
    assert top == 0
    assert right <= 200
    assert bottom <= 100


def test_hand_bounding_box_clamps_bottom_right_edge_too():
    landmarks = [FakeLandmark(1.0, 1.0)]
    box = hand_bounding_box(landmarks, frame_width=200, frame_height=100, padding=40)
    left, top, right, bottom = box
    assert right == 200
    assert bottom == 100


def test_crop_to_model_input_shape_and_dtype():
    gray_frame = np.random.randint(0, 256, size=(100, 200), dtype=np.uint8)
    box = (10, 10, 90, 90)

    tensor = crop_to_model_input(gray_frame, box)

    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.dtype == torch.float32


def test_crop_to_model_input_normalizes_pixels_to_zero_one_range():
    gray_frame = np.full((100, 100), 255, dtype=np.uint8)
    box = (0, 0, 100, 100)

    tensor = crop_to_model_input(gray_frame, box)

    assert torch.all(tensor <= 1.0)
    assert torch.all(tensor >= 0.0)
    assert torch.allclose(tensor, torch.ones_like(tensor))


def test_crop_to_model_input_stretches_out_low_contrast_images():
    # a flatly lit crop, two halves only 20 shades apart, standing in for a normal, room-lit
    # webcam frame. the training photos (sign language MNIST) are shot under harsh studio
    # lighting and come out very high contrast, so histogram equalization needs to spread a flat
    # image like this one out towards the full range, not leave it as is. using two solid blocks
    # rather than random noise so the contrast survives the resize down to 28x28 afterwards.
    low_contrast = np.full((100, 100), 110, dtype=np.uint8)
    low_contrast[50:, :] = 130
    box = (0, 0, 100, 100)

    tensor = crop_to_model_input(low_contrast, box)

    assert tensor.min() < 0.3
    assert tensor.max() > 0.7


def test_predict_letter_returns_a_valid_letter_and_confidence():
    # an untrained model won't predict anything meaningful, this just checks the plumbing: the
    # output index gets mapped back to one of the real letters, and confidence is a proper
    # probability, not that it guesses correctly
    model = SmallCNN(num_classes=len(CLASS_LABELS))
    input_tensor = torch.rand(1, 1, 28, 28)

    letter, confidence = predict_letter(model, input_tensor)

    assert letter in INDEX_TO_LETTER.values()
    assert 0.0 <= confidence <= 1.0


def test_predict_letter_never_returns_j_or_z():
    # the dataset has no J or Z (both need motion, see src/data_loading.py), so the label
    # mapping should never be able to produce them regardless of what the model outputs
    model = SmallCNN(num_classes=len(CLASS_LABELS))
    for _ in range(5):
        input_tensor = torch.rand(1, 1, 28, 28)
        letter, _ = predict_letter(model, input_tensor)
        assert letter not in ("J", "Z")


def test_index_to_letter_matches_training_label_ordering():
    # this has to line up exactly with class_labels = sorted(LABEL_TO_LETTER.keys()) in
    # src/cnn_baseline.py's train_and_evaluate(), otherwise every prediction comes out as the
    # wrong letter even though the model itself is fine
    assert CLASS_LABELS == sorted(CLASS_LABELS)
    assert len(INDEX_TO_LETTER) == len(CLASS_LABELS) == 24
