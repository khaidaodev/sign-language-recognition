"""
Tests for src/sequence_model.py. Just checks the shapes coming out of the model are right, and
that it doesn't blow up on edge cases like a single-example batch.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sequence_model import SignLSTM  # noqa: E402


def test_output_shape_matches_batch_size_and_num_classes():
    model = SignLSTM(input_size=258, num_classes=10)
    x = torch.randn(4, 60, 258)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_works_with_a_single_example_batch():
    model = SignLSTM(input_size=258, num_classes=10)
    x = torch.randn(1, 60, 258)
    logits = model(x)
    assert logits.shape == (1, 10)


def test_works_with_a_single_lstm_layer():
    # dropout only applies between stacked LSTM layers, so this checks num_layers=1 (no
    # dropout applied inside the LSTM itself) doesn't error out
    model = SignLSTM(input_size=258, num_classes=5, num_layers=1)
    x = torch.randn(2, 30, 258)
    logits = model(x)
    assert logits.shape == (2, 5)


def test_different_sequence_lengths_both_work():
    model = SignLSTM(input_size=258, num_classes=10)
    short = torch.randn(2, 10, 258)
    long_seq = torch.randn(2, 100, 258)
    assert model(short).shape == (2, 10)
    assert model(long_seq).shape == (2, 10)
