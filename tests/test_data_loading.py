"""
Quick test for the label-to-letter mapping in src/data_loading.py, since it's easy to get the
J/Z skip wrong and mislabel a bunch of letters without noticing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loading import LABEL_TO_LETTER  # noqa: E402


def test_no_j_or_z_in_labels():
    letters = set(LABEL_TO_LETTER.values())
    assert "J" not in letters
    assert "Z" not in letters


def test_24_classes():
    assert len(LABEL_TO_LETTER) == 24


def test_label_zero_is_a():
    assert LABEL_TO_LETTER[0] == "A"
