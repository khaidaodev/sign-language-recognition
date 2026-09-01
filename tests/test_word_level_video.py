"""
Tests for the pure logic in src/word_level_video.py (picking which existing words to top up).
Doesn't touch the download/extraction orchestration in main(), that's network + subprocess
heavy and not something worth mocking out here, best_covered_words() is the one bit of real
decision-making logic in this file.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import word_level_video  # noqa: E402


@pytest.fixture
def fake_processed_dir(tmp_path, monkeypatch):
    clips = {"few": 1, "some": 3, "most": 5, "plenty": 6}
    for gloss, count in clips.items():
        gloss_dir = tmp_path / gloss
        gloss_dir.mkdir()
        for i in range(count):
            np.save(gloss_dir / f"{i}.npy", np.zeros((5, 258)))
    monkeypatch.setattr(word_level_video, "PROCESSED_DIR", tmp_path)
    return tmp_path


def test_best_covered_words_ranks_by_clip_count(fake_processed_dir):
    assert word_level_video.best_covered_words(2) == ["plenty", "most"]


def test_best_covered_words_respects_n(fake_processed_dir):
    assert word_level_video.best_covered_words(1) == ["plenty"]


def test_best_covered_words_returns_everything_if_n_is_larger(fake_processed_dir):
    words = word_level_video.best_covered_words(10)
    assert set(words) == {"few", "some", "most", "plenty"}


def test_best_covered_words_empty_when_nothing_processed_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(word_level_video, "PROCESSED_DIR", tmp_path / "does_not_exist")
    assert word_level_video.best_covered_words(10) == []
