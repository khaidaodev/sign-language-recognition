"""
Tests for src/sequence_dataset.py. Builds a tiny fake data/processed/ folder instead of using
real keypoint data, so these run instantly and don't need any videos to have been downloaded.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sequence_dataset import KeypointSequenceDataset, pad_or_truncate  # noqa: E402


def test_pad_or_truncate_pads_short_sequences():
    short = np.ones((5, 3))
    padded = pad_or_truncate(short, max_frames=10)
    assert padded.shape == (10, 3)
    assert (padded[:5] == 1).all()
    assert (padded[5:] == 0).all()


def test_pad_or_truncate_cuts_long_sequences():
    long_seq = np.arange(20 * 3).reshape(20, 3)
    truncated = pad_or_truncate(long_seq, max_frames=10)
    assert truncated.shape == (10, 3)
    assert (truncated == long_seq[:10]).all()


def test_pad_or_truncate_leaves_exact_length_alone():
    exact = np.ones((10, 3))
    assert pad_or_truncate(exact, max_frames=10).shape == (10, 3)


@pytest.fixture
def fake_processed_dir(tmp_path):
    """A made-up data/processed/ with 2 words, a few clips each, all different lengths."""
    rng = np.random.default_rng(0)
    clips = {
        "book": [7, 12, 5],
        "drink": [9, 30],
    }
    for gloss, lengths in clips.items():
        gloss_dir = tmp_path / gloss
        gloss_dir.mkdir()
        for i, length in enumerate(lengths):
            np.save(gloss_dir / f"{i}.npy", rng.random((length, 258)))
    return tmp_path


def test_dataset_finds_all_clips(fake_processed_dir):
    dataset = KeypointSequenceDataset(processed_dir=fake_processed_dir)
    assert len(dataset) == 5  # 3 "book" clips + 2 "drink" clips
    assert dataset.num_classes == 2


def test_dataset_every_item_is_the_same_shape_regardless_of_source_length(fake_processed_dir):
    dataset = KeypointSequenceDataset(processed_dir=fake_processed_dir, max_frames=15)
    for i in range(len(dataset)):
        sequence, label = dataset[i]
        assert sequence.shape == (15, 258)
        assert label in (0, 1)


def test_dataset_label_mapping_is_consistent_with_gloss_folder_names(fake_processed_dir):
    dataset = KeypointSequenceDataset(processed_dir=fake_processed_dir)
    assert set(dataset.gloss_to_label) == {"book", "drink"}
    for label, gloss in dataset.label_to_gloss.items():
        assert dataset.gloss_to_label[gloss] == label


def test_dataset_raises_a_clear_error_when_theres_no_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="word_level_video.py"):
        KeypointSequenceDataset(processed_dir=tmp_path)
