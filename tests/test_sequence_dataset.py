"""
Tests for src/sequence_dataset.py. Builds a tiny fake data/processed/ folder instead of using
real keypoint data, so these run instantly and don't need any videos to have been downloaded.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pose_extraction import FRAME_VECTOR_SIZE, HAND_LANDMARKS, POSE_LANDMARKS  # noqa: E402
from sequence_dataset import (  # noqa: E402
    KeypointSequenceDataset,
    MirrorAugmentedDataset,
    mirror_keypoints,
    normalize_keypoints,
    pad_or_truncate,
)


def _blank_frame():
    return np.zeros(FRAME_VECTOR_SIZE, dtype=np.float32)


def _frame_with_pose(left_shoulder, right_shoulder):
    """A single frame with just the two shoulder landmarks set, everything else zero (hands
    "not detected"). Good enough for testing normalize_keypoints without needing every one of
    the 33 pose landmarks filled in."""
    frame = _blank_frame()
    pose = frame[: POSE_LANDMARKS * 4].reshape(POSE_LANDMARKS, 4)
    pose[11, 0], pose[11, 1], pose[11, 3] = left_shoulder[0], left_shoulder[1], 1.0  # visibility
    pose[12, 0], pose[12, 1], pose[12, 3] = right_shoulder[0], right_shoulder[1], 1.0
    frame[: POSE_LANDMARKS * 4] = pose.flatten()
    return frame


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


def test_normalize_keypoints_centers_on_the_shoulder_midpoint():
    # shoulders 0.4 apart horizontally, centered at (0.5, 0.5)
    frame = _frame_with_pose(left_shoulder=(0.3, 0.5), right_shoulder=(0.7, 0.5))
    normalized = normalize_keypoints(np.stack([frame]))[0]
    pose = normalized[: POSE_LANDMARKS * 4].reshape(POSE_LANDMARKS, 4)

    # after centering + scaling by shoulder width (0.4), the shoulders should sit at +/-0.5
    # either side of the origin, symmetric regardless of where they started in the raw frame
    assert pose[11, :2] == pytest.approx([-0.5, 0.0], abs=1e-5)
    assert pose[12, :2] == pytest.approx([0.5, 0.0], abs=1e-5)


def test_normalize_keypoints_is_invariant_to_position_and_distance_from_camera():
    # same sign, performed standing further left and further from the camera (smaller shoulder
    # width) - normalizing should make these land on the same numbers
    near_center = _frame_with_pose(left_shoulder=(0.3, 0.5), right_shoulder=(0.7, 0.5))
    far_left = _frame_with_pose(left_shoulder=(0.1, 0.4), right_shoulder=(0.3, 0.4))

    norm_a = normalize_keypoints(np.stack([near_center]))[0]
    norm_b = normalize_keypoints(np.stack([far_left]))[0]

    pose_a = norm_a[: POSE_LANDMARKS * 4].reshape(POSE_LANDMARKS, 4)
    pose_b = norm_b[: POSE_LANDMARKS * 4].reshape(POSE_LANDMARKS, 4)
    assert pose_a[11, :2] == pytest.approx(pose_b[11, :2].tolist(), abs=1e-5)
    assert pose_a[12, :2] == pytest.approx(pose_b[12, :2].tolist(), abs=1e-5)


def test_normalize_keypoints_leaves_a_frame_with_no_pose_detected_alone():
    frame = _blank_frame()  # nothing detected at all, matches pose_extraction.py's zero-fill
    normalized = normalize_keypoints(np.stack([frame]))[0]
    assert np.all(normalized == 0)


def test_normalize_keypoints_does_not_fake_a_hand_that_was_not_detected():
    frame = _frame_with_pose(left_shoulder=(0.3, 0.5), right_shoulder=(0.7, 0.5))
    normalized = normalize_keypoints(np.stack([frame]))[0]
    left_hand = normalized[POSE_LANDMARKS * 4 : POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3]
    right_hand = normalized[POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 :]
    # both hands were zero (not detected) in the input, normalizing should not turn that into
    # some small nonzero number just because the shoulder center got subtracted
    assert np.all(left_hand == 0)
    assert np.all(right_hand == 0)


def test_normalize_keypoints_leaves_visibility_untouched():
    frame = _frame_with_pose(left_shoulder=(0.3, 0.5), right_shoulder=(0.7, 0.5))
    normalized = normalize_keypoints(np.stack([frame]))[0]
    pose = normalized[: POSE_LANDMARKS * 4].reshape(POSE_LANDMARKS, 4)
    assert pose[11, 3] == pytest.approx(1.0)
    assert pose[12, 3] == pytest.approx(1.0)


def _frame_with_distinct_landmarks():
    """Every pose/hand landmark gets its own unique x value, so a test can check exactly which
    landmark's data ended up where after mirroring, rather than just checking shapes."""
    frame = _blank_frame()
    pose = frame[: POSE_LANDMARKS * 4].reshape(POSE_LANDMARKS, 4)
    for i in range(POSE_LANDMARKS):
        pose[i] = [i * 0.01, 0.5, 0.0, 1.0]
    left_hand = frame[POSE_LANDMARKS * 4 : POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3].reshape(
        HAND_LANDMARKS, 3
    )
    right_hand = frame[POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 :].reshape(HAND_LANDMARKS, 3)
    for i in range(HAND_LANDMARKS):
        left_hand[i] = [i * 0.01 + 0.5, 0.2, 0.0]
        right_hand[i] = [i * 0.01 + 0.7, 0.3, 0.0]
    frame[: POSE_LANDMARKS * 4] = pose.flatten()
    frame[POSE_LANDMARKS * 4 : POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3] = left_hand.flatten()
    frame[POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 :] = right_hand.flatten()
    return frame


def test_mirror_keypoints_negates_x_and_swaps_left_right_pose_landmarks():
    frame = _frame_with_distinct_landmarks()
    mirrored = mirror_keypoints(np.stack([frame]))[0]
    pose = mirrored[: POSE_LANDMARKS * 4].reshape(POSE_LANDMARKS, 4)

    assert pose[0, 0] == pytest.approx(0.0)  # nose has no pair, just gets negated (0 stays 0)
    # left shoulder (11) should now hold right shoulder's (12) old x, negated, and vice versa
    assert pose[11, 0] == pytest.approx(-12 * 0.01)
    assert pose[12, 0] == pytest.approx(-11 * 0.01)
    assert pose[15, 0] == pytest.approx(-16 * 0.01)  # left/right wrist
    # y, z, visibility untouched
    assert pose[11, 1:].tolist() == pytest.approx([0.5, 0.0, 1.0])


def test_mirror_keypoints_swaps_hand_blocks():
    frame = _frame_with_distinct_landmarks()
    mirrored = mirror_keypoints(np.stack([frame]))[0]
    left_hand = mirrored[POSE_LANDMARKS * 4 : POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3].reshape(
        HAND_LANDMARKS, 3
    )
    right_hand = mirrored[POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 :].reshape(HAND_LANDMARKS, 3)

    # the mirrored "left hand" is the old right hand, x negated, y/z unchanged
    assert left_hand[0].tolist() == pytest.approx([-0.7, 0.3, 0.0])
    assert right_hand[0].tolist() == pytest.approx([-0.5, 0.2, 0.0])


def test_mirror_keypoints_is_its_own_inverse():
    # mirroring a mirror should give back the original: negating x twice cancels out, and
    # swapping each left/right pair twice puts everything back where it started
    frame = _frame_with_distinct_landmarks()
    twice_mirrored = mirror_keypoints(mirror_keypoints(np.stack([frame])))[0]
    assert twice_mirrored == pytest.approx(frame)


def test_mirror_keypoints_leaves_a_padded_zero_frame_zero():
    frame = _blank_frame()
    mirrored = mirror_keypoints(np.stack([frame]))[0]
    assert np.all(mirrored == 0)


class _ListDataset:
    """Minimal stand-in for a Dataset, just wraps a plain list of (sequence, label) tuples."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def test_mirror_augmented_dataset_doubles_the_length():
    base = _ListDataset([(torch.zeros(1, FRAME_VECTOR_SIZE), 0), (torch.zeros(1, FRAME_VECTOR_SIZE), 1)])
    augmented = MirrorAugmentedDataset(base)
    assert len(augmented) == 4


def test_mirror_augmented_dataset_even_indices_are_the_originals():
    frame = torch.from_numpy(_frame_with_distinct_landmarks()).float().unsqueeze(0)
    base = _ListDataset([(frame, 7)])
    augmented = MirrorAugmentedDataset(base)

    sequence, label = augmented[0]
    assert label == 7
    assert torch.equal(sequence, frame)


def test_mirror_augmented_dataset_odd_indices_are_mirrored():
    frame_array = _frame_with_distinct_landmarks()
    frame = torch.from_numpy(frame_array).float().unsqueeze(0)
    base = _ListDataset([(frame, 7)])
    augmented = MirrorAugmentedDataset(base)

    sequence, label = augmented[1]
    assert label == 7
    expected = mirror_keypoints(np.stack([frame_array]))
    assert sequence.numpy() == pytest.approx(expected)


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
            np.save(gloss_dir / f"{i}.npy", rng.random((length, FRAME_VECTOR_SIZE)))
    return tmp_path


def test_dataset_finds_all_clips(fake_processed_dir):
    dataset = KeypointSequenceDataset(processed_dir=fake_processed_dir)
    assert len(dataset) == 5  # 3 "book" clips + 2 "drink" clips
    assert dataset.num_classes == 2


def test_dataset_every_item_is_the_same_shape_regardless_of_source_length(fake_processed_dir):
    dataset = KeypointSequenceDataset(processed_dir=fake_processed_dir, max_frames=15)
    for i in range(len(dataset)):
        sequence, label = dataset[i]
        assert sequence.shape == (15, FRAME_VECTOR_SIZE)
        assert label in (0, 1)


def test_dataset_label_mapping_is_consistent_with_gloss_folder_names(fake_processed_dir):
    dataset = KeypointSequenceDataset(processed_dir=fake_processed_dir)
    assert set(dataset.gloss_to_label) == {"book", "drink"}
    for label, gloss in dataset.label_to_gloss.items():
        assert dataset.gloss_to_label[gloss] == label


def test_dataset_raises_a_clear_error_when_theres_no_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="word_level_video.py"):
        KeypointSequenceDataset(processed_dir=tmp_path)


@pytest.fixture
def unevenly_covered_processed_dir(tmp_path):
    """Words with very different amounts of data behind them, like a real WLASL download batch:
    some words only got 1-2 clips downloaded, others got a handful more."""
    rng = np.random.default_rng(0)
    clips = {"few": 1, "some": 3, "most": 5, "plenty": 6}
    for gloss, count in clips.items():
        gloss_dir = tmp_path / gloss
        gloss_dir.mkdir()
        for i in range(count):
            np.save(gloss_dir / f"{i}.npy", rng.random((10, FRAME_VECTOR_SIZE)))
    return tmp_path


def test_max_words_keeps_only_the_best_covered_words(unevenly_covered_processed_dir):
    dataset = KeypointSequenceDataset(processed_dir=unevenly_covered_processed_dir, max_words=2)
    assert set(dataset.gloss_to_label) == {"plenty", "most"}
    assert len(dataset) == 6 + 5


def test_max_words_none_keeps_every_word(unevenly_covered_processed_dir):
    dataset = KeypointSequenceDataset(processed_dir=unevenly_covered_processed_dir, max_words=None)
    assert set(dataset.gloss_to_label) == {"few", "some", "most", "plenty"}
