"""
Tests for src/pose_extraction.py. Builds a tiny throwaway video on the fly instead of needing a
real signing clip checked into the repo, just there to prove the shapes come out right and that
frames with no person in them get zero-filled instead of crashing or being skipped.
"""

import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pose_extraction import FRAME_VECTOR_SIZE, extract_keypoints  # noqa: E402


@pytest.fixture
def blank_video(tmp_path):
    """A few frames of plain grey, nothing recognisable as a person in it."""
    video_path = tmp_path / "blank.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64)
    )
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    for _ in range(5):
        writer.write(frame)
    writer.release()
    return video_path


def test_extract_keypoints_shape(blank_video):
    keypoints = extract_keypoints(blank_video)
    assert keypoints.shape == (5, FRAME_VECTOR_SIZE)
    assert keypoints.dtype == np.float32


def test_extract_keypoints_zero_fills_when_nothing_detected(blank_video):
    # a plain grey frame has no person in it, so every value should stay at the zero-fill default
    keypoints = extract_keypoints(blank_video)
    assert np.all(keypoints == 0)


def test_extract_keypoints_empty_video_returns_empty_array(tmp_path):
    keypoints = extract_keypoints(tmp_path / "does_not_exist.mp4")
    assert keypoints.shape == (0, FRAME_VECTOR_SIZE)


def test_extract_keypoints_respects_max_frames(blank_video):
    keypoints = extract_keypoints(blank_video, max_frames=2)
    assert keypoints.shape == (2, FRAME_VECTOR_SIZE)
