"""
Tests for src/pose_extraction.py.

These use fake pose/hand landmarker objects instead of the real MediaPipe Tasks models, so the
tests run instantly and don't need the model files downloaded (that happens once, for real, the
first time someone runs this on an actual video). The fakes just need to look like the real
PoseLandmarker/HandLandmarker: something with a `detect_for_video(image, timestamp_ms)` method
that hands back an object with the same fields MediaPipe's real result objects have
(`pose_landmarks`, `hand_landmarks`, `handedness`), which is exactly what extract_keypoints()
reads off them.
"""

import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pose_extraction import (  # noqa: E402
    FRAME_VECTOR_SIZE,
    HAND_LANDMARKS,
    POSE_LANDMARKS,
    _hands_to_arrays,
    _pose_to_array,
    extract_keypoints,
)


class FakeLandmark:
    def __init__(self, x, y, z, visibility=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class FakeCategory:
    def __init__(self, category_name):
        self.category_name = category_name


class FakePoseResult:
    def __init__(self, pose_landmarks=None):
        # a list with one list of landmarks per detected person, empty list = nobody detected
        self.pose_landmarks = pose_landmarks or []


class FakeHandResult:
    def __init__(self, handedness=None, hand_landmarks=None):
        self.handedness = handedness or []
        self.hand_landmarks = hand_landmarks or []


class FakeLandmarker:
    """Stands in for both PoseLandmarker and HandLandmarker, just returns whatever result it
    was built with on every call, and records whether it got closed."""

    def __init__(self, result):
        self.result = result
        self.closed = False

    def detect_for_video(self, image, timestamp_ms):
        return self.result

    def close(self):
        self.closed = True


def _a_detected_pose():
    return FakePoseResult(pose_landmarks=[[FakeLandmark(0.1, 0.2, 0.3) for _ in range(POSE_LANDMARKS)]])


def _both_hands_detected():
    left = [FakeLandmark(0.4, 0.5, 0.6) for _ in range(HAND_LANDMARKS)]
    right = [FakeLandmark(0.7, 0.8, 0.9) for _ in range(HAND_LANDMARKS)]
    return FakeHandResult(
        handedness=[[FakeCategory("Left")], [FakeCategory("Right")]],
        hand_landmarks=[left, right],
    )


@pytest.fixture
def blank_video(tmp_path):
    """A few frames of plain grey, content doesn't matter, the fake landmarkers below decide
    what "gets detected", not MediaPipe actually looking at the pixels."""
    video_path = tmp_path / "blank.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64)
    )
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    for _ in range(5):
        writer.write(frame)
    writer.release()
    return video_path


def test_extract_keypoints_shape_and_dtype(blank_video):
    pose_landmarker = FakeLandmarker(_a_detected_pose())
    hand_landmarker = FakeLandmarker(_both_hands_detected())

    keypoints = extract_keypoints(
        blank_video, pose_landmarker=pose_landmarker, hand_landmarker=hand_landmarker
    )
    assert keypoints.shape == (5, FRAME_VECTOR_SIZE)
    assert keypoints.dtype == np.float32


def test_extract_keypoints_zero_fills_when_nothing_detected(blank_video):
    pose_landmarker = FakeLandmarker(FakePoseResult())
    hand_landmarker = FakeLandmarker(FakeHandResult())

    keypoints = extract_keypoints(
        blank_video, pose_landmarker=pose_landmarker, hand_landmarker=hand_landmarker
    )
    assert np.all(keypoints == 0)


def test_extract_keypoints_empty_video_returns_empty_array(tmp_path):
    pose_landmarker = FakeLandmarker(_a_detected_pose())
    hand_landmarker = FakeLandmarker(_both_hands_detected())

    keypoints = extract_keypoints(
        tmp_path / "does_not_exist.mp4",
        pose_landmarker=pose_landmarker,
        hand_landmarker=hand_landmarker,
    )
    assert keypoints.shape == (0, FRAME_VECTOR_SIZE)


def test_extract_keypoints_respects_max_frames(blank_video):
    pose_landmarker = FakeLandmarker(_a_detected_pose())
    hand_landmarker = FakeLandmarker(_both_hands_detected())

    keypoints = extract_keypoints(
        blank_video,
        max_frames=2,
        pose_landmarker=pose_landmarker,
        hand_landmarker=hand_landmarker,
    )
    assert keypoints.shape == (2, FRAME_VECTOR_SIZE)


def test_extract_keypoints_does_not_close_landmarkers_it_did_not_create(blank_video):
    # these might be reused across many videos in a batch (see word_level_video.py), closing
    # someone else's landmarker out from under them would break the rest of their batch
    pose_landmarker = FakeLandmarker(_a_detected_pose())
    hand_landmarker = FakeLandmarker(_both_hands_detected())

    extract_keypoints(blank_video, pose_landmarker=pose_landmarker, hand_landmarker=hand_landmarker)

    assert pose_landmarker.closed is False
    assert hand_landmarker.closed is False


def test_pose_to_array_zero_fills_when_no_pose_detected():
    array = _pose_to_array(FakePoseResult())
    assert array.shape == (POSE_LANDMARKS * 4,)
    assert np.all(array == 0)


def test_pose_to_array_reads_out_the_first_detected_person():
    array = _pose_to_array(_a_detected_pose())
    assert array.shape == (POSE_LANDMARKS * 4,)
    # x, y, z, visibility for the first landmark
    assert array[:4].tolist() == pytest.approx([0.1, 0.2, 0.3, 1.0])


def test_hands_to_arrays_sorts_left_and_right_into_the_correct_slots():
    left, right = _hands_to_arrays(_both_hands_detected())
    assert left.shape == (HAND_LANDMARKS * 3,)
    assert right.shape == (HAND_LANDMARKS * 3,)
    assert left[:3].tolist() == pytest.approx([0.4, 0.5, 0.6])
    assert right[:3].tolist() == pytest.approx([0.7, 0.8, 0.9])


def test_hands_to_arrays_zero_fills_a_hand_that_was_not_detected():
    # only a right hand was found this frame
    right_only = [FakeLandmark(0.7, 0.8, 0.9) for _ in range(HAND_LANDMARKS)]
    result = FakeHandResult(handedness=[[FakeCategory("Right")]], hand_landmarks=[right_only])

    left, right = _hands_to_arrays(result)
    assert np.all(left == 0)
    assert right[:3].tolist() == pytest.approx([0.7, 0.8, 0.9])
