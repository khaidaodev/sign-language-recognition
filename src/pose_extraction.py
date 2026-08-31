"""
Turns a video of someone signing into a sequence of body/hand keypoints instead of raw pixels.

Why keypoints instead of raw frames: a video is a lot of high-dimensional, noisy pixel data
(background, lighting, clothes, skin tone) that a sign classifier doesn't actually need.
MediaPipe finds the (x, y, z) position of body joints and finger joints in every frame, that's
the only signal that actually matters for recognising a sign, so we can throw the raw video away
straight after and just keep the numbers. Much smaller, and a sequence model trains faster and
generalises better on this than on pixels.

Per frame we keep:
    - 33 pose landmarks x (x, y, z, visibility) = 132 values
    - 21 left-hand landmarks x (x, y, z)        =  63 values
    - 21 right-hand landmarks x (x, y, z)       =  63 values
    total: 258 values per frame

If a hand (or the whole person) isn't detected in a frame, that section gets filled with zeros
rather than the frame being skipped, keeps every clip's sequence the same shape frame-for-frame,
which the sequence model needs.

This used to be built on mp.solutions.holistic (MediaPipe's old "legacy" API). That API has been
dropped entirely from mediapipe wheels for newer Python versions (0.10.30+, 1.x - anything built
for Python 3.13/3.14), there's no version pin that brings it back. So this now uses MediaPipe's
newer "Tasks" API instead: two separate models, PoseLandmarker and HandLandmarker, run side by
side on every frame. Functionally the same output (same 258-value vector), just a different API
to get there. The two model files (a few MB each) get downloaded once into data/mediapipe_models/
the first time this runs, same idea as wlasl_metadata.py downloading the dataset index.

Run it on a single video with:
    python3 src/pose_extraction.py data/raw/videos/book/69241.mp4
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import requests
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "data" / "mediapipe_models"

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
FRAME_VECTOR_SIZE = POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 * 2  # 258

# "lite" variants: smaller and faster than "full"/"heavy", plenty accurate enough for picking
# out hand/body position, no need for the heaviest model here.
MODEL_URLS = {
    "pose_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


def download_models(force: bool = False) -> None:
    """Grabs the pose + hand model bundle files if we don't already have them."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODEL_URLS.items():
        out_path = MODELS_DIR / filename
        if out_path.exists() and not force:
            continue
        print(f"Downloading {url} ...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)


def _make_pose_landmarker():
    download_models()
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "pose_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _make_hand_landmarker():
    download_models()
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def _pose_to_array(pose_result) -> np.ndarray:
    if not pose_result.pose_landmarks:
        return np.zeros(POSE_LANDMARKS * 4, dtype=np.float32)
    # first (and only, num_poses=1) detected person
    landmarks = pose_result.pose_landmarks[0]
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility or 0.0] for lm in landmarks],
        dtype=np.float32,
    ).flatten()


def _hands_to_arrays(hand_result) -> tuple[np.ndarray, np.ndarray]:
    """Returns (left_hand_array, right_hand_array). The Tasks API doesn't have separate
    "left hand" / "right hand" slots like the old Holistic API did, it detects up to num_hands
    hands and labels each one via `handedness`, so we sort them into the same two slots
    ourselves, zero-filling whichever hand (or both) wasn't found."""
    left = np.zeros(HAND_LANDMARKS * 3, dtype=np.float32)
    right = np.zeros(HAND_LANDMARKS * 3, dtype=np.float32)
    for handedness, landmarks in zip(hand_result.handedness, hand_result.hand_landmarks):
        array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()
        label = handedness[0].category_name
        if label == "Left":
            left = array
        elif label == "Right":
            right = array
    return left, right


def extract_keypoints(
    video_path,
    max_frames: int | None = None,
    pose_landmarker=None,
    hand_landmarker=None,
) -> np.ndarray:
    """Reads a video and returns a (num_frames, 258) array of pose + hand keypoints, one row
    per frame.

    pose_landmarker/hand_landmarker can be passed in directly (mainly so tests can point this at
    fake landmarkers instead of the real models, which need a one-time download), otherwise this
    downloads the real models and builds them itself."""
    owns_landmarkers = pose_landmarker is None
    if owns_landmarkers:
        pose_landmarker = _make_pose_landmarker()
        hand_landmarker = _make_hand_landmarker()

    cap = cv2.VideoCapture(str(video_path))
    frame_vectors = []
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image(image_format=ImageFormat.SRGB, data=rgb)
            # VIDEO running mode just needs strictly increasing timestamps, doesn't need to be
            # real wall-clock time, so the frame index works fine
            pose_result = pose_landmarker.detect_for_video(image, frame_index)
            hand_result = hand_landmarker.detect_for_video(image, frame_index)

            left_hand, right_hand = _hands_to_arrays(hand_result)
            vector = np.concatenate([_pose_to_array(pose_result), left_hand, right_hand])
            frame_vectors.append(vector)

            frame_index += 1
            if max_frames and len(frame_vectors) >= max_frames:
                break
    finally:
        cap.release()
        if owns_landmarkers:
            pose_landmarker.close()
            hand_landmarker.close()

    if not frame_vectors:
        return np.zeros((0, FRAME_VECTOR_SIZE), dtype=np.float32)
    return np.stack(frame_vectors)


def process_video_file(video_path, out_path=None):
    """Extracts keypoints for one video and saves them as a .npy file next to (or wherever
    told) the raw video. Returns (out_path, array_shape)."""
    keypoints = extract_keypoints(video_path)
    if out_path is None:
        out_path = PROCESSED_DIR / Path(video_path).with_suffix(".npy").name
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, keypoints)
    return out_path, keypoints.shape


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 src/pose_extraction.py <video path>")
        raise SystemExit(1)
    saved_path, shape = process_video_file(sys.argv[1])
    print(f"saved {shape[0]} frames x {shape[1]} keypoint values to {saved_path}")
