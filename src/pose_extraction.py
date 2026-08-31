"""
Turns a video of someone signing into a sequence of body/hand keypoints instead of raw pixels.

Why keypoints instead of raw frames: a video is a lot of high-dimensional, noisy pixel data
(background, lighting, clothes, skin tone) that a sign classifier doesn't actually need.
MediaPipe Holistic finds the (x, y, z) position of body joints and finger joints in every frame,
that's the only signal that actually matters for recognising a sign, so we can throw the raw
video away straight after and just keep the numbers. Much smaller, and a sequence model trains
faster and generalises better on this than on pixels.

Per frame we keep:
    - 33 pose landmarks x (x, y, z, visibility) = 132 values
    - 21 left-hand landmarks x (x, y, z)        =  63 values
    - 21 right-hand landmarks x (x, y, z)       =  63 values
    total: 258 values per frame

If a hand (or the whole person) isn't detected in a frame, that section gets filled with zeros
rather than the frame being skipped, keeps every clip's sequence the same shape frame-for-frame,
which the sequence model needs.

Run it on a single video with:
    python src/pose_extraction.py data/raw/videos/book/69241.mp4
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
FRAME_VECTOR_SIZE = POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 * 2  # 258


def _pose_to_array(landmarks) -> np.ndarray:
    if landmarks is None:
        return np.zeros(POSE_LANDMARKS * 4, dtype=np.float32)
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks.landmark],
        dtype=np.float32,
    ).flatten()


def _hand_to_array(landmarks) -> np.ndarray:
    if landmarks is None:
        return np.zeros(HAND_LANDMARKS * 3, dtype=np.float32)
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32
    ).flatten()


def extract_keypoints(video_path, max_frames: int | None = None) -> np.ndarray:
    """Reads a video and returns a (num_frames, 258) array of pose + hand keypoints, one row
    per frame."""
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(str(video_path))
    frame_vectors = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)
            vector = np.concatenate(
                [
                    _pose_to_array(result.pose_landmarks),
                    _hand_to_array(result.left_hand_landmarks),
                    _hand_to_array(result.right_hand_landmarks),
                ]
            )
            frame_vectors.append(vector)
            if max_frames and len(frame_vectors) >= max_frames:
                break
    finally:
        cap.release()
        holistic.close()

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
        print("usage: python src/pose_extraction.py <video path>")
        raise SystemExit(1)
    saved_path, shape = process_video_file(sys.argv[1])
    print(f"saved {shape[0]} frames x {shape[1]} keypoint values to {saved_path}")
