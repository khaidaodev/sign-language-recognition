"""
Loads the keypoint sequences that src/pose_extraction.py saves (data/processed/<word>/<video_id>.npy)
into something a PyTorch model can actually train on.

Every clip has a different number of frames (some signs are quicker than others), but a model
needs every example in a batch to be the same shape. So this pads short sequences with zero-rows
and cuts long ones down to a fixed length (MAX_FRAMES below). Zero-padding matches what
pose_extraction.py already does for frames where MediaPipe didn't detect anyone, so a padded
frame and a "nothing detected" frame look the same to the model, that's deliberate, not sloppy.

Two things happen to every sequence before it reaches the model:

1. Normalization (see normalize_keypoints below): raw MediaPipe coordinates are relative to the
   whole camera frame, so "hand at x=0.6" means something different depending on where the
   person is standing and how close they are to the camera. Centering on the shoulders and
   scaling by shoulder width removes that nuisance variation, which matters a lot when there's
   only a handful of real training clips per word, the model shouldn't have to spend that scarce
   data learning "person standing slightly left of frame" is irrelevant.
2. Padding/truncating to MAX_FRAMES.

Run it with:
    python3 src/sequence_dataset.py     # prints how many clips/words it found in data/processed/
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_extraction import HAND_LANDMARKS, POSE_LANDMARKS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

MAX_FRAMES = 60  # roughly 2 seconds at 25-30fps, most WLASL clips are shorter than this

LEFT_SHOULDER = 11  # MediaPipe pose landmark indices, same in both the old and new API
RIGHT_SHOULDER = 12


def normalize_keypoints(sequence: np.ndarray) -> np.ndarray:
    """Centers every frame on the midpoint between the shoulders and scales by shoulder width,
    so the same sign performed in a different spot in frame, or closer/further from the camera,
    produces (roughly) the same numbers. Visibility values are left alone, those already mean
    the same thing regardless of position/scale.

    A frame where no pose was detected (zero-filled, see pose_extraction.py) is left as-is,
    there's no shoulder position to normalize against. Same for a hand that wasn't detected in
    an otherwise-normal frame, normalizing a zero-filled hand would turn it into a small nonzero
    number (0 minus the shoulder center), which would wrongly look like "a hand was detected
    right at the shoulders" to the model.
    """
    sequence = sequence.astype(np.float32).copy()
    for frame in sequence:
        pose = frame[: POSE_LANDMARKS * 4].reshape(POSE_LANDMARKS, 4)
        left_hand = frame[POSE_LANDMARKS * 4 : POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3].reshape(
            HAND_LANDMARKS, 3
        )
        right_hand = frame[POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 :].reshape(HAND_LANDMARKS, 3)

        left_shoulder = pose[LEFT_SHOULDER, :2]
        right_shoulder = pose[RIGHT_SHOULDER, :2]
        if not (left_shoulder.any() or right_shoulder.any()):
            continue  # no pose detected this frame, nothing to normalize against

        center = (left_shoulder + right_shoulder) / 2
        scale = float(np.linalg.norm(left_shoulder - right_shoulder))
        if scale < 1e-6:
            continue

        pose[:, 0] = (pose[:, 0] - center[0]) / scale
        pose[:, 1] = (pose[:, 1] - center[1]) / scale
        pose[:, 2] = pose[:, 2] / scale
        # column 3 (visibility) untouched

        for hand in (left_hand, right_hand):
            if not hand.any():
                continue  # this hand wasn't detected in this frame, leave the zero-fill alone
            hand[:, 0] = (hand[:, 0] - center[0]) / scale
            hand[:, 1] = (hand[:, 1] - center[1]) / scale
            hand[:, 2] = hand[:, 2] / scale

        frame[: POSE_LANDMARKS * 4] = pose.flatten()
        frame[POSE_LANDMARKS * 4 : POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3] = left_hand.flatten()
        frame[POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 :] = right_hand.flatten()

    return sequence


def pad_or_truncate(sequence: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Makes any (num_frames, feature_size) array exactly (max_frames, feature_size): pads with
    zero rows if it's short, cuts off the end if it's long."""
    num_frames, feature_size = sequence.shape
    if num_frames >= max_frames:
        return sequence[:max_frames]
    padding = np.zeros((max_frames - num_frames, feature_size), dtype=sequence.dtype)
    return np.concatenate([sequence, padding], axis=0)


class KeypointSequenceDataset(Dataset):
    """One example = one signed word, as a (MAX_FRAMES, 258) keypoint sequence plus its label.
    The word (gloss) is just read off the parent folder name, e.g. data/processed/book/69241.npy
    is one example of the word "book".

    max_words: with a real, fairly small download batch, some words only have 3-4 example clips
    while others have 8-10, and every extra word makes the classification problem harder (that's
    one more class to tell apart) without necessarily adding much signal if it's only got a
    couple of examples. Setting this keeps just the max_words words with the most example clips,
    so training runs on a vocabulary size that actually matches how much data is behind it. Set
    to None to use every word that has at least one clip, however few.
    """

    def __init__(
        self,
        processed_dir: Path = PROCESSED_DIR,
        max_frames: int = MAX_FRAMES,
        max_words: int | None = None,
    ):
        self.max_frames = max_frames
        samples = sorted(Path(processed_dir).glob("*/*.npy"))
        if not samples:
            raise FileNotFoundError(
                f"No .npy keypoint files found in {processed_dir}. Run "
                "src/word_level_video.py first to download clips and extract keypoints."
            )

        if max_words is not None:
            counts: dict[str, int] = {}
            for path in samples:
                counts[path.parent.name] = counts.get(path.parent.name, 0) + 1
            # most clips first, alphabetical as a tiebreaker so this is reproducible
            kept_glosses = {
                gloss
                for gloss, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_words]
            }
            samples = [path for path in samples if path.parent.name in kept_glosses]

        self.samples = samples
        glosses = sorted({path.parent.name for path in self.samples})
        self.gloss_to_label = {gloss: i for i, gloss in enumerate(glosses)}
        self.label_to_gloss = {i: gloss for gloss, i in self.gloss_to_label.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        sequence = np.load(path)
        sequence = normalize_keypoints(sequence)
        sequence = pad_or_truncate(sequence, self.max_frames)
        label = self.gloss_to_label[path.parent.name]
        return torch.from_numpy(sequence).float(), label

    @property
    def num_classes(self) -> int:
        return len(self.gloss_to_label)


if __name__ == "__main__":
    dataset = KeypointSequenceDataset()
    print(f"{len(dataset)} clips across {dataset.num_classes} words: {sorted(dataset.gloss_to_label)}")
    sequence, label = dataset[0]
    print(
        f"one example: sequence shape {tuple(sequence.shape)}, "
        f"label {label} ({dataset.label_to_gloss[label]})"
    )
