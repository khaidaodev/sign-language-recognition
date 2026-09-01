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


# left/right landmark pairs in MediaPipe's 33-point pose topology (landmark 0, the nose, has no
# pair, it's on the midline). Needed to mirror a pose correctly, not just flip x, "left shoulder"
# has to become "right shoulder" too, otherwise a flipped skeleton would have its labels crossed.
POSE_LEFT_RIGHT_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def mirror_keypoints(sequence: np.ndarray) -> np.ndarray:
    """Returns the left-right mirrored version of a keypoint sequence: same sign, as if
    performed facing the other way (or by someone who signs with their other hand). This is a
    standard augmentation for pose-based sign recognition, doubling the effective amount of
    training data for free, no new downloads needed, a mirrored real clip is still a completely
    legitimate example of that sign.

    Only makes sense to run this after normalize_keypoints, since mirroring is just flipping the
    sign of every x coordinate, that's only a clean flip-in-place once everything is centered on
    the body (shoulder midpoint at x=0) rather than positioned somewhere arbitrary in the camera
    frame.
    """
    mirrored = sequence.astype(np.float32).copy()
    for frame in mirrored:
        pose = frame[: POSE_LANDMARKS * 4].reshape(POSE_LANDMARKS, 4)
        left_hand = frame[POSE_LANDMARKS * 4 : POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3].reshape(
            HAND_LANDMARKS, 3
        )
        right_hand = frame[POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 :].reshape(HAND_LANDMARKS, 3)

        pose[:, 0] = -pose[:, 0]
        for left_idx, right_idx in POSE_LEFT_RIGHT_PAIRS:
            pose[[left_idx, right_idx]] = pose[[right_idx, left_idx]]

        left_hand[:, 0] = -left_hand[:, 0]
        right_hand[:, 0] = -right_hand[:, 0]

        # left_hand/right_hand are views into frame, not copies, so grab flattened copies of
        # both *before* writing either back, otherwise writing the first one back corrupts the
        # view the second one still needs to read from (they'd both end up holding the same data)
        new_left_hand = right_hand.flatten()
        new_right_hand = left_hand.flatten()

        frame[: POSE_LANDMARKS * 4] = pose.flatten()
        # whichever hand was on the left is now on the right side of the mirrored image, so the
        # two hand blocks swap places, not just their x values
        frame[POSE_LANDMARKS * 4 : POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3] = new_left_hand
        frame[POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 :] = new_right_hand

    return mirrored


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


class MirrorAugmentedDataset(Dataset):
    """Wraps a dataset (meant to be a *training* split, see below) of real clips and doubles it:
    every real clip's index maps to two items here, the original and its left-right mirrored
    version (mirror_keypoints above).

    Only ever wrap a training split with this, never validation. If a mirrored copy of a clip
    ended up in the validation set while training saw the original (or vice versa), that's the
    model getting evaluated on something awfully close to what it trained on, val accuracy would
    look better than the model actually generalizes."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset) * 2

    def __getitem__(self, idx: int):
        real_idx, mirror = divmod(idx, 2)
        sequence, label = self.base_dataset[real_idx]
        if mirror:
            sequence = torch.from_numpy(mirror_keypoints(sequence.numpy())).float()
        return sequence, label


if __name__ == "__main__":
    dataset = KeypointSequenceDataset()
    print(f"{len(dataset)} clips across {dataset.num_classes} words: {sorted(dataset.gloss_to_label)}")
    sequence, label = dataset[0]
    print(
        f"one example: sequence shape {tuple(sequence.shape)}, "
        f"label {label} ({dataset.label_to_gloss[label]})"
    )
