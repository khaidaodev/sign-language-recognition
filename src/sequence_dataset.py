"""
Loads the keypoint sequences that src/pose_extraction.py saves (data/processed/<word>/<video_id>.npy)
into something a PyTorch model can actually train on.

Every clip has a different number of frames (some signs are quicker than others), but a model
needs every example in a batch to be the same shape. So this pads short sequences with zero-rows
and cuts long ones down to a fixed length (MAX_FRAMES below). Zero-padding matches what
pose_extraction.py already does for frames where MediaPipe didn't detect anyone, so a padded
frame and a "nothing detected" frame look the same to the model, that's deliberate, not sloppy.

Run it with:
    python src/sequence_dataset.py     # prints how many clips/words it found in data/processed/
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

MAX_FRAMES = 60  # roughly 2 seconds at 25-30fps, most WLASL clips are shorter than this


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
    is one example of the word "book"."""

    def __init__(self, processed_dir: Path = PROCESSED_DIR, max_frames: int = MAX_FRAMES):
        self.max_frames = max_frames
        self.samples = sorted(Path(processed_dir).glob("*/*.npy"))
        if not self.samples:
            raise FileNotFoundError(
                f"No .npy keypoint files found in {processed_dir}. Run "
                "src/word_level_video.py first to download clips and extract keypoints."
            )

        glosses = sorted({path.parent.name for path in self.samples})
        self.gloss_to_label = {gloss: i for i, gloss in enumerate(glosses)}
        self.label_to_gloss = {i: gloss for gloss, i in self.gloss_to_label.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        sequence = np.load(path)
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
