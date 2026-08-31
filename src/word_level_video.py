"""
Stage 2: recognising a signed word from a short video clip, instead of a single letter from a
still photo (that's stage 1, see cnn_baseline.py). Three steps, each in its own file:

    1. wlasl_metadata.py   - download the WLASL dataset index, pick a manageable subset of words
    2. video_downloader.py - download + trim the actual video clips for that subset
    3. pose_extraction.py  - turn each video into a sequence of body/hand keypoints (MediaPipe)

This file just runs all three back to back. The sequence model itself (something like an LSTM
or a small transformer over the keypoint sequences) isn't built yet, that's the next bit once
there's actually keypoint data sitting in data/processed/ to train it on.

A lot of this dataset's video links are old and dead by now (see video_downloader.py for why),
so don't expect every clip to download successfully, that's expected and handled, not a bug.
Whatever does download is enough to get keypoint sequences flowing end to end.

Run it with, e.g.:
    python src/word_level_video.py --num-words 100 --max-per-word 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_extraction import process_video_file  # noqa: E402
from video_downloader import VIDEO_DIR, download_subset  # noqa: E402
from wlasl_metadata import save_subset  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-words", type=int, default=100, help="how many words to include")
    parser.add_argument(
        "--max-per-word", type=int, default=10, help="max video instances to try per word"
    )
    args = parser.parse_args()

    save_subset(num_words=args.num_words, max_instances_per_word=args.max_per_word)
    download_subset()

    video_paths = sorted(VIDEO_DIR.glob("*/*.mp4"))
    print(f"\nextracting keypoints for {len(video_paths)} downloaded clips...")
    for video_path in video_paths:
        gloss = video_path.parent.name
        out_path = PROCESSED_DIR / gloss / f"{video_path.stem}.npy"
        _, shape = process_video_file(video_path, out_path)
        print(f"  {gloss}/{video_path.name}: {shape}")


if __name__ == "__main__":
    main()
