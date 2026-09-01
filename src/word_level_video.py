"""
Stage 2: recognising a signed word from a short video clip, instead of a single letter from a
still photo (that's stage 1, see cnn_baseline.py). Three steps, each in its own file:

    1. wlasl_metadata.py   - download the WLASL dataset index, pick a manageable subset of words
    2. video_downloader.py - download + trim the actual video clips for that subset
    3. pose_extraction.py  - turn each video into a sequence of body/hand keypoints (MediaPipe)

This file just runs all three back to back. The sequence model itself (an LSTM, see
sequence_model.py/train_sequence_model.py) reads whatever ends up in data/processed/.

A lot of this dataset's video links are old and dead by now (see video_downloader.py for why),
so don't expect every clip to download successfully, that's expected and handled, not a bug.
Whatever does download is enough to get keypoint sequences flowing end to end.

Two ways to run it:

    python3 src/word_level_video.py --num-words 100 --max-per-word 10
        Picks the num_words most-represented words in WLASL and downloads up to max_per_word
        clips for each. Use this the first time, or to grow the vocabulary (more words).

    python3 src/word_level_video.py --top-up-existing 40 --max-per-word 25
        Doesn't touch which words are in the vocabulary, instead re-downloads the N
        best-covered words already in data/processed/ with a higher cap, to get more example
        clips per word without adding more classes to tell apart. Useful once training shows
        the model overfitting because there's only a handful of clips per word, more data per
        word tends to help a lot more than more words at that point. Already-downloaded clips
        are skipped, only the new instances get fetched.

Either way, keypoint extraction only runs on clips that don't already have a .npy file, so
re-running this after downloading more data doesn't waste time reprocessing what's already done.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_extraction import process_video_file  # noqa: E402
from video_downloader import VIDEO_DIR, download_subset  # noqa: E402
from wlasl_metadata import save_subset, save_subset_for_glosses  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def best_covered_words(n: int) -> list[str]:
    """The n words already in data/processed/ with the most extracted clips, most first."""
    if not PROCESSED_DIR.exists():
        return []
    counts = {
        gloss_dir.name: len(list(gloss_dir.glob("*.npy")))
        for gloss_dir in PROCESSED_DIR.iterdir()
        if gloss_dir.is_dir()
    }
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [gloss for gloss, _ in ranked[:n]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-words", type=int, default=100, help="how many words to include")
    parser.add_argument(
        "--max-per-word", type=int, default=10, help="max video instances to try per word"
    )
    parser.add_argument(
        "--top-up-existing",
        type=int,
        default=None,
        metavar="N",
        help="instead of picking words by total WLASL instance count, top up the N "
        "best-covered words already in data/processed/ with more instances (see --max-per-word)",
    )
    args = parser.parse_args()

    if args.top_up_existing:
        words = best_covered_words(args.top_up_existing)
        if not words:
            print("data/processed/ is empty, nothing to top up, run without --top-up-existing first")
            return
        print(f"topping up {len(words)} words: {', '.join(words)}")
        save_subset_for_glosses(words, max_instances_per_word=args.max_per_word)
    else:
        save_subset(num_words=args.num_words, max_instances_per_word=args.max_per_word)

    download_subset()

    video_paths = sorted(VIDEO_DIR.glob("*/*.mp4"))
    already_done = 0
    to_process = []
    for video_path in video_paths:
        gloss = video_path.parent.name
        out_path = PROCESSED_DIR / gloss / f"{video_path.stem}.npy"
        if out_path.exists():
            already_done += 1
        else:
            to_process.append((video_path, out_path))

    print(
        f"\nextracting keypoints for {len(to_process)} new clips "
        f"({already_done} already processed, skipping those)..."
    )
    for video_path, out_path in to_process:
        gloss = video_path.parent.name
        _, shape = process_video_file(video_path, out_path)
        print(f"  {gloss}/{video_path.name}: {shape}")


if __name__ == "__main__":
    main()
