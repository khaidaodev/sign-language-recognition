"""
Downloads the WLASL (Word-Level American Sign Language) dataset index and picks out a smaller,
manageable subset of it to actually work with.

WLASL_v0.3.json lists 2,000 signed words (glosses), each with a bunch of video "instances", one
row per video of someone signing that word, scraped from all over the place: aslsignbank,
YouTube, handspeak.com, a bunch of other ASL dictionary sites. Some words have 40 example videos,
some have as few as 5. Training on all 2,000 words / ~21,000 videos is a much bigger job than
this needs to be to prove the idea works, so this picks the N most-represented words (most
example videos = most for a model to actually learn from) and caps how many videos per word we
bother with.

Not the "official" WLASL100 split some papers use, that comes from a fixed word list published
separately, not just whichever words happen to have the most videos. This is my own subset, built
the same way papers describe it (rank by instance count, take the top N), since I couldn't find
that exact split published anywhere I could download.

Run it with:
    python src/wlasl_metadata.py
"""

import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

METADATA_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
METADATA_PATH = RAW_DIR / "WLASL_v0.3.json"
SUBSET_PATH = RAW_DIR / "wlasl_subset.json"


def download_metadata(force: bool = False) -> Path:
    """Grabs the full 2,000-word dataset index (~12MB of JSON) if we don't already have it."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if METADATA_PATH.exists() and not force:
        return METADATA_PATH

    print(f"Downloading {METADATA_URL} ...")
    resp = requests.get(METADATA_URL, timeout=60)
    resp.raise_for_status()
    METADATA_PATH.write_bytes(resp.content)
    return METADATA_PATH


def build_subset(all_glosses, num_words=100, max_instances_per_word=10):
    """Picks the num_words glosses with the most video instances, and caps how many instances
    we keep per word. Keeps instances in the order the dataset lists them, no shuffling, so this
    is deterministic and reproducible."""
    ranked = sorted(all_glosses, key=lambda g: len(g["instances"]), reverse=True)
    subset = []
    for gloss_entry in ranked[:num_words]:
        trimmed = dict(gloss_entry)
        trimmed["instances"] = gloss_entry["instances"][:max_instances_per_word]
        subset.append(trimmed)
    return subset


def save_subset(num_words=100, max_instances_per_word=10, force=False) -> Path:
    metadata_path = download_metadata(force=force)
    all_glosses = json.loads(metadata_path.read_text())
    subset = build_subset(all_glosses, num_words, max_instances_per_word)

    SUBSET_PATH.write_text(json.dumps(subset, indent=2))
    total_videos = sum(len(g["instances"]) for g in subset)
    print(f"{len(subset)} words, {total_videos} video instances -> {SUBSET_PATH}")
    return SUBSET_PATH


if __name__ == "__main__":
    save_subset()
