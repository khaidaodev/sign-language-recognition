"""
Tests for the subset-picking logic in src/wlasl_metadata.py, using small made-up gloss lists
instead of the real 2,000-word dataset so this runs instantly and doesn't need a download.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wlasl_metadata import build_subset  # noqa: E402


def _gloss(name, num_instances):
    return {"gloss": name, "instances": [{"video_id": str(i)} for i in range(num_instances)]}


def test_build_subset_picks_most_represented_words():
    glosses = [_gloss("book", 40), _gloss("go", 5), _gloss("chair", 26)]
    subset = build_subset(glosses, num_words=2, max_instances_per_word=100)
    names = [g["gloss"] for g in subset]
    assert names == ["book", "chair"]


def test_build_subset_caps_instances_per_word():
    glosses = [_gloss("book", 40)]
    subset = build_subset(glosses, num_words=1, max_instances_per_word=5)
    assert len(subset[0]["instances"]) == 5


def test_build_subset_keeps_all_instances_when_under_the_cap():
    glosses = [_gloss("go", 5)]
    subset = build_subset(glosses, num_words=1, max_instances_per_word=10)
    assert len(subset[0]["instances"]) == 5


def test_build_subset_num_words_larger_than_available():
    glosses = [_gloss("book", 40), _gloss("go", 5)]
    subset = build_subset(glosses, num_words=10, max_instances_per_word=10)
    assert len(subset) == 2
