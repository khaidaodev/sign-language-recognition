"""
Tests for the subset-picking logic in src/wlasl_metadata.py, using small made-up gloss lists
instead of the real 2,000-word dataset so this runs instantly and doesn't need a download.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wlasl_metadata import build_subset, build_subset_for_glosses  # noqa: E402


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


def test_build_subset_for_glosses_keeps_exactly_the_words_asked_for():
    glosses = [_gloss("book", 40), _gloss("go", 5), _gloss("chair", 26)]
    subset = build_subset_for_glosses(glosses, wanted_glosses=["chair", "go"], max_instances_per_word=100)
    names = {g["gloss"] for g in subset}
    assert names == {"chair", "go"}


def test_build_subset_for_glosses_preserves_the_requested_order():
    # topping up should list the best-covered word first, same order best_covered_words gives it
    glosses = [_gloss("book", 40), _gloss("go", 5), _gloss("chair", 26)]
    subset = build_subset_for_glosses(glosses, wanted_glosses=["go", "book"], max_instances_per_word=100)
    assert [g["gloss"] for g in subset] == ["go", "book"]


def test_build_subset_for_glosses_raises_the_cap():
    # this is the whole point of --top-up-existing: get MORE instances for words that already
    # have some, so the cap should still apply per word same as build_subset
    glosses = [_gloss("book", 40)]
    subset = build_subset_for_glosses(glosses, wanted_glosses=["book"], max_instances_per_word=25)
    assert len(subset[0]["instances"]) == 25


def test_build_subset_for_glosses_skips_unknown_words_instead_of_erroring():
    glosses = [_gloss("book", 40)]
    subset = build_subset_for_glosses(glosses, wanted_glosses=["book", "not-a-real-word"], max_instances_per_word=10)
    assert [g["gloss"] for g in subset] == ["book"]
