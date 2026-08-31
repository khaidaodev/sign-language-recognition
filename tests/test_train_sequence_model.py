"""
End-to-end smoke test for src/train_sequence_model.py: runs the real training loop against a
tiny made-up dataset (a handful of random keypoint sequences, not real signs) just to prove the
whole pipeline, dataset -> model -> loss -> backprop -> saved checkpoint, actually runs without
crashing. Not checking whether it learns anything sensible, random data has nothing real to
learn, this is purely a wiring check.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import train_sequence_model  # noqa: E402
from sequence_dataset import KeypointSequenceDataset  # noqa: E402


@pytest.fixture
def tiny_dataset(tmp_path):
    rng = np.random.default_rng(0)
    for gloss in ["book", "drink", "computer"]:
        gloss_dir = tmp_path / gloss
        gloss_dir.mkdir()
        for i in range(4):
            length = rng.integers(5, 20)
            np.save(gloss_dir / f"{i}.npy", rng.random((length, 258)))
    return KeypointSequenceDataset(processed_dir=tmp_path)


def test_train_runs_end_to_end_and_saves_a_checkpoint(tiny_dataset, tmp_path, monkeypatch):
    # 2 epochs is plenty to prove the wiring works, no need for the real 20 on made up data
    monkeypatch.setattr(train_sequence_model, "EPOCHS", 2)
    monkeypatch.setattr(train_sequence_model, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train_sequence_model, "RESULTS_DIR", tmp_path / "results")

    model, history = train_sequence_model.train(dataset=tiny_dataset)

    assert len(history) == 2
    assert (tmp_path / "models" / "sequence_model.pt").exists()
    assert (tmp_path / "results" / "sequence_model_history.json").exists()
    for row in history:
        assert 0.0 <= row["train_acc"] <= 1.0
        assert 0.0 <= row["val_acc"] <= 1.0
