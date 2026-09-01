"""
Trains the LSTM (src/sequence_model.py) on whatever keypoint sequences are sitting in
data/processed/ (built by running src/word_level_video.py first).

Won't do anything useful until data/processed/ actually has a decent number of clips in it,
which needs real video downloads to have worked (see video_downloader.py for why a lot of them
won't, this dataset's links are old). Written now so it's ready to go the moment there's real
data, rather than waiting until then to write it.

Run it with:
    python3 src/train_sequence_model.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from pose_extraction import FRAME_VECTOR_SIZE
from sequence_dataset import AugmentedTrainDataset, KeypointSequenceDataset
from sequence_model import SignLSTM

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # a bit of L2 regularization, real training data here is thin (a handful of
                      # clips per word), so the model overfits fast without some push-back

# With a real but small download batch, some words only end up with 3-4 example clips while
# others have 8-10. Every extra word makes classification harder without necessarily adding much
# signal if there's barely any data behind it, so this caps the vocabulary to the best-covered
# words rather than training on the full downloaded set right away. See
# sequence_dataset.py's KeypointSequenceDataset for how the cutoff is chosen. Bump this up (or
# set to None) once more clips have been downloaded per word.
MAX_WORDS = 40


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over `loader`. Updates the model's weights if given an optimizer, otherwise
    just evaluates without changing anything. Returns (average loss, accuracy)."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_training):
        for sequences, labels in loader:
            logits = model(sequences)
            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


def train(dataset: KeypointSequenceDataset | None = None):
    """dataset can be passed in directly (mainly so tests can point this at a small fake
    dataset instead of the real data/processed/ folder), otherwise this loads the real one."""
    torch.manual_seed(RANDOM_STATE)
    dataset = dataset or KeypointSequenceDataset(max_words=MAX_WORDS)

    # 80/20 train/val split, same idea as the fingerspelling CNN in stage 1: hold some clips
    # back so the accuracy number below actually means the model never trained on them
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_STATE)
    )
    # augment only the training half, val stays real, unmodified clips throughout, see
    # AugmentedTrainDataset's docstring for why that split matters
    augmented_train_set = AugmentedTrainDataset(train_set, rng=np.random.default_rng(RANDOM_STATE))

    train_loader = DataLoader(augmented_train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = SignLSTM(input_size=FRAME_VECTOR_SIZE, num_classes=dataset.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(
        f"{len(dataset)} clips ({train_size} train, augmented to {len(augmented_train_set)} "
        f"views/epoch / {val_size} val), {dataset.num_classes} words"
    )

    best_val_acc = 0.0
    history = []
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"epoch {epoch:2d}/{EPOCHS}  train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_to_gloss": dataset.label_to_gloss,
                    "num_classes": dataset.num_classes,
                },
                MODELS_DIR / "sequence_model.pt",
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "sequence_model_history.json", "w") as f:
        json.dump({"history": history, "best_val_acc": best_val_acc}, f, indent=2)

    print(f"\nbest val accuracy: {best_val_acc:.3f}")
    print(f"saved best model to {MODELS_DIR / 'sequence_model.pt'}, history to {RESULTS_DIR}")

    return model, history


if __name__ == "__main__":
    train()
