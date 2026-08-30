"""
My first model: a small CNN (Convolutional Neural Network, a type of model that's good at
looking at images because it scans for patterns like edges and shapes rather than treating every
pixel separately) that looks at a 28x28 image of a hand and guesses which letter it's signing.

Run it with:
    python src/cnn_baseline.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from data_loading import LABEL_TO_LETTER, load_split

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
EPOCHS = 8
BATCH_SIZE = 128
LEARNING_RATE = 1e-3


class SmallCNN(nn.Module):
    """Two conv layers to pick up on shapes, then a couple of normal layers to make the call."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def make_loaders(label_to_index: dict):
    X_train, y_train_raw = load_split("train")
    X_test, y_test_raw = load_split("test")

    y_train = np.array([label_to_index[v] for v in y_train_raw])
    y_test = np.array([label_to_index[v] for v in y_test_raw])

    train_ds = TensorDataset(
        torch.tensor(X_train).unsqueeze(1),  # add channel dim -> (N, 1, 28, 28)
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test).unsqueeze(1),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader, y_test


def train_and_evaluate():
    torch.manual_seed(RANDOM_STATE)

    class_labels = sorted(LABEL_TO_LETTER.keys())
    label_to_index = {label: i for i, label in enumerate(class_labels)}
    letters = [LABEL_TO_LETTER[label] for label in class_labels]

    train_loader, test_loader, y_test = make_loaders(label_to_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(num_classes=len(class_labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training on {device}, {len(class_labels)} letter classes: {letters}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"epoch {epoch}/{EPOCHS} - loss: {avg_loss:.4f}")

    # evaluate
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
    all_preds = np.array(all_preds)

    report = classification_report(y_test, all_preds, target_names=letters, output_dict=True, zero_division=0)
    accuracy = (all_preds == y_test).mean()
    print(classification_report(y_test, all_preds, target_names=letters, zero_division=0))
    print(f"Overall accuracy: {accuracy:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": "small_cnn_baseline",
        "epochs": EPOCHS,
        "n_train": len(train_loader.dataset),
        "n_test": len(test_loader.dataset),
        "accuracy": float(accuracy),
        "report": report,
    }
    with open(RESULTS_DIR / "cnn_baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # confusion matrix, 24 classes so make it a bit bigger than usual
    cm = confusion_matrix(y_test, all_preds)
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(cm, display_labels=letters).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"Sign letter CNN: confusion matrix (accuracy {accuracy:.3f})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cnn_baseline_confusion_matrix.png", dpi=150)
    plt.close(fig)

    torch.save(model.state_dict(), MODELS_DIR / "cnn_baseline.pt")
    print(f"\nSaved metrics + plot to {RESULTS_DIR}, model to {MODELS_DIR}")

    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()
