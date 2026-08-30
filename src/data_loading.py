"""
Downloads the sign language MNIST dataset and turns it into numpy arrays I can train on.

It's images of people fingerspelling ASL letters with one hand, 28x28 pixels, grayscale. 24
letters (no J or Z, since those need you to move your hand, and this dataset is just still
images). Originally a Kaggle dataset, grabbed here from a GitHub mirror since Kaggle wasn't
reachable where I was building this.

Run it with:
    python src/data_loading.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

BASE_URL = "https://raw.githubusercontent.com/gurpreet0610/sign_language_CNN/master/sign-language-mnist"
FILES = {
    "train": "sign_mnist_train.csv",
    "test": "sign_mnist_test.csv",
}

# label numbers skip 9 (J) and 25 (Z) since those letters need motion, this dataset is still images
LABEL_TO_LETTER = {i: chr(ord("A") + i) for i in range(25) if i != 9}


def download_csv(split: str, force: bool = False) -> Path:
    """Grabs one of the two CSVs (train/test) if we don't already have it."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / FILES[split]
    if out_path.exists() and not force:
        return out_path

    url = f"{BASE_URL}/{FILES[split]}"
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path


def load_split(split: str, force: bool = False):
    """Returns (images, labels) for 'train' or 'test'. Images are (N, 28, 28) float32 in [0, 1]."""
    csv_path = download_csv(split, force=force)
    df = pd.read_csv(csv_path)

    labels = df["label"].to_numpy()
    pixels = df.drop(columns=["label"]).to_numpy(dtype="float32") / 255.0
    images = pixels.reshape(-1, 28, 28)

    return images, labels


if __name__ == "__main__":
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print(f"classes: {sorted(set(y_train))} -> letters {[LABEL_TO_LETTER[i] for i in sorted(set(y_train))]}")
