"""
Live webcam demo for the letter-fingerspelling model (src/cnn_baseline.py). Point your webcam at
one hand and it guesses which letter you're signing, updated every frame.

This is the "50%" version: it only covers the stage 1 letter CNN, since that's the model that's
actually good (~95%+ test accuracy). The stage 2 word-level model (src/sequence_model.py) needs a
whole clip's worth of frames rather than one still frame to make a guess, and its accuracy is
still low (best so far ~22% on a 20-word vocabulary, see README), so a live demo for that is a
separate, harder piece of future work rather than something worth wiring up yet.

How it works: MediaPipe's hand landmarker finds a hand in each frame (same model/pattern as
src/pose_extraction.py, just run live instead of on a saved video). We crop a box around the
detected hand, shrink it to 28x28 grayscale to match the training data
(sign language MNIST, see src/data_loading.py), and feed that into the trained SmallCNN.

Needs models/cnn_baseline.pt to exist first, run `python3 src/cnn_baseline.py` if it doesn't
(that trains it and saves it there).

Run it with:
    python3 demo/webcam_demo.py
Press 'q' in the video window to quit.

macOS sometimes offers more than one camera device, e.g. an iPhone signed into the same Apple ID
can appear as a webcam via Continuity Camera, and camera index 0 isn't always the real built-in
one. If the picture looks wrong, pass --camera-index 1, 2, etc. until you land on the right one:
    python3 demo/webcam_demo.py --camera-index 1
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cnn_baseline import SmallCNN  # noqa: E402
from data_loading import LABEL_TO_LETTER  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# same ordering src/cnn_baseline.py used when it trained (class_labels = sorted(...)), so output
# index i from the model corresponds to CLASS_LABELS[i], which maps to a letter through
# LABEL_TO_LETTER. Has to match exactly or the letters come out scrambled.
CLASS_LABELS = sorted(LABEL_TO_LETTER.keys())
INDEX_TO_LETTER = {i: LABEL_TO_LETTER[label] for i, label in enumerate(CLASS_LABELS)}


def hand_bounding_box(landmarks, frame_width, frame_height, padding=40):
    """Takes a list of MediaPipe hand landmarks (normalized 0-1 x/y coords) and returns a pixel
    space box (left, top, right, bottom) around the hand, padded out a bit so the crop isn't
    drawn right up against the fingertips, forced square (pads the shorter side out to match the
    longer one, centered) so shrinking it to the model's 28x28 input doesn't stretch/squash the
    hand out of shape, and clamped so it never goes outside the frame."""
    xs = [lm.x * frame_width for lm in landmarks]
    ys = [lm.y * frame_height for lm in landmarks]

    left = min(xs) - padding
    right = max(xs) + padding
    top = min(ys) - padding
    bottom = max(ys) + padding

    width, height = right - left, bottom - top
    if width > height:
        extra = (width - height) / 2
        top -= extra
        bottom += extra
    else:
        extra = (height - width) / 2
        left -= extra
        right += extra

    left = max(0, int(left))
    top = max(0, int(top))
    right = min(frame_width, int(right))
    bottom = min(frame_height, int(bottom))
    return left, top, right, bottom


def crop_to_model_input(gray_frame, box):
    """Crops a grayscale frame down to the given box, resizes to 28x28 and scales pixels to
    [0, 1], the exact preprocessing load_split() applies in src/data_loading.py. Returns a
    (1, 1, 28, 28) float32 tensor, ready to feed straight into SmallCNN.

    Also runs histogram equalization (spreads out the brightness values in the image so they
    cover the full 0-255 range instead of being bunched up) before resizing. The training photos
    (sign language MNIST) were shot under harsh, one-directional studio lighting and come out
    very high contrast, a normal webcam under normal room lighting looks much flatter by
    comparison, and the model never saw that during training. Equalizing pushes a live, flatly
    lit hand towards the kind of high-contrast look the model actually learned on."""
    left, top, right, bottom = box
    crop = gray_frame[top:bottom, left:right]
    equalized = cv2.equalizeHist(crop)
    resized = cv2.resize(equalized, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized)
    return tensor.unsqueeze(0).unsqueeze(0)  # add channel + batch dims -> (1, 1, 28, 28)


def predict_letter(model, input_tensor):
    """Runs the model on a preprocessed (1, 1, 28, 28) tensor, returns (letter, confidence)."""
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        index = int(torch.argmax(probabilities))
    return INDEX_TO_LETTER[index], float(probabilities[index])


def _load_model():
    checkpoint_path = MODELS_DIR / "cnn_baseline.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} doesn't exist yet, run `python3 src/cnn_baseline.py` first to "
            "train it and save it there."
        )
    model = SmallCNN(num_classes=len(CLASS_LABELS))
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def run_demo(camera_index: int = 0) -> None:
    """The actual live loop: opens the webcam, runs hand detection + the CNN on every frame, and
    shows the result in a window until 'q' is pressed. Not something we can unit test, this is
    the glue that ties the tested pieces above to a real camera and a real screen."""
    # imported here rather than at the top of the file so the pure functions above can be
    # imported and tested without needing a real MediaPipe hand landmarker set up
    from mediapipe import Image, ImageFormat

    from pose_extraction import _make_hand_landmarker

    model = _load_model()
    hand_landmarker = _make_hand_landmarker()
    # explicit AVFoundation backend: on macOS, letting OpenCV pick a backend automatically was
    # unreliable here, camera opened fine (isOpened() True) but .read() never actually returned a
    # frame. Also worth knowing: if something else with a virtual camera (e.g. OBS, or an iPhone
    # via Continuity Camera) is active, it can register its own camera device and get picked up
    # instead of the real one, use --camera-index to pick a different one if that happens.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"couldn't open camera {camera_index}")

    frame_index = 0
    consecutive_failed_reads = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # right after macOS grants camera access the feed can take a moment to actually
                # start producing frames, so a handful of failed reads in a row isn't fatal, only
                # give up if it never comes good
                consecutive_failed_reads += 1
                if consecutive_failed_reads > 60:
                    print("no frames came from the camera, giving up")
                    break
                continue
            consecutive_failed_reads = 0
            # NOT mirroring this: the training photos are normal (non-mirrored) shots of a hand,
            # so mirroring here would feed the model a backwards version of every letter you show
            # it, which is close to what was happening before this fix
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image(image_format=ImageFormat.SRGB, data=rgb)
            # VIDEO mode just needs strictly increasing timestamps, frame index works fine, same
            # trick src/pose_extraction.py uses
            hand_result = hand_landmarker.detect_for_video(image, frame_index)
            frame_index += 1

            if hand_result.hand_landmarks:
                landmarks = hand_result.hand_landmarks[0]
                box = hand_bounding_box(landmarks, width, height)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                input_tensor = crop_to_model_input(gray, box)
                letter, confidence = predict_letter(model, input_tensor)

                left, top, right, bottom = box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{letter} ({confidence:.0%})",
                    (left, max(20, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame, "show me a hand", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                )

            cv2.imshow("sign language letters - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        hand_landmarker.close()
        cv2.destroyAllWindows()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="which camera device to use (default 0). macOS can expose more than one, e.g. an "
             "iPhone via Continuity Camera, try 1, 2, etc. if 0 picks the wrong one.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_demo(camera_index=args.camera_index)
