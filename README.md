# Sign Language Recognition

A model that looks at a photo of someone fingerspelling a letter in ASL (American Sign Language) and guesses which letter it is.

## Why this problem

Text-to-speech is everywhere now, but the other direction, software that actually understands sign language, barely exists, even though it's a first language for a lot of deaf and hard-of-hearing people. This is a small first step towards that: teaching a model to read a single signed letter from an image. Recognising a signed word or sentence from video is a much bigger, harder problem, that's the long-term direction this is heading in, this is just the starting point.

## Where it's at right now

Working model: it takes a 28x28 grayscale photo of a hand signing a letter and predicts which one it is, out of 24 letters (J and Z aren't included, they need you to move your hand and this dataset is all still images).

How it works: a small CNN (convolutional neural network, basically a model that scans an image for patterns like edges and shapes instead of looking at every pixel on its own). Two convolution layers to pick up on shapes in the hand, then a couple of normal layers to turn that into an actual letter guess.

Results on the test set (7,172 images, kept separate from training):

- Overall accuracy: **89.3%**
- Best letters: D, L (both basically perfect, easy to tell apart from everything else)
- Worst letters: R and S (both around 66-73%), these get mixed up with each other a lot, which actually makes sense, R and S look pretty similar in ASL

Full breakdown per letter, plus the confusion matrix showing exactly which letters get mixed up with which, is in `results/`.

## Stage 2: word-level video recognition (in progress)

The real next step: recognising whole signed **words** from **video**, not just a single letter from a still photo. Much harder, since you need to track movement over time, not just one frame.

The pipeline:

- `src/wlasl_metadata.py` downloads the [WLASL](https://github.com/dxli94/WLASL) dataset index (2,000 signed words, ~21,000 example video clips) and picks out a smaller subset to actually train on, the N words with the most example videos, capped per word, rather than trying to train on all 2,000 words at once. It can also top up words already partially downloaded with more instances instead of adding new words (`--top-up-existing`), useful once training shows a word doesn't have enough clips behind it.
- `src/video_downloader.py` downloads the actual clips for that subset and trims each one to the frame range the dataset says the sign happens in. This dataset is old and scraped from a lot of small ASL dictionary sites, plenty of those links are dead now, so this expects failures and just skips/logs them rather than stopping, keeps whatever it can get.
- `src/pose_extraction.py` turns each downloaded clip into a sequence of body + hand keypoints using MediaPipe's Tasks API, `PoseLandmarker` + `HandLandmarker` (258 numbers per frame: 33 pose landmarks + 21 landmarks per hand, each with x/y/z), instead of feeding raw video frames into a model. Lighter and works better for this than raw pixels.
- `src/word_level_video.py` runs all three of those back to back.
- `src/sequence_dataset.py` loads whatever's in `data/processed/` into something PyTorch can actually train on: normalizes every frame (centers on the shoulder midpoint, scales by shoulder width, so the same sign performed in a different spot in frame or a different distance from the camera looks the same to the model), pads/cuts every clip to a fixed length, and caps the vocabulary to the best-covered words rather than training on every word regardless of how little data is behind it.
- `src/sequence_model.py` is the actual model: a bidirectional LSTM (a type of neural network built for sequences, it reads the keypoints frame by frame and keeps a running "memory" of what it's seen, forwards and backwards through the clip) that takes a keypoint sequence and predicts which word it is.
- `src/train_sequence_model.py` trains it: 80/20 train/val split, on-the-fly augmentation on the training half only (a coin-flip left-right mirror plus small coordinate jitter, re-rolled fresh every time a clip is seen, so the same real clip looks a bit different epoch to epoch), saves the best checkpoint and the training history.

### Real results, and why the number is what it is

First real run: 100 words, ~650 downloaded clips, raw unnormalized keypoints. **4.6% validation accuracy**, barely above the ~1% random-guess baseline for 100 classes. Two problems, both visible in the training curve: heavy overfitting (training accuracy climbing while validation flatlined), and only about 5 real training clips per word on average, way too little for a 100-way classification problem, especially when the model also has to learn "the same sign performed slightly left of frame is still the same sign" from scratch, camera position was eating into that already-scarce data.

Fixed both, one at a time, checking the real number after each change:

| Change | Validation accuracy | Random baseline |
|---|---|---|
| Raw keypoints, 100 words | 4.6% | ~1% |
| + shoulder-centered normalization, capped to the 40 best-covered words | 9.7% | 2.5% |
| + topped up those 40 words with more real clips (7.8 → 12 avg per word) | 10.5% | 2.5% |
| + left-right mirror augmentation on the training split | 11.6% | 2.5% |

At that point training accuracy was still climbing well past validation accuracy (41% vs 11.6%), so this was still a data-limited problem: WLASL is an old dataset with a lot of dead links. Tried to fix that directly first, `--top-up-existing` with a higher cap and a fixed YouTube downloader (it was asking for a video/audio format combo a lot of modern YouTube videos don't have, plus no ffmpeg installed to merge separate streams) found real, previously-unfetched instances, but the actual videos behind them turned out to be genuinely gone (deleted, region-blocked, dead hosting), 0 new clips landed despite the fixes actually working correctly. Broadening the search to 50 more words WLASL lists (ranks 101-150 by instance count) didn't turn up a healthier alternative set either, same top word list came out on top.

So more data for this specific vocabulary had hit a real ceiling, roughly 12 real clips per word. The next lever was the vocabulary size itself: is 40 words actually the right target, or is a smaller, better-supported vocabulary a better trade? Tested 15/20/25/30/40 words, each across 3 random seeds (single-seed numbers on a validation set this small, 40-90 clips, are noisy enough to be misleading on their own):

| Vocabulary size | Val accuracy across seeds | Average | Random baseline |
|---|---|---|---|
| 15 words | 14.3%, 16.7%, 16.7% | 15.9% | 6.7% |
| **20 words** | 16.7%, 20.4%, **22.2%** | **19.8%** | 5.0% |
| 25 words | 12.1% (noisy, small val set) | ~12% | 4.0% |
| 30 words | (not fully tested) | - | 3.3% |
| 40 words | 10.5%, 11.6% | ~11% | 2.5% |

20 words came out clearly and consistently ahead, **~4x better than random guessing**, roughly double the 40-word result, for giving up half the vocabulary. Makes sense: with only ~12-14 real clips per word either way, the number of classes the model has to tell apart matters more than a slightly bigger word list. `MAX_WORDS` in `train_sequence_model.py` is set to 20 now, with the reasoning and numbers in a comment there too.

Model, data loader, and training loop were all unit tested against made-up keypoint data before ever touching real downloads, to make sure the wiring (shapes, normalization, augmentation, checkpoint saving) was correct independent of whether the data behind it was any good, see `tests/`.

Also still want a live demo where you show your webcam a sign and it guesses in real time, that's the next big piece.

**Update:** built half of that, but it's not fully working yet, calling this a milestone rather than done. `demo/webcam_demo.py` points MediaPipe's hand landmarker at a live webcam feed (same model as stage 2's keypoint extraction, just run frame-by-frame on a live camera instead of a saved video), crops around the detected hand, shrinks it to 28x28 to match what the CNN was trained on, and overlays its guess + confidence on the video window. All the actual logic (the crop/resize/normalize step, and turning the model's output back into a letter) is unit tested and correct, 8 new tests in `tests/test_webcam_demo.py`. What's not confirmed working yet: on my Mac, `cv2.VideoCapture` opens the camera fine but `.read()` never actually returns a frame, so the live window never shows anything. Camera permission is granted, and it's not a code bug in the logic above, it looks like an OpenCV/macOS camera backend issue, still tracking it down. Once that's sorted this becomes a genuinely working live demo with no further changes needed to the detection/model code.

A live demo for the stage 2 word-level model is a separate, harder piece on top of that: that model needs a whole clip's worth of frames to make a guess, not one still frame, so it needs some kind of sliding window over live video rather than a single crop-and-classify step, and its accuracy is still only ~22% on 20 words (see above), so a live version of it would mostly demo how often it's wrong right now. Left as future work.

## How to run this yourself

```bash
pip3 install -r requirements.txt
python3 src/cnn_baseline.py
```

Downloads the dataset automatically the first time (a couple of CSV files, not huge), trains for 8 epochs, prints the results, saves the plots.

For the live letter demo (needs a webcam):

```bash
python3 demo/webcam_demo.py
```

Needs `models/cnn_baseline.pt` to exist first (the `cnn_baseline.py` command above makes it). Press `q` in the video window to quit. See the note above though, this isn't confirmed working end-to-end on real hardware yet.

For stage 2 (once there's actually data downloaded, see above):

```bash
python3 src/word_level_video.py --num-words 150 --max-per-word 10         # downloads clips, extracts keypoints
python3 src/word_level_video.py --top-up-existing 20 --max-per-word 30    # optional: more clips for the best words
python3 src/train_sequence_model.py                                        # trains the LSTM on whatever downloaded
```

## Where the data's from

Kaggle has a well known "Sign Language MNIST" dataset. Couldn't get the Kaggle download working where I was building this, so I grabbed the same data from a GitHub copy instead, more detail in `data/README.md`.

## Testing and git

There's a `tests/` folder with real pytest tests, covers pose extraction, the sequence dataset loader, the LSTM model, and the training loop against made-up keypoint data (that's what let the pipeline get checked before real downloads even worked, see stage 2 above). 75 tests, and I run them before committing anything that touches the core logic. Git-wise the commits track the actual stages as they happened, metadata and downloading, then pose extraction, then the sequence model and training script, then the normalization/augmentation/vocabulary fixes that got the val accuracy up, so the history's basically a log of the debugging that's written up above.

## Tools used

Python, PyTorch for the CNN, pandas for loading the data, matplotlib for the plots. MediaPipe (Tasks API - `PoseLandmarker`/`HandLandmarker`, the old `solutions` API this used to use got dropped from newer mediapipe releases) and OpenCV for the stage 2 keypoint extraction.
