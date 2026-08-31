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

So far:

- `src/wlasl_metadata.py` downloads the [WLASL](https://github.com/dxli94/WLASL) dataset index (2,000 signed words, ~21,000 example video clips) and picks out a smaller subset to actually train on, the N words with the most example videos, capped per word, rather than trying to train on all 2,000 words at once.
- `src/video_downloader.py` downloads the actual clips for that subset and trims each one to the frame range the dataset says the sign happens in. This dataset is old and scraped from a lot of small ASL dictionary sites, plenty of those links are dead now, so this expects failures and just skips/logs them rather than stopping, keeps whatever it can get.
- `src/pose_extraction.py` turns each downloaded clip into a sequence of body + hand keypoints using MediaPipe's Tasks API, `PoseLandmarker` + `HandLandmarker` (258 numbers per frame: 33 pose landmarks + 21 landmarks per hand, each with x/y/z), instead of feeding raw video frames into a model. Lighter and works better for this than raw pixels. Two small model files get downloaded automatically the first time it runs.
- `src/word_level_video.py` runs all three of those back to back.
- `src/sequence_dataset.py` loads whatever's in `data/processed/` into something PyTorch can actually train on, padding/cutting every clip to the same fixed length since clips run different lengths but a model needs every example in a batch to match shape.
- `src/sequence_model.py` is the actual model: a bidirectional LSTM (a type of neural network built for sequences, it reads the keypoints frame by frame and keeps a running "memory" of what it's seen, forwards and backwards through the clip) that takes a keypoint sequence and predicts which word it is.
- `src/train_sequence_model.py` trains it: 80/20 train/val split, same idea as the fingerspelling CNN in stage 1, saves the best checkpoint and the training history.

None of the training pieces have actually been run on real data yet though, `data/processed/` is empty right now, since downloading a proper batch of WLASL clips needs a decent chunk of time and a network that isn't fighting a lot of dead links (see `video_downloader.py`). Model, data loader, and training loop are all written and tested against made-up keypoint data to make sure the wiring's correct (shapes, padding, checkpoint saving all work), but the real "does it actually learn to recognise signs" number doesn't exist yet, that's the next thing: actually run `word_level_video.py` for real with enough clips, then `train_sequence_model.py` on the result.

Also still want a live demo where you show your webcam a letter and it guesses in real time.

## How to run this yourself

```bash
pip3 install -r requirements.txt
python3 src/cnn_baseline.py
```

Downloads the dataset automatically the first time (a couple of CSV files, not huge), trains for 8 epochs, prints the results, saves the plots.

For stage 2 (once there's actually data downloaded, see above):

```bash
python3 src/word_level_video.py --num-words 100 --max-per-word 10   # downloads clips, extracts keypoints
python3 src/train_sequence_model.py                                  # trains the LSTM on whatever downloaded
```

## Where the data's from

Kaggle has a well known "Sign Language MNIST" dataset. Couldn't get the Kaggle download working where I was building this, so I grabbed the same data from a GitHub copy instead, more detail in `data/README.md`.

## Tools used

Python, PyTorch for the CNN, pandas for loading the data, matplotlib for the plots. MediaPipe (Tasks API - `PoseLandmarker`/`HandLandmarker`, the old `solutions` API this used to use got dropped from newer mediapipe releases) and OpenCV for the stage 2 keypoint extraction.
