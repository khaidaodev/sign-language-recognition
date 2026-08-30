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

## What's still left to build

- The real next step: recognising whole signed **words** from **video**, not just a single letter from a still photo. Much harder, since you need to track movement over time, not just one frame. Planning to use MediaPipe to pull hand/body keypoints out of video and train a sequence model on that.
- A live demo where you show your webcam a letter and it guesses in real time

Notes on both of those are in `src/word_level_video.py` and `demo/webcam_demo.py`.

## How to run this yourself

```bash
pip install -r requirements.txt
python src/cnn_baseline.py
```

Downloads the dataset automatically the first time (a couple of CSV files, not huge), trains for 8 epochs, prints the results, saves the plots.

## Where the data's from

Kaggle has a well known "Sign Language MNIST" dataset. Couldn't get the Kaggle download working where I was building this, so I grabbed the same data from a GitHub copy instead, more detail in `data/README.md`.

## Tools used

Python, PyTorch for the CNN, pandas for loading the data, matplotlib for the plots. Planning to add MediaPipe and OpenCV for the video/webcam stage later.
