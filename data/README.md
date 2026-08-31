# About the data

Not committed to this repo (see `.gitignore`), `src/data_loading.py` downloads it automatically the first time you run anything.

## Sign Language MNIST

It's a well known dataset originally on Kaggle, called ["Sign Language MNIST"](https://www.kaggle.com/datasets/datamunge/sign-language-mnist). Each row is one 28x28 grayscale photo of someone fingerspelling an ASL letter, flattened out into 784 pixel values, plus a label column saying which letter it is. Named after the classic MNIST handwritten digits dataset since it's set up the exact same way, just letters instead of digits.

24 letters total (not 26), J and Z are left out on purpose, both of those need you to move your hand while signing, and every image here is a single still frame.

- 27,455 training images
- 7,172 test images

Couldn't get the actual Kaggle download working in the environment I built this in, so `data_loading.py` pulls the same CSVs from a GitHub copy instead: [gurpreet0610/sign_language_CNN](https://github.com/gurpreet0610/sign_language_CNN).

## Word-level video data (WLASL, stage 2)

For stage 2 (recognising whole signed words from video), using [WLASL](https://github.com/dxli94/WLASL) (Word-Level American Sign Language). `src/wlasl_metadata.py` downloads the full 2,000-word index and picks out the words with the most example videos, capped per word, rather than the full ~21,000-clip dataset.

Not the "official" WLASL100 split some papers use, that's a fixed word list published separately, this is my own subset built the same way (rank by number of example videos, take the top N), since I couldn't find that exact list published anywhere downloadable.

A lot of the video links in this dataset are dead now (old ASL dictionary sites that have since shut down or started blocking hotlinking, plus a chunk of the clips are unlisted/removed YouTube videos), `src/video_downloader.py` is written to expect that: it downloads whatever it can, skips and logs the rest, no need for every clip to succeed for this to work. YouTube-hosted clips need `yt-dlp` installed separately to fetch.

Downloaded clips go in `data/raw/videos/<word>/<video_id>.mp4`, extracted keypoint sequences go in `data/processed/<word>/<video_id>.npy`, neither committed (see `.gitignore`).

`src/pose_extraction.py` turns each clip into a keypoint sequence using MediaPipe's Tasks API (`PoseLandmarker` + `HandLandmarker`), which needs two small model bundle files. Those get downloaded automatically the first time it runs, into `data/mediapipe_models/` (also not committed, same reasoning as everything else here, easy to re-download, no reason to bloat the repo with them).
