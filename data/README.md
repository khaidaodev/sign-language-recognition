# About the data

Not committed to this repo (see `.gitignore`), `src/data_loading.py` downloads it automatically the first time you run anything.

## Sign Language MNIST

It's a well known dataset originally on Kaggle, called ["Sign Language MNIST"](https://www.kaggle.com/datasets/datamunge/sign-language-mnist). Each row is one 28x28 grayscale photo of someone fingerspelling an ASL letter, flattened out into 784 pixel values, plus a label column saying which letter it is. Named after the classic MNIST handwritten digits dataset since it's set up the exact same way, just letters instead of digits.

24 letters total (not 26), J and Z are left out on purpose, both of those need you to move your hand while signing, and every image here is a single still frame.

- 27,455 training images
- 7,172 test images

Couldn't get the actual Kaggle download working in the environment I built this in, so `data_loading.py` pulls the same CSVs from a GitHub copy instead: [gurpreet0610/sign_language_CNN](https://github.com/gurpreet0610/sign_language_CNN).

## Word-level video data (not used yet)

For the next stage (recognising whole signed words from video), planning to use [WLASL](https://github.com/dxli94/WLASL), starting with a small subset of the most common words rather than the full ~2000 word vocabulary.
