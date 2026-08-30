"""
Haven't started this bit yet. This is the more ambitious next step: instead of a single still
image of a letter, recognise a whole signed word from a short video clip.

The idea:
    - Use MediaPipe to pull out hand/body keypoints (the x/y position of joints, frame by frame)
      from each video instead of feeding in raw pixels, way lighter and works better for this.
    - Train a sequence model (something like an LSTM or a small transformer) on those keypoint
      sequences to predict which word is being signed.
    - Dataset: WLASL (Word-Level American Sign Language), starting with a small subset of the
      most common words (WLASL100) rather than the full ~2000 word vocabulary, to keep training
      realistic. https://github.com/dxli94/WLASL

Needs a good amount of preprocessing work (downloading/trimming the actual video clips) before
any model training can start.
"""

raise NotImplementedError("haven't built this bit yet, see the plan in README.md")
