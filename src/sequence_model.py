"""
The actual model for stage 2: takes a sequence of body/hand keypoints (one row per video frame,
see src/pose_extraction.py) and predicts which word is being signed.

An LSTM (Long Short-Term Memory network, a type of RNN, "recurrent neural network" - a network
that reads a sequence one step at a time and carries a running memory of what it's seen so far)
rather than a plain feedforward network, because a sign is defined by how the hands move over
time, not just where they are in one single frame. The same set of keypoints in a different
order can mean a completely different sign, or no sign at all, and a feedforward network has no
way to represent "order" like that.

Run it with:
    python src/sequence_model.py     # sanity check: random input through the model, checks shapes
"""

import torch
from torch import nn


class SignLSTM(nn.Module):
    """LSTM classifier over keypoint sequences.

    Input:  (batch, seq_len, input_size) - a batch of padded keypoint sequences
    Output: (batch, num_classes) - raw scores (logits) for each word in the vocabulary, highest
            score wins
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        # bidirectional means the LSTM reads the sequence forwards AND backwards and sticks the
        # two together, which is why the classifier's input is hidden_size * 2, not hidden_size.
        # Doesn't need to be "real-time" here since it works on a whole recorded clip, not a live
        # stream, so there's no reason to only ever look backwards in time.
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden_size*2), one output vector per frame
        lstm_out, _ = self.lstm(x)
        # only the last frame's output is used, by then it's seen the whole sequence (forwards
        # and backwards), so it's the model's summary of the entire sign
        last_frame_out = lstm_out[:, -1, :]
        return self.classifier(last_frame_out)


if __name__ == "__main__":
    model = SignLSTM(input_size=258, num_classes=100)
    dummy_batch = torch.randn(4, 60, 258)  # 4 example clips, 60 frames each, 258 keypoint values
    logits = model(dummy_batch)
    print(f"input shape:  {tuple(dummy_batch.shape)}  (batch_size, seq_len, input_size)")
    print(f"output shape: {tuple(logits.shape)}  (batch_size, num_classes)")
