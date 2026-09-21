import numpy as np
import torch
from torch.utils.data import Dataset


class GOESSequenceDataset(Dataset):
    """Sliding-window next-frame dataset over a memory-mapped GOES frame stack.

    Expects a ``.npy`` file of shape (N, H, W) of frames normalized to [0, 1]
    (see scripts/extract_frames.py). Each sample is a sequence of ``seq_len``
    consecutive frames (x) and the frame immediately after it (y).
    """

    def __init__(self, frames_path, seq_len=6, start_idx=0, end_idx=None):
        self.frames = np.load(frames_path, mmap_mode="r")
        self.seq_len = seq_len
        self.start_idx = start_idx
        self.end_idx = len(self.frames) if end_idx is None else end_idx

        self.length = (self.end_idx - self.start_idx) - self.seq_len
        if self.length <= 0:
            raise ValueError("Not enough frames for the chosen seq_len and range.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = self.start_idx + idx

        x = self.frames[idx:idx + self.seq_len]   # (T, H, W)
        y = self.frames[idx + self.seq_len]       # (H, W)

        x = torch.tensor(np.array(x), dtype=torch.float32)             # (T, H, W)
        y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        return x, y
