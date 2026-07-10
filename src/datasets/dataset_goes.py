import numpy as np
import torch
from torch.utils.data import Dataset

class GOESSequenceDataset(Dataset):
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

        x = self.frames[idx:idx + self.seq_len]          # (seq_len, H, W)
        y = self.frames[idx + self.seq_len]              # (H, W)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1, H, W)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        return x, y