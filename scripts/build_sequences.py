import numpy as np
import os

FRAME_PATH = "processed/frames.npy"
SAVE_DIR = "processed"

frames = np.load(FRAME_PATH)

sequence_length = 6

X = []
y = []

for i in range(len(frames) - sequence_length):
    X.append(frames[i:i + sequence_length])
    y.append(frames[i + sequence_length])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)

os.makedirs(SAVE_DIR, exist_ok=True)
np.save(os.path.join(SAVE_DIR, "X.npy"), X)
np.save(os.path.join(SAVE_DIR, "y.npy"), y)