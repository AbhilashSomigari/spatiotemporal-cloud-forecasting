import os
import numpy as np
from tqdm import tqdm

FRAMES_PATH = "data/cropped_frames/frames_crop_256.npy"
OUT_DIR = "data/sequences_crop"

SEQ_LEN = 6

os.makedirs(OUT_DIR, exist_ok=True)

frames = np.load(FRAMES_PATH)

print("Loaded frames:", frames.shape)

X = []
y = []

for i in tqdm(range(len(frames) - SEQ_LEN)):
    X.append(frames[i:i+SEQ_LEN])
    y.append(frames[i+SEQ_LEN])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)

np.save(os.path.join(OUT_DIR, "X_crop.npy"), X)
np.save(os.path.join(OUT_DIR, "y_crop.npy"), y)

print("Saved cropped sequences.")