import numpy as np

frames = np.load("data/cropped_frames/frames_crop_128.npy", mmap_mode="r")

print("frames shape:", frames.shape)
print("min/max:", frames.min(), frames.max())
print("NaN?", np.isnan(frames).any())

seq_len = 6
N = len(frames) - seq_len

print("Number of sequence samples:", N)
print("Example X shape:", frames[0:6].shape)
print("Example y shape:", frames[6].shape)