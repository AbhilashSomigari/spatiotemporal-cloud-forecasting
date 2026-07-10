import numpy as np
from tqdm import tqdm

FRAMES_PATH = "data/cropped_frames/frames_crop_256.npy"
SEQ_LEN = 6

frames = np.load(FRAMES_PATH, mmap_mode="r")

N = len(frames) - SEQ_LEN
train_end = int(0.70 * N)
val_end = int(0.85 * N)

mae_sum, mse_sum, count = 0.0, 0.0, 0

for i in tqdm(range(val_end, N)):
    y_true = frames[i + SEQ_LEN]
    y_pred = frames[i + SEQ_LEN - 1]

    diff = y_true - y_pred

    mae_sum += np.abs(diff).sum()
    mse_sum += (diff ** 2).sum()
    count += diff.size

print("MAE:", mae_sum / count)
print("MSE:", mse_sum / count)