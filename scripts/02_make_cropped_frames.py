import os
import glob
import numpy as np
import xarray as xr
import cv2
from tqdm import tqdm

RAW_DIR = "data/raw"
OUT_PATH = "data/cropped_frames/frames_crop_256.npy"

# CHANGE HERE
OUT_H, OUT_W = 256, 256

# Same crop
Y1, Y2 = 300, 1200
X1, X2 = 500, 1800

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

files = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.nc"), recursive=True))

frames = []

for file in tqdm(files):
    try:
        ds = xr.open_dataset(file, engine="netcdf4")
        img = ds["CMI"].values.astype(np.float32)
        ds.close()

        img = np.nan_to_num(img, nan=np.nanmean(img))

        crop = img[Y1:Y2, X1:X2]

        # CHANGE: resize to 128
        crop_resized = cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

        min_val = crop_resized.min()
        max_val = crop_resized.max()

        if max_val - min_val < 1e-6:
            continue

        crop_norm = (crop_resized - min_val) / (max_val - min_val)

        frames.append(crop_norm.astype(np.float32))

    except Exception as e:
        print("Skip:", e)

frames = np.array(frames, dtype=np.float32)

print("Shape:", frames.shape)

np.save(OUT_PATH, frames)
print("Saved:", OUT_PATH)