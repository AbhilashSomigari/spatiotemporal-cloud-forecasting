import os
import xarray as xr
import numpy as np
from tqdm import tqdm

RAW_DIR = "2025"
SAVE_PATH = "processed/frames.npy"

frames = []

#  collect all files
all_files = []
for root, dirs, files in os.walk(RAW_DIR):
    for f in files:
        if f.endswith(".nc"):
            all_files.append(os.path.join(root, f))

#  IMPORTANT: sort by time
all_files.sort()

print("Total files:", len(all_files))

for file in tqdm(all_files):
    try:
        ds = xr.open_dataset(file)
        img = ds["CMI"].values

        #  same crop (keep consistent!)
        crop = img[500:1012, 1000:1512]

        crop = np.nan_to_num(crop, nan=255.0)

        #  normalize
        crop = np.clip(crop, 180, 330)
        crop = (crop - 180) / (330 - 180)

        frames.append(crop)

    except Exception as e:
        print("Skipping:", file)

frames = np.array(frames)

print("Final shape:", frames.shape)

#  save
os.makedirs("processed", exist_ok=True)
np.save(SAVE_PATH, frames)