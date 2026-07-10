import os
import glob
import random
import xarray as xr
import numpy as np
import cv2

RAW_DIR = "data/raw"
OUT_DIR = "outputs/crop_samples"

# Crop coordinates (same as training)
Y1, Y2 = 300, 1200
X1, X2 = 500, 1800

os.makedirs(OUT_DIR, exist_ok=True)

# Pick a random file
files = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.nc"), recursive=True))
file = random.choice(files)

print("Processing file:", file)

# Load image
ds = xr.open_dataset(file, engine="netcdf4")
img = ds["CMI"].values.astype(np.float32)
ds.close()

# Handle NaNs
img = np.nan_to_num(img, nan=np.nanmean(img))

# Crop
crop = img[Y1:Y2, X1:X2]

# Resize
crop_resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)

# Normalize to 0–255 for saving
def to_uint8(x):
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return (x * 255).astype(np.uint8)

img_u8 = to_uint8(img)
crop_u8 = to_uint8(crop)
crop_resized_u8 = to_uint8(crop_resized)

# Save images
cv2.imwrite(os.path.join(OUT_DIR, "original.png"), img_u8)
cv2.imwrite(os.path.join(OUT_DIR, "cropped.png"), crop_u8)
cv2.imwrite(os.path.join(OUT_DIR, "cropped_resized_128.png"), crop_resized_u8)

print("Saved images in:", OUT_DIR)