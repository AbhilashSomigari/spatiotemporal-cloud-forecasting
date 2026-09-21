"""Save a sample original/cropped/resized image trio for visual sanity-checking
the crop region used by extract_frames.py."""
import argparse
import glob
import os
import random

import cv2
import numpy as np
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="outputs/crop_samples")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("Y1", "Y2", "X1", "X2"), default=(300, 1200, 500, 1800))
    parser.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"), default=(128, 128))
    return parser.parse_args()


def to_uint8(x):
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return (x * 255).astype(np.uint8)


def main():
    args = parse_args()
    y1, y2, x1, x2 = args.crop
    h, w = args.resize

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.raw_dir, "**", "*.nc"), recursive=True))
    if not files:
        raise SystemExit(f"No .nc files found under {args.raw_dir}")
    path = random.choice(files)
    print("Processing file:", path)

    ds = xr.open_dataset(path, engine="netcdf4")
    img = ds["CMI"].values.astype(np.float32)
    ds.close()

    if np.isnan(img).any():
        img = np.nan_to_num(img, nan=np.nanmean(img))

    crop = img[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(args.out_dir, "original.png"), to_uint8(img))
    cv2.imwrite(os.path.join(args.out_dir, "cropped.png"), to_uint8(crop))
    cv2.imwrite(os.path.join(args.out_dir, f"cropped_resized_{h}.png"), to_uint8(crop_resized))

    print("Saved images in:", args.out_dir)


if __name__ == "__main__":
    main()
