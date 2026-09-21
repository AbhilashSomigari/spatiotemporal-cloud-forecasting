"""Convert raw GOES NetCDF (.nc) files into a single normalized frame stack.

Crops each frame to a fixed region, optionally resizes it, clips brightness
temperature to a fixed physical range, and normalizes to [0, 1]. Frames are
kept in chronological order (files are sorted by name/timestamp) and saved
as one (N, H, W) float32 .npy array for GOESSequenceDataset to memory-map.

A fixed temperature range (rather than per-frame min/max) is used so pixel
values stay comparable across frames -- this matters for a forecasting
model, since per-frame normalization would make the same physical
temperature map to different values at different timestamps.

Usage:
    python scripts/extract_frames.py --raw-dir data/raw --out-path processed/frames.npy
    python scripts/extract_frames.py --raw-dir data/raw --out-path processed/frames_128.npy --resize 128 128
"""
import argparse
import glob
import os

import numpy as np
import xarray as xr
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw", help="Directory of raw .nc files (searched recursively).")
    parser.add_argument("--out-path", default="processed/frames.npy", help="Output path for the frame stack.")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("Y1", "Y2", "X1", "X2"), default=(300, 1200, 500, 1800),
                         help="Pixel crop region within the raw image.")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"), default=None,
                         help="Optional (height, width) to resize the crop to. Skips resizing if omitted.")
    parser.add_argument("--temp-range", type=float, nargs=2, metavar=("MIN_K", "MAX_K"), default=(180.0, 330.0),
                         help="Brightness temperature (Kelvin) range to clip and normalize against.")
    return parser.parse_args()


def load_and_process_frame(path, crop, resize, temp_range):
    y1, y2, x1, x2 = crop
    tmin, tmax = temp_range

    ds = xr.open_dataset(path, engine="netcdf4")
    try:
        img = ds["CMI"].values.astype(np.float32)
    finally:
        ds.close()

    if np.isnan(img).any():
        img = np.nan_to_num(img, nan=np.nanmean(img))

    crop_img = img[y1:y2, x1:x2]

    if resize is not None:
        import cv2
        h, w = resize
        crop_img = cv2.resize(crop_img, (w, h), interpolation=cv2.INTER_AREA)

    crop_img = np.clip(crop_img, tmin, tmax)
    return (crop_img - tmin) / (tmax - tmin)


def main():
    args = parse_args()

    files = sorted(glob.glob(os.path.join(args.raw_dir, "**", "*.nc"), recursive=True))
    print("Total files:", len(files))
    if not files:
        raise SystemExit(f"No .nc files found under {args.raw_dir}")

    frames = []
    for path in tqdm(files):
        try:
            frames.append(load_and_process_frame(path, args.crop, args.resize, args.temp_range))
        except Exception as e:
            print("Skipping:", path, "-", e)

    frames = np.array(frames, dtype=np.float32)
    print("Final shape:", frames.shape)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    np.save(args.out_path, frames)
    print("Saved:", args.out_path)


if __name__ == "__main__":
    main()
