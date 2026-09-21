"""Sanity-check a processed frame stack produced by extract_frames.py."""
import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-path", default="processed/frames.npy")
    parser.add_argument("--seq-len", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    frames = np.load(args.frames_path, mmap_mode="r")

    print("frames shape:", frames.shape)
    print("min/max:", frames.min(), frames.max())
    print("NaN?", np.isnan(frames).any())

    n = len(frames) - args.seq_len
    print("Number of sequence samples:", n)
    if n > 0:
        print("Example X shape:", frames[0:args.seq_len].shape)
        print("Example y shape:", frames[args.seq_len].shape)


if __name__ == "__main__":
    main()
