"""Evaluate a persistence baseline or a trained model on the test split.

Usage:
    python -m evaluation.evaluate --model persistence --frames-path processed/frames.npy
    python -m evaluation.evaluate --model simple_cnn --frames-path processed/frames.npy --checkpoint-path models/simple_cnn.pth
    python -m evaluation.evaluate --model convlstm --frames-path processed/frames.npy --checkpoint-path models/convlstm.pth
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.convlstm_model import ConvLSTMForecaster
from models.simple_cnn import SimpleCNNForecaster
from src.datasets.dataset_goes import GOESSequenceDataset
from src.utils.splits import get_split_ranges


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["persistence", "simple_cnn", "convlstm"], required=True)
    parser.add_argument("--frames-path", default="processed/frames.npy")
    parser.add_argument("--checkpoint-path", default=None, help="Required for simple_cnn/convlstm; defaults to models/<model>.pth")
    parser.add_argument("--results-path", default="outputs/results.json")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32, help="ConvLSTM hidden channels.")
    return parser.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(args, device):
    if args.model == "simple_cnn":
        model = SimpleCNNForecaster(in_channels=args.seq_len)
    else:
        model = ConvLSTMForecaster(input_dim=1, hidden_dim=args.hidden_dim)

    checkpoint_path = args.checkpoint_path or f"models/{args.model}.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device)


def record_result(results_path, entry):
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    results = [r for r in results if r["model"] != entry["model"]]
    results.append(entry)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    args = parse_args()
    device = get_device()
    print("Using device:", device)

    num_frames = len(np.load(args.frames_path, mmap_mode="r"))
    ranges = get_split_ranges(num_frames, args.seq_len)
    ds = GOESSequenceDataset(args.frames_path, args.seq_len, *ranges[args.split])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = None if args.model == "persistence" else build_model(args, device)
    if model is not None:
        model.eval()

    mae_sum, mse_sum, count = 0.0, 0.0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if model is None:
                # naive baseline: predict the last observed frame
                pred = x[:, -1:, :, :]
            else:
                pred = model(x)

            diff = pred - y
            mae_sum += torch.abs(diff).sum().item()
            mse_sum += (diff ** 2).sum().item()
            count += diff.numel()

    mae, mse = mae_sum / count, mse_sum / count
    print(f"{args.model} [{args.split}] | mae={mae:.6f} | mse={mse:.6f}")

    record_result(args.results_path, {
        "model": args.model,
        "split": args.split,
        "frames_path": args.frames_path,
        "mae": mae,
        "mse": mse,
    })


if __name__ == "__main__":
    main()
