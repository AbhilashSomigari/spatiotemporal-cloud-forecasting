"""Train SimpleCNNForecaster or ConvLSTMForecaster on GOES frame sequences.

Usage:
    python -m training.train --model simple_cnn --frames-path processed/frames.npy
    python -m training.train --model convlstm --frames-path processed/frames.npy --hidden-dim 32
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.convlstm_model import ConvLSTMForecaster
from models.simple_cnn import SimpleCNNForecaster
from src.datasets.dataset_goes import GOESSequenceDataset
from src.utils.splits import get_split_ranges


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["simple_cnn", "convlstm"], required=True)
    parser.add_argument("--frames-path", default="processed/frames.npy")
    parser.add_argument("--checkpoint-path", default=None, help="Defaults to models/<model>.pth")
    parser.add_argument("--results-path", default="outputs/results.json")
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32, help="ConvLSTM hidden channels.")
    parser.add_argument("--max-steps", type=int, default=None, help="Cap steps/epoch, for smoke-testing.")
    return parser.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(args):
    if args.model == "simple_cnn":
        return SimpleCNNForecaster(in_channels=args.seq_len)
    return ConvLSTMForecaster(input_dim=1, hidden_dim=args.hidden_dim)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mae_sum, mse_sum, count = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        diff = pred - y

        mae_sum += torch.abs(diff).sum().item()
        mse_sum += (diff ** 2).sum().item()
        count += diff.numel()

    return mae_sum / count, mse_sum / count


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
    checkpoint_path = args.checkpoint_path or f"models/{args.model}.pth"
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    device = get_device()
    print("Using device:", device)

    num_frames = len(np.load(args.frames_path, mmap_mode="r"))
    ranges = get_split_ranges(num_frames, args.seq_len)

    train_ds = GOESSequenceDataset(args.frames_path, args.seq_len, *ranges["train"])
    val_ds = GOESSequenceDataset(args.frames_path, args.seq_len, *ranges["val"])
    test_ds = GOESSequenceDataset(args.frames_path, args.seq_len, *ranges["test"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(args).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_mse = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if args.max_steps is not None and step + 1 >= args.max_steps:
                break

        avg_loss = total_loss / (step + 1)
        val_mae, val_mse = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{args.epochs} | train_loss={avg_loss:.6f} | val_mae={val_mae:.6f} | val_mse={val_mse:.6f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), checkpoint_path)
            print("Saved best model to", checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_mae, test_mse = evaluate(model, test_loader, device)
    print(f"Final test | mae={test_mae:.6f} | mse={test_mse:.6f}")

    record_result(args.results_path, {
        "model": args.model,
        "split": "test",
        "frames_path": args.frames_path,
        "mae": test_mae,
        "mse": test_mse,
    })


if __name__ == "__main__":
    main()
