"""Paired block-bootstrap CI for a model's MAE reduction vs. the persistence baseline.

Consecutive test sequences overlap and are autocorrelated, so the bootstrap
resamples contiguous blocks (default 12 sequences = 1 hour at 5-minute cadence).

Usage:
    python -m evaluation.bootstrap_ci --model simple_cnn --residual --frames-path processed/frames_128.npy
"""
import argparse

import numpy as np
import torch

from models.convlstm_model import ConvLSTMForecaster
from models.simple_cnn import SimpleCNNForecaster
from src.datasets.dataset_goes import GOESSequenceDataset
from src.utils.splits import get_split_ranges


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["simple_cnn", "convlstm"], required=True)
    parser.add_argument("--frames-path", default="processed/frames.npy")
    parser.add_argument("--checkpoint-path", default=None, help="Defaults to models/<model>[_residual].pth")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--block", type=int, default=12, help="Bootstrap block length in sequences.")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    name = f"{args.model}_residual" if args.residual else args.model
    if args.model == "simple_cnn":
        model = SimpleCNNForecaster(in_channels=args.seq_len, residual=args.residual)
    else:
        model = ConvLSTMForecaster(input_dim=1, hidden_dim=args.hidden_dim, residual=args.residual)
    model.load_state_dict(torch.load(args.checkpoint_path or f"models/{name}.pth", map_location="cpu"))
    model.eval()

    num_frames = len(np.load(args.frames_path, mmap_mode="r"))
    ds = GOESSequenceDataset(args.frames_path, args.seq_len, *get_split_ranges(num_frames, args.seq_len)["test"])
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.stack([ds[i][1] for i in range(len(ds))])

    with torch.no_grad():
        pred = torch.cat([model(x[i:i + 16]) for i in range(0, len(x), 16)])
    model_mae = (pred - y).abs().mean(dim=(1, 2, 3)).numpy()
    base_mae = (x[:, -1:] - y).abs().mean(dim=(1, 2, 3)).numpy()

    n, b = len(model_mae), args.block
    rng = np.random.default_rng(args.seed)
    boots = []
    for _ in range(args.n_boot):
        starts = rng.integers(0, n - b + 1, n // b + 1)
        idx = np.concatenate([np.arange(s, s + b) for s in starts])[:n]
        boots.append(1 - model_mae[idx].mean() / base_mae[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f"{name}: {n} test sequences | MAE {model_mae.mean():.6f} vs persistence {base_mae.mean():.6f}")
    print(f"MAE reduction {1 - model_mae.mean() / base_mae.mean():.1%} (95% CI {lo:.1%} to {hi:.1%}); "
          f"beats persistence on {(model_mae < base_mae).mean():.0%} of sequences")


if __name__ == "__main__":
    main()
