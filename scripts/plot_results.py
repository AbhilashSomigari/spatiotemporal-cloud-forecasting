"""Plot MAE per model from outputs/results.json (written by training/train.py
and evaluation/evaluate.py)."""
import argparse
import json
import os

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", default="outputs/results.json")
    parser.add_argument("--out-path", default="outputs/mae_comparison.png")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.results_path):
        raise SystemExit(
            f"No results file at {args.results_path}. Run training/train.py or "
            "evaluation/evaluate.py first to generate results."
        )

    with open(args.results_path) as f:
        results = json.load(f)

    if not results:
        raise SystemExit(f"{args.results_path} is empty.")

    models = [r["model"] for r in results]
    maes = [r["mae"] for r in results]

    plt.figure()
    plt.bar(models, maes)
    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.title("Test MAE by Model")
    plt.grid(axis="y")

    plt.savefig(args.out_path)
    print("Saved plot to", args.out_path)


if __name__ == "__main__":
    main()
