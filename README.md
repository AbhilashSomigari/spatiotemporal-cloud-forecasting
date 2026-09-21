# Spatiotemporal Cloud Forecasting

Next-frame forecasting of cloud movement from GOES satellite imagery. Given a
sequence of past infrared brightness-temperature frames, predict the frame
that follows. Includes a naive persistence baseline, a CNN that stacks input
frames as channels, and a ConvLSTM that models the sequence recurrently.

## Project layout

```
src/datasets/dataset_goes.py   GOESSequenceDataset: sliding-window (x, y) pairs over a frame stack
src/utils/splits.py            Chronological train/val/test split helper
models/simple_cnn.py           SimpleCNNForecaster
models/convlstm_model.py       ConvLSTMForecaster
scripts/extract_frames.py      Raw .nc -> cropped/normalized (N, H, W) frame stack (.npy)
scripts/check_frames.py        Sanity-check a processed frame stack
scripts/visualize_crop.py      Save sample original/cropped/resized preview images
scripts/plot_results.py        Plot MAE per model from outputs/results.json
training/train.py              Train simple_cnn or convlstm, checkpoint on best val MSE
evaluation/evaluate.py         Evaluate persistence / simple_cnn / convlstm on a split
tests/test_models.py           Shape/unit tests using synthetic data (no raw data needed)
```

Raw `.nc` files, processed `.npy` frame stacks, and trained `.pth` checkpoints
are all gitignored (`data/`, `processed/`, `*.npy`, `*.pth`) — this repo holds
code only.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

1. **Extract frames** from raw GOES `.nc` files into a normalized frame stack:

   ```bash
   python scripts/extract_frames.py --raw-dir data/raw --out-path processed/frames.npy
   ```

   Frames are cropped to a fixed region, brightness temperature is clipped to
   a fixed Kelvin range (default 180-330K) and normalized to [0, 1]. A fixed
   range (rather than per-frame min/max) keeps values comparable across time,
   which matters for a forecasting model.

2. **Sanity-check** the output:

   ```bash
   python scripts/check_frames.py --frames-path processed/frames.npy
   ```

3. **Train** a model (splits the frame stack 70/15/15 into train/val/test
   internally):

   ```bash
   python -m training.train --model simple_cnn --frames-path processed/frames.npy
   python -m training.train --model convlstm --frames-path processed/frames.npy --hidden-dim 32
   ```

   Checkpoints save to `models/<model>.pth` (best val MSE), and final test
   metrics are appended to `outputs/results.json`.

4. **Evaluate** any model (or the persistence baseline) on a given split:

   ```bash
   python -m evaluation.evaluate --model persistence --frames-path processed/frames.npy
   python -m evaluation.evaluate --model simple_cnn --frames-path processed/frames.npy
   ```

5. **Plot** MAE across whatever's in `outputs/results.json`:

   ```bash
   python scripts/plot_results.py
   ```

## Tests

```bash
pytest tests/
```

Tests use synthetic random tensors/arrays, so they run without any raw
satellite data.
