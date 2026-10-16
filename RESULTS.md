# Results

## Setup
- **Data:** GOES-16 ABI L2 CMIP, CONUS sector, band 13 (10.3 µm), 2024-06-18 to 2024-06-20. 864 frames at 5-minute cadence, from the public `noaa-goes16` S3 bucket.
- **Preprocessing:** `scripts/extract_frames.py --resize 128 128`, which crops the default region, clips to 180–330 K, and normalizes to [0, 1].
- **Task:** predict the next frame (5 minutes ahead) from the previous 6.
- **Split:** chronological 70/15/15, giving 129 test sequences.
- **Training:** Adam, lr 1e-4, batch 4, 20 epochs, one seed, Apple M4 (MPS). The best checkpoint is chosen by validation MSE.

## Test-split results

| Model | MAE | ≈ K | MSE | vs. persistence (MAE) |
|---|---|---|---|---|
| Persistence (last frame) | 0.00630 | 0.95 | 0.000151 | — |
| `simple_cnn --residual` | **0.00409** | **0.61** | **0.000058** | **−35.0%** (95% CI −34.5% to −35.4%) |
| `convlstm --residual` | 0.00500 | 0.75 | 0.000089 | −20.7% (95% CI −20.2% to −20.9%) |
| `simple_cnn` (absolute frame) | 0.01132 | 1.70 | 0.000417 | +80% |
| `convlstm` (absolute frame) | 0.01690 | 2.53 | 0.000803 | +168% |

The CIs come from `evaluation/bootstrap_ci.py`, a paired block bootstrap with 1-hour blocks and 5,000 resamples. Both residual models beat persistence on every one of the 129 test sequences.

## Takeaways
- At a 5-minute horizon, persistence is a very strong baseline. Models that predict the whole frame lose to it, after 5 epochs and after 20.
- Predicting the **change** from the last frame, with a zero-initialized output layer so training starts at persistence, beats it clearly.
- The simple CNN beats the ConvLSTM here, likely because 6 frames over 30 minutes give a recurrent model little to work with and the dataset is small.

## Limitations
- One 3-day window, one seed, 128×128 resolution, one-step horizon.
- Longer horizons (30–60 minutes), more data and several seeds are the next steps.

Raw logs: `outputs/run_20ep.log`, `outputs/results_20ep.json` (20 epochs), and `outputs/run_2024_170-172.log`, `outputs/results.json` (the first 5-epoch run, absolute-frame models only).
