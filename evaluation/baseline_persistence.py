from src.datasets.dataset_goes import GOESSequenceDataset
from torch.utils.data import DataLoader
import torch

test_ds = GOESSequenceDataset("processed/frames.npy", seq_len=6, start_idx=7342, end_idx=8632)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

total_mae = 0
total_mse = 0
count = 0

for x, y in test_loader:
    y_pred = x[:, -1, :, :, :]

    mae = torch.mean(torch.abs(y_pred - y))
    mse = torch.mean((y_pred - y) ** 2)

    total_mae += mae.item()
    total_mse += mse.item()
    count += 1

print("Final MAE:", total_mae / count)
print("Final MSE:", total_mse / count)