import torch
from torch.utils.data import DataLoader

from src.datasets.dataset_goes import GOESSequenceDataset
from venv.convlstm_model import ConvLSTMForecaster

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

val_ds = GOESSequenceDataset("processed/frames.npy", seq_len=6, start_idx=6036, end_idx=7325)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

model = ConvLSTMForecaster(input_channels=1, hidden_channels=16).to(device)
model.load_state_dict(torch.load("convlstm.pth", map_location=device))
model.eval()

total_mae = 0.0
total_mse = 0.0
count = 0

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)

        mae = torch.mean(torch.abs(out - y))
        mse = torch.mean((out - y) ** 2)

        total_mae += mae.item()
        total_mse += mse.item()
        count += 1

print("Validation MAE:", total_mae / count)
print("Validation MSE:", total_mse / count)