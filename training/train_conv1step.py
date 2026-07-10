import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.dataset_goes import GOESSequenceDataset
from venv.convlstm_model import ConvLSTMForecaster

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

train_ds = GOESSequenceDataset("processed/frames.npy", seq_len=6, start_idx=0, end_idx=6036)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)

model = ConvLSTMForecaster(input_channels=1, hidden_channels=16).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

x, y = next(iter(train_loader))
x = x.to(device)
y = y.to(device)

model.train()
optimizer.zero_grad()

out = model(x)
loss = criterion(out, y)

print("Output shape:", out.shape)
print("Target shape:", y.shape)
print("Loss before backward:", loss.item())

loss.backward()
optimizer.step()

print("One ConvLSTM training step completed.")