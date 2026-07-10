import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.dataset_goes import GOESSequenceDataset
from models.simple_cnn import SimpleCNNForecaster

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# datasets
train_ds = GOESSequenceDataset("processed/frames.npy", seq_len=6, start_idx=0, end_idx=6036)
val_ds   = GOESSequenceDataset("processed/frames.npy", seq_len=6, start_idx=6036, end_idx=7325)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False)

model = SimpleCNNForecaster().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 3

for epoch in range(epochs):
    model.train()
    train_loss = 0
    count = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        count += 1

        if count == 300:   # limit steps (important for Mac)
            break

    train_loss /= count

    # 🔵 validation
    model.eval()
    val_loss = 0
    vcount = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = criterion(out, y)

            val_loss += loss.item()
            vcount += 1

            if vcount == 200:
                break

    val_loss /= vcount

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Val Loss:   {val_loss:.6f}")