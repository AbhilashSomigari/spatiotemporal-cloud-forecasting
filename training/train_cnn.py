import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.dataset_goes import GOESSequenceDataset
from models.simple_cnn import SimpleCNNForecaster

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# dataset
train_ds = GOESSequenceDataset("processed/frames.npy", seq_len=6, start_idx=0, end_idx=6036)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

# model
model = SimpleCNNForecaster().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# training loop (small)
epochs = 1

for epoch in range(epochs):
    total_loss = 0
    count = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

        # print every 50 steps
        if count % 50 == 0:
            print(f"Step {count}, Loss: {loss.item():.6f}")

        #  stop early (important for now)
        if count == 200:
            break

    print("Epoch avg loss:", total_loss / count)

    torch.save(model.state_dict(), "simple_cnn.pth")
    print("Model saved.")