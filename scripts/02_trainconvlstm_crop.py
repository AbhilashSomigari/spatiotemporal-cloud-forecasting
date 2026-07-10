import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


FRAMES_PATH = "data/cropped_frames/frames_crop_128.npy"
MODEL_PATH = "models/convlstm_crop_128.pth"

SEQ_LEN = 6
BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-4
HIDDEN_DIM = 32


class CloudFrameDataset(Dataset):
    def __init__(self, frames, start_idx, end_idx, seq_len=6):
        self.frames = frames
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.seq_len = seq_len

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        i = self.start_idx + idx

        x = self.frames[i:i+self.seq_len]      # (6, H, W)
        y = self.frames[i+self.seq_len]        # (H, W)

        x = torch.from_numpy(np.array(x)).float()
        y = torch.from_numpy(np.array(y)).float()

        x = x.unsqueeze(1)                     # (6, 1, H, W)
        y = y.unsqueeze(0)                     # (1, H, W)

        return x, y


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.cell = ConvLSTMCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=3
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        h = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        c = torch.zeros(B, self.hidden_dim, H, W, device=x.device)

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)

        out = self.output_conv(h)
        return out


def evaluate(model, loader, device):
    model.eval()

    mae_sum = 0.0
    mse_sum = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            diff = pred - y

            mae_sum += torch.abs(diff).sum().item()
            mse_sum += (diff ** 2).sum().item()
            count += diff.numel()

    return mae_sum / count, mse_sum / count


def main():
    os.makedirs("models", exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    frames = np.load(FRAMES_PATH, mmap_mode="r")

    N = len(frames) - SEQ_LEN
    train_end = int(0.70 * N)
    val_end = int(0.85 * N)

    train_ds = CloudFrameDataset(frames, 0, train_end, SEQ_LEN)
    val_ds = CloudFrameDataset(frames, train_end, val_end, SEQ_LEN)
    test_ds = CloudFrameDataset(frames, val_end, N, SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ConvLSTMForecaster(input_dim=1, hidden_dim=HIDDEN_DIM).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mse = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % 200 == 0:
                print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(train_loader)
        val_mae, val_mse = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {avg_loss:.6f}")
        print(f"Val MAE: {val_mae:.6f}")
        print(f"Val MSE: {val_mse:.6f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), MODEL_PATH)
            print("Saved best model.")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_mae, test_mse = evaluate(model, test_loader, device)

    print("Final Test Results - ConvLSTM Crop")
    print("Test MAE:", test_mae)
    print("Test MSE:", test_mse)


if __name__ == "__main__":
    main()