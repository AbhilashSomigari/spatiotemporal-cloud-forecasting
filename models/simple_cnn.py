import torch
import torch.nn as nn

class SimpleCNNForecaster(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: (B, 6, 1, H, W)
        x = x.squeeze(2)   # -> (B, 6, H, W)
        return self.net(x)