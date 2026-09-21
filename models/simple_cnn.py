import torch.nn as nn


class SimpleCNNForecaster(nn.Module):
    """Stacks the T input frames as channels and predicts the next frame."""

    def __init__(self, in_channels=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # frames are normalized to [0, 1]
        )

    def forward(self, x):
        # x: (B, T, H, W)
        return self.net(x)
