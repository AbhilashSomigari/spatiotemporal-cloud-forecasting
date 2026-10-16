import torch.nn as nn


class SimpleCNNForecaster(nn.Module):
    """Stacks the T input frames as channels and predicts the next frame.

    With ``residual=True`` the network predicts the change from the last input
    frame instead of the frame itself. The output layer starts at zero, so an
    untrained model equals the persistence baseline.
    """

    def __init__(self, in_channels=6, residual=False):
        super().__init__()
        self.residual = residual
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Identity() if residual else nn.Sigmoid(),  # frames are normalized to [0, 1]
        )
        if residual:
            nn.init.zeros_(self.net[6].weight)
            nn.init.zeros_(self.net[6].bias)

    def forward(self, x):
        # x: (B, T, H, W)
        out = self.net(x)
        if self.residual:
            return (x[:, -1:] + out).clamp(0, 1)
        return out
