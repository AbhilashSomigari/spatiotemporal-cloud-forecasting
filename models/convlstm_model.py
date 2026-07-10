import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        # x:      (B, C, H, W)
        # h_prev: (B, hidden, H, W)
        # c_prev: (B, hidden, H, W)

        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)      # input gate
        f = torch.sigmoid(f)      # forget gate
        o = torch.sigmoid(o)      # output gate
        g = torch.tanh(g)         # candidate cell

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c
    
class ConvLSTMForecaster(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=16, kernel_size=3):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.cell = ConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size
        )

        self.output_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        device = x.device

        h = torch.zeros(B, self.hidden_channels, H, W, device=device)
        c = torch.zeros(B, self.hidden_channels, H, W, device=device)

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)

        out = self.output_conv(h)   # (B, 1, H, W)
        return out