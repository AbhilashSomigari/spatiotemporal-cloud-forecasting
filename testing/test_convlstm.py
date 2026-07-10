from torch.utils.data import DataLoader
from src.datasets.dataset_goes import GOESSequenceDataset
from models.convlstm_model import ConvLSTMForecaster

ds = GOESSequenceDataset("processed/frames.npy", seq_len=6, start_idx=0, end_idx=6036)
loader = DataLoader(ds, batch_size=2, shuffle=False)

model = ConvLSTMForecaster(input_channels=1, hidden_channels=16)

for x, y in loader:
    out = model(x)
    print("Input x shape:", x.shape)
    print("Target y shape:", y.shape)
    print("Model output shape:", out.shape)
    break