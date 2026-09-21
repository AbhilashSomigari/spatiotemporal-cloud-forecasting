import numpy as np
import torch

from models.convlstm_model import ConvLSTMForecaster
from models.simple_cnn import SimpleCNNForecaster
from src.datasets.dataset_goes import GOESSequenceDataset
from src.utils.splits import get_split_ranges


def test_simple_cnn_output_shape():
    x = torch.randn(2, 6, 32, 32)
    model = SimpleCNNForecaster(in_channels=6)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)


def test_convlstm_output_shape():
    x = torch.randn(2, 6, 32, 32)
    model = ConvLSTMForecaster(input_dim=1, hidden_dim=8)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)


def test_dataset_shapes(tmp_path):
    frames = np.random.rand(20, 16, 16).astype(np.float32)
    frames_path = tmp_path / "frames.npy"
    np.save(frames_path, frames)

    ds = GOESSequenceDataset(str(frames_path), seq_len=6)
    assert len(ds) == 20 - 6

    x, y = ds[0]
    assert x.shape == (6, 16, 16)
    assert y.shape == (1, 16, 16)


def test_split_ranges_cover_all_frames():
    ranges = get_split_ranges(num_frames=100, seq_len=6)
    assert ranges["train"][0] == 0
    assert ranges["test"][1] == 100
    assert ranges["train"][1] <= ranges["val"][1] <= ranges["test"][1]
