def get_split_ranges(num_frames, seq_len, train_frac=0.70, val_frac=0.85):
    """Chronological 70/15/15-style train/val/test index ranges.

    ``num_frames`` is the length of the raw frame array. Returns start/end
    frame-index pairs suitable for ``GOESSequenceDataset(start_idx=..., end_idx=...)``.
    Splits are computed over usable sequence positions (``num_frames - seq_len``)
    and mapped back to frame indices so each split has enough context for its
    own sequences without crossing into another split's frames.
    """
    n = num_frames - seq_len
    if n <= 0:
        raise ValueError("num_frames must be greater than seq_len.")

    train_end = int(train_frac * n)
    val_end = int(val_frac * n)

    return {
        "train": (0, train_end + seq_len),
        "val": (train_end, val_end + seq_len),
        "test": (val_end, num_frames),
    }
