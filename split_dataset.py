from src.datasets.dataset_goes import GOESSequenceDataset

total_frames = 8632
seq_len = 6

train_end = int(0.70 * total_frames)
val_end = int(0.85 * total_frames)

train_ds = GOESSequenceDataset("processed/frames.npy", seq_len=seq_len, start_idx=0, end_idx=train_end)
val_ds   = GOESSequenceDataset("processed/frames.npy", seq_len=seq_len, start_idx=train_end, end_idx=val_end)
test_ds  = GOESSequenceDataset("processed/frames.npy", seq_len=seq_len, start_idx=val_end, end_idx=total_frames)

print("Train length:", len(train_ds))
print("Val length:", len(val_ds))
print("Test length:", len(test_ds))