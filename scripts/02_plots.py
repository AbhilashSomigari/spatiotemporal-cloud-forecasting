import matplotlib.pyplot as plt

# Cropped image sizes
sizes = [512, 256, 128]

# Example MAE values (replace with your actual results)
persistence_mae = [0.002089, 0.0035, 0.0054]
cnn_mae = [0.001418, 0.0042, 0.0075]
convlstm_mae = [0.0012, 0.0050, 0.0083]

plt.figure()

plt.plot(sizes, persistence_mae, marker='o', label='Persistence')
plt.plot(sizes, cnn_mae, marker='s', label='CNN')
plt.plot(sizes, convlstm_mae, marker='^', label='ConvLSTM')

plt.xlabel("Cropped Image Size")
plt.ylabel("MAE")
plt.title("MAE vs Cropped Image Size")
plt.legend()

plt.gca().invert_xaxis()  # Optional: shows 512 → 128 direction
plt.grid()

plt.show()