import matplotlib.pyplot as plt
import numpy as np

# Data
gamma = np.array([0.0, 1.0, 3.0, 5.0, 7.0, 10.0, 12.5, 15.0, 17.5, 20.0])
mae = np.array([0.6008, 0.6012, 0.5987, 0.5992, 0.5961, 0.5960, 0.5936, 0.5931, 0.5953, 0.5968])
mae_std = np.array([0.0110, 0.0105, 0.0107, 0.0089, 0.0110, 0.0098, 0.0075, 0.0133, 0.0112, 0.0122])

# Find best MAE
best_idx = np.argmin(mae)
best_gamma = gamma[best_idx]
best_mae = mae[best_idx]

plt.figure(figsize=(7,5))

# Plot all points
plt.errorbar(gamma, mae, marker='o', capsize=4, label="MAE")

# Highlight best point with different color
plt.scatter(best_gamma, best_mae, color='red', zorder=5, label=f"Best γ = {best_gamma}")

# Annotation
plt.annotate(
    f"γ={best_gamma}, MAE={best_mae:.4f}",
    (best_gamma, best_mae),
    textcoords="offset points",
    xytext=(10,-15)
)

plt.xlabel("Gamma (γ)")
plt.ylabel("MAE")
plt.title("MAE vs Gamma (Soft Weighting)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("mlp_mae.png")