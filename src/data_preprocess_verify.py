import numpy as np
import matplotlib.pyplot as plt
import pickle
import scienceplots

plt.style.use('science')
# PROCESSED_DATA_PATH = "../data/processed_actuator_sysid_dataset.npz"
# If you want to skip the noise/quantization step and just use the raw data with engineered features, use this path instead:
PROCESSED_DATA_PATH = "../data/processed_clean_actuator_sysid_dataset.npz"
SCALER_PATH = "../models/scalers.pkl"

def verify():
    # 1. Load the processed data
    
    data = np.load(PROCESSED_DATA_PATH)
    X_train, Y_train = data['X_train'], data['Y_train']
    X_test,  Y_test  = data['X_test'],  data['Y_test']
    X_ood,   Y_ood   = data['X_ood'],   data['Y_ood']

    print("--- Statistical Verification ---")
    # Check Y-Scaling (should be inside [-1, 1])
    print(f"Y_train Range: [{Y_train.min():.4f}, {Y_train.max():.4f}] (Expected: ~[-0.96, 0.96])")
    print(f"Y_ood Range:   [{Y_ood.min():.4f}, {Y_ood.max():.4f}]   (Expected: includes values < -0.96 or > 0.96)")

    # Check X-Standardization (Train should be Mean 0, Std 1)
    print(f"X_train Mean:  {np.mean(X_train, axis=(0,1))}") # Should be ~0
    print(f"X_train Std:   {np.std(X_train, axis=(0,1))}")  # Should be ~1

    # 2. Visual Noise Check
    # Let's look at one trajectory to see the 'fuzz' from your quantization/noise
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(X_train[0, :500, 0], label=r"Noisy $\theta$ (Scaled)")
    plt.title(r"Noisy $\omega$ (Standardized)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(X_train[0, :500, 1], color='orange', label=r"Noisy $\omega$ (Scaled)")
    plt.title(r"Noisy $\omega$ (Standardized)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # 3. Target Distribution Check
    plt.figure(figsize=(10, 4))
    plt.hist(Y_train[:, 0], bins=50, alpha=0.5, label='In-Distribution (Train)')
    plt.hist(Y_ood[:, 0], bins=50, alpha=0.5, label='Out-of-Distribution (OOD)')
    plt.axvline(-0.96, color='red', linestyle='--', label='Theoretical ID Boundary')
    plt.axvline(0.96, color='red', linestyle='--')
    plt.title("Damping Target Distribution (Scaled Space)")
    plt.xlabel("Scaled Value [-1, 1]")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    verify()