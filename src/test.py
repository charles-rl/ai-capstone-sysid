import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from training_models import CNNLSTMModel
import scienceplots

plt.style.use('science')

# --- CONFIGURATION ---
DATA_PATH = "../data/processed_actuator_sysid_dataset.npz"
SCALER_PATH = "../models/scalers.pkl"
CHKPT_PATH = "../models/best_sysid_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "in_channels": 4,
    "learning_rate": 1e-4,
    "cnn1_dims": 128,
    "cnn2_dims": 64,
    "lstm_dims": 64,
    "weight_decay": 1e-3,
    "clip_value": 5.0
}

class SysIDDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y_data, dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

def evaluate_set(model, loader, name):
    model.eval()
    all_mus = []
    all_sigmas = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            mu, sigma = model(x)
            all_mus.append(mu.cpu().numpy())
            all_sigmas.append(sigma.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    return np.vstack(all_mus), np.vstack(all_sigmas), np.vstack(all_targets)

def main():
    # 1. Load Data and Scalers
    print("Loading data and scalers...")
    data = np.load(DATA_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)
    y_scaler = scalers['y_scaler']

    # 2. Setup DataLoaders
    test_loader = DataLoader(SysIDDataset(data['X_test'], data['Y_test']), batch_size=256)
    ood_loader = DataLoader(SysIDDataset(data['X_ood'], data['Y_ood']), batch_size=256)

    # 3. Load Model
    model = CNNLSTMModel(config=CONFIG, n_params=3, chkpt_file_pth=CHKPT_PATH, device=DEVICE)
    model.load_model()
    print("Model loaded successfully.")

    # 4. Run Evaluation
    print("Evaluating Test Set (ID)...")
    mu_id, sigma_id, y_id = evaluate_set(model, test_loader, "ID")
    
    print("Evaluating OOD Set...")
    mu_ood, sigma_ood, y_ood = evaluate_set(model, ood_loader, "OOD")

    # 5. Inverse Transform (Back to physical units: Damping, Friction, Armature)
    # Note: We only inverse transform the Means and Targets, Sigmas stay in scaled space for now
    y_id_phys = y_scaler.inverse_transform(y_id)
    mu_id_phys = y_scaler.inverse_transform(mu_id)
    
    y_ood_phys = y_scaler.inverse_transform(y_ood)
    mu_ood_phys = y_scaler.inverse_transform(mu_ood)

    # 6. CALCULATE FINAL METRICS
    mse_id = np.mean((y_id - mu_id)**2)
    mse_ood = np.mean((y_ood - mu_ood)**2)
    
    avg_sigma_id = np.mean(sigma_id)
    avg_sigma_ood = np.mean(sigma_ood)

    print("\n" + "="*30)
    print(f"RESULTS: IN-DISTRIBUTION (TEST)")
    print(f"  MSE (Scaled): {mse_id:.6f}")
    print(f"  Avg Uncertainty (Sigma): {avg_sigma_id:.6f}")
    
    print("-" * 30)
    print(f"RESULTS: OUT-OF-DISTRIBUTION (OOD)")
    print(f"  MSE (Scaled): {mse_ood:.6f} ({mse_ood/mse_id:.1f}x higher error)")
    print(f"  Avg Uncertainty (Sigma): {avg_sigma_ood:.6f} ({avg_sigma_ood/avg_sigma_id:.1f}x higher uncertainty)")
    print("="*30)

    # 7. VISUALIZATION
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Graph 1: Ground Truth vs Prediction (Parity Plot)
    # Let's focus on Armature (Index 2)
    ax1.scatter(y_id_phys[:, 2], mu_id_phys[:, 2], alpha=0.3, label='Test (ID)', color='blue')
    ax1.scatter(y_ood_phys[:, 2], mu_ood_phys[:, 2], alpha=0.3, label='OOD', color='red')
    ax1.plot([0, 1.0], [0, 1.0], 'k--', label='Perfect Prediction')
    ax1.set_xlabel("Ground Truth Armature")
    ax1.set_ylabel(r"Predicted Mean ($\mu$)")
    ax1.set_title("Regression Accuracy: ID vs OOD")
    ax1.legend()

    # Graph 2: Uncertainty Distribution (Histogram)
    ax2.hist(sigma_id.flatten(), bins=50, alpha=0.5, label='ID Uncertainty', color='blue', density=True)
    ax2.hist(sigma_ood.flatten(), bins=50, alpha=0.5, label='OOD Uncertainty', color='red', density=True)
    ax2.set_xlabel(r"Predicted Standard Deviation ($\sigma$)")
    ax2.set_ylabel("Density (Log Scale)")
    ax2.set_yscale('log')
    ax2.set_title("Bayesian: Uncertainty Shift")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("../figures/test_results_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
