import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from training_models import CNNLSTMModel
import scienceplots
from train_rf import compute_rf_predictions
from tqdm import tqdm

plt.style.use('science')

# --- CONFIGURATION ---
DATA_PATH = "../data/processed_actuator_sysid_dataset.npz"
SCALER_PATH = "../models/scalers.pkl"
CHKPT_PATH = "../models/best_sysid_model.pth"
RF_RAW_CHKPT_PATH = "../models/random-forest-raw.pkl"
RF_MANUAL_CHKPT_PATH = "../models/random-forest-manual.pkl"
RF_PCA_CHKPT_PATH = "../models/random-forest-pca.pkl"
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

# Add a toggle for which model you want to evaluate
# Options: "cnn-lstm", "rf-raw", "rf-manual", "rf-pca"
EVAL_MODEL = "cnn-lstm"

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

def evaluate_rf_set(model, x_data, y_data, model_type, name, x_train=None, x_scaler=None, pca=None):
    print(f"Preparing data for {model_type} ({name})...")
    if model_type == "rf-raw":
        X_features = x_data.reshape(x_data.shape[0], -1)
    elif model_type == "rf-manual":
        from train_rf import extract_rf_features
        if x_scaler is None:
            raise ValueError("x_scaler must be provided to inverse transform data for rf-manual")
        print("Inverse transforming data to physical units for feature extraction...")
        x_data_phys = x_scaler.inverse_transform(x_data.reshape(-1, 4)).reshape(x_data.shape)
        X_features = np.array([extract_rf_features(traj, theta_gain=0.7, omega_gain=0.4) for traj in tqdm(x_data_phys)])
    elif model_type == "rf-pca":
        if pca is None:
            raise ValueError("Fitted PCA model must be provided for 'rf-pca' model type.")
        print("Applying learned PCA transformation...")
        x_data_flat = x_data.reshape(x_data.shape[0], -1)
        X_features = pca.transform(x_data_flat)
    else:
        raise ValueError(f"Unknown RF type: {model_type}")
        
    print(f"Running predictions on {name} data...")
    mu, var = compute_rf_predictions(model, X_features)
    sigma = np.sqrt(var) # convert variance back to standard deviation for plotting
    return mu, sigma, y_data

def main():
    # 1. Load Data and Scalers
    print("Loading data and scalers...")
    data = np.load(DATA_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)
    y_scaler = scalers['y_scaler']
    x_scaler = scalers['x_scaler']

    # 2. Setup DataLoaders
    # PyTorch DataLoaders
    test_loader = DataLoader(SysIDDataset(data['X_test'], data['Y_test']), batch_size=256)
    ood_loader = DataLoader(SysIDDataset(data['X_ood'], data['Y_ood']), batch_size=256)

    # 3. Load Models and Run Evaluation
    if EVAL_MODEL == "cnn-lstm":
        print("Loading CNN-LSTM Model...")
        model = CNNLSTMModel(config=CONFIG, n_params=3, chkpt_file_pth=CHKPT_PATH, device=DEVICE)
        model.load_model()
        print("Model loaded successfully.")
        
        print("Evaluating Test Set (ID)...")
        mu_id, sigma_id, y_id = evaluate_set(model, test_loader, "ID")
        
        print("Evaluating OOD Set...")
        mu_ood, sigma_ood, y_ood = evaluate_set(model, ood_loader, "OOD")
    
    elif EVAL_MODEL in ["rf-raw", "rf-manual", "rf-pca"]:
        if EVAL_MODEL == "rf-raw":
            rf_path = RF_RAW_CHKPT_PATH
        elif EVAL_MODEL == "rf-manual":
            rf_path = RF_MANUAL_CHKPT_PATH
        elif EVAL_MODEL == "rf-pca":
            rf_path = RF_PCA_CHKPT_PATH
            
        print(f"Loading {EVAL_MODEL} model from {rf_path}...")
        with open(rf_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
        
        if EVAL_MODEL == "rf-manual":
            # These names must match the order in extract_rf_features in train_rf.py
            feature_names = [
                "theta_mean", "theta_std", "theta_max", "theta_min",
                "omega_mean", "omega_std", "omega_max", "omega_min",
                "cos_t_mean", "cos_t_std", "cos_t_max", "cos_t_min",
                "sin_t_mean", "sin_t_std", "sin_t_max", "sin_t_min",
                "zero_crossing_rate", 
                "dominant_freq_idx", "fft_energy",
                "steady_state", "peak_time", "overshoot", "rise_time", "settling_time"
            ]
            
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            print("\n" + "="*45)
            print(f"{'RANK':4s} | {'FEATURE NAME':20s} | {'IMPORTANCE'}")
            print("-" * 45)
            for f in range(len(feature_names)):
                idx = indices[f]
                print(f"{f+1:4d} | {feature_names[idx]:20s} | {importances[idx]:.4f}")
            print("="*45 + "\n")
        
        # Load PCA model if necessary
        pca_model = None
        if EVAL_MODEL == "rf-pca":
            pca_path = "../models/pca_model.pkl"
            print(f"Loading PCA transformer from {pca_path}...")
            with open(pca_path, 'rb') as f:
                pca_model = pickle.load(f)
        
        print(f"Evaluating Test Set (ID) for {EVAL_MODEL}...")
        mu_id, sigma_id, y_id = evaluate_rf_set(model, data['X_test'], data['Y_test'], EVAL_MODEL, "ID", x_train=data['X_train'], x_scaler=x_scaler, pca=pca_model)
        
        print(f"Evaluating OOD Set for {EVAL_MODEL}...")
        mu_ood, sigma_ood, y_ood = evaluate_rf_set(model, data['X_ood'], data['Y_ood'], EVAL_MODEL, "OOD", x_train=data['X_train'], x_scaler=x_scaler, pca=pca_model)
        
    else:
        raise ValueError(f"Unknown EVAL_MODEL {EVAL_MODEL}")

    # 5. Inverse Transform (Back to physical units: Damping, Friction, Armature)
    # Inverse transform the Means and Targets
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
    # Let's focus on Damping (Index 0)
    ax1.scatter(y_id_phys[:, 0], mu_id_phys[:, 0], alpha=0.3, label='Test (ID)', color='blue')
    ax1.scatter(y_ood_phys[:, 0], mu_ood_phys[:, 0], alpha=0.3, label='OOD', color='red')
    ax1.plot([0.5, 5.0], [0.5, 5.0], 'k--', label='Perfect Prediction')
    ax1.set_xlabel("Ground Truth Damping")
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
    plt.savefig(f"../figures/test_results_comparison_{EVAL_MODEL}.png")
    plt.show()

if __name__ == "__main__":
    main()
