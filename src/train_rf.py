import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import scienceplots
from sklearn.ensemble import RandomForestRegressor
import wandb
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.style.use('science')

DATA_PATH = "../data/processed_actuator_sysid_dataset.npz"
SCALER_PATH = "../models/scalers.pkl"
# "Random-Forest-Raw", "Random-Forest-Manual", "Random-Forest-PCA"
RUN_NAME = "Random-Forest-Manual"
MODEL_SAVE_PATH = f"../models/{RUN_NAME.lower()}.pkl"

def ema_filter(signal, gain=0.2):
    """
    Applies the Exponential Moving Average (IIR Low-Pass) filter.
    Formula: y[n] = (x[n] * gain) + (y[n-1] * (1 - gain))
    Using scipy's lfilter for fast, vectorized C-level execution.
    """
    b = [gain]
    a = [1, -(1 - gain)]
    # lfilter requires the 1D array, applies the difference equation instantly
    return lfilter(b, a, signal)

def extract_rf_features(trajectory, dt=0.02, theta_gain=0.2, omega_gain=0.2, plot=False):
    """
    Extracts 24 physically meaningful features from a 600x4 trajectory.
    trajectory shape: (600, 4) ->[theta, omega, cos_theta, sin_theta]
    """
    # 1. Separate and Filter the signals
    theta_raw = trajectory[:, 0]
    omega_raw = trajectory[:, 1]
    cos_t     = trajectory[:, 2]
    sin_t     = trajectory[:, 3]
    
    theta = ema_filter(theta_raw, gain=theta_gain)
    omega = ema_filter(omega_raw, gain=omega_gain)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(theta_raw, label=r"Raw $\theta$", alpha=0.6)
        plt.plot(theta, label=r"Filtered $\theta$", linewidth=2)
        plt.title(r"$\theta$ Filtering " + f"(gain={theta_gain})")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(omega_raw, label=r"Raw $\omega$", alpha=0.6)
        plt.plot(omega, label=r"Filtered $\omega$", linewidth=2)
        plt.title(r"$\omega$ Filtering " + f"(gain={omega_gain})")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    features =[]

    # --- BLOCK 1: Statistical Features (16 features) ---
    for sig in [theta, omega, cos_t, sin_t]:
        features.extend([
            np.mean(sig),
            np.std(sig),
            np.max(sig),
            np.min(sig)
        ])
        
    # --- BLOCK 2: Zero-Crossing Rate (1 feature) ---
    # How many times does the velocity cross 0? (Proxy for underdamped oscillation)
    zero_crossings = ((omega[:-1] * omega[1:]) < 0).sum()
    features.append(zero_crossings)

    # --- BLOCK 3: Frequency Domain (2 features) ---
    # Apply FFT to the raw theta (FFT naturally separates noise into high frequencies)
    fft_magnitudes = np.abs(np.fft.rfft(theta_raw))
    # Exclude the DC component (index 0) to find the actual oscillation frequency
    dominant_freq_idx = np.argmax(fft_magnitudes[1:]) + 1 
    fft_energy = np.sum(fft_magnitudes)
    
    features.extend([dominant_freq_idx, fft_energy])

    # --- BLOCK 4: Control Theory Features (5 features) ---
    # Calculated on the FILTERED theta to prevent noise spikes from ruining indices
    
    # 1. Steady-state value
    y_ss = np.mean(theta[-50:])
    
    # 2. Maximum Peak & Peak Time
    peak_idx = np.argmax(np.abs(theta))
    peak_time = peak_idx * dt
    max_val = theta[peak_idx]
    
    # 3. Maximum Peak Overshoot
    overshoot = np.abs(max_val - y_ss)
    
    # 4. Rise Time (10% to 90% of max_val)
    try:
        idx_10 = np.where(np.abs(theta) >= 0.1 * np.abs(max_val))[0][0]
        idx_90 = np.where(np.abs(theta) >= 0.9 * np.abs(max_val))[0][0]
        rise_time = (idx_90 - idx_10) * dt
    except IndexError:
        rise_time = 0.0 # Fallback for critically damped/flat signals
        
    # 5. Settling Time (Time after which the signal stays within 5% of max_val from y_ss)
    tolerance = 0.05 * np.abs(max_val)
    # Find all indices where the signal is OUTSIDE the tolerance
    out_of_bounds = np.where(np.abs(theta - y_ss) > tolerance)[0]
    
    if len(out_of_bounds) > 0:
        settling_time = out_of_bounds[-1] * dt
    else:
        settling_time = 0.0 # It never left the tolerance band

    features.extend([y_ss, peak_time, overshoot, rise_time, settling_time])

    # Total Features: 16 + 1 + 2 + 5 = 24
    return np.array(features)

def compute_rf_predictions(rf, X):
    # Get predictions from all individual trees
    tree_preds = np.array([tree.predict(X) for tree in rf.estimators_]) # Shape: (n_estimators, n_samples)
    
    # Calculate mean (which is what rf.predict(X) returns) and variance
    mu = np.mean(tree_preds, axis=0) # Shape: (n_samples, n_targets)
    var = np.var(tree_preds, axis=0) # Shape: (n_samples, n_targets) - This is sigma^2
    
    return mu, var

def calculate_nll(y_true, mu, var):
    eps = 1e-6
    var_safe = np.clip(var, eps, None)
    # Gaussian NLL: 0.5 * log(2*pi*var) + (y - mu)^2 / (2 * var)
    nll = 0.5 * np.log(2 * np.pi * var_safe) + 0.5 * ((y_true - mu)**2 / var_safe)
    return np.mean(nll)

def apply_pca_to_raw(X_train, X_val, n_components=30):
    """
    X_train/X_val shape: (N, 600, 4)
    Returns: (N, n_components)
    """
    # 1. Scale the data
    # Data was already scaled in preprocessing
    
    # 2. Flatten the data (600x4 -> 2400)
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]
    X_train_flat = X_train.reshape(N_train, -1)
    X_val_flat = X_val.reshape(N_val, -1)
    
    # 3. Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_val_pca = pca.transform(X_val_flat)
    
    # Save the fitted PCA model
    pca_path = "../models/pca_model.pkl"
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"  --> Saved PCA transformer to {pca_path}")
    
    # 4. Check how much info we kept (For your report!)
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA reduced 2400 dims to {n_components}. Info Retained: {explained_var:.2f}%")
    
    return X_train_pca, X_val_pca

def main():
    wandb.init(
        project="NYCU-AI-Capstone-SysID",
        config={
            "max_estimators": 100,
            "step_size": 1,
            "max_depth": 20,
            "random_state": 42,
            "rf_feature_type": RUN_NAME
        },
        name=RUN_NAME
    )
    config = wandb.config

    data = np.load(DATA_PATH)
    
    X_train, X_val = data['X_train'], data['X_val']
    Y_train, Y_val = data['Y_train'], data['Y_val']
    
    # Subsample 10% for experiment
    X_train, Y_train = X_train[:len(X_train)//10], Y_train[:len(Y_train)//10]
    
    # For testing when you want to see the graphs, uncomment this block
    # idx = np.random.randint(0, len(X_train), size=1)[0]
    # features = extract_rf_features(X_train[idx], theta_gain=0.7, omega_gain=0.4, plot=True)
    # import sys
    # sys.exit(0)
    
    if RUN_NAME == "Random-Forest-Manual":
        print("Inverse transforming data to physical units for feature extraction...")
        with open(SCALER_PATH, 'rb') as f:
            scalers = pickle.load(f)
        x_scaler = scalers['x_scaler']
        
        X_train_phys = x_scaler.inverse_transform(X_train.reshape(-1, 4)).reshape(X_train.shape)
        X_val_phys = x_scaler.inverse_transform(X_val.reshape(-1, 4)).reshape(X_val.shape)
        
        print("Extracting features for training data...")
        X_train_rf = np.array([extract_rf_features(traj, theta_gain=0.7, omega_gain=0.4) for traj in tqdm(X_train_phys)])
        print("Extracting features for validation data...")
        X_val_rf   = np.array([extract_rf_features(traj, theta_gain=0.7, omega_gain=0.4) for traj in tqdm(X_val_phys)])
    elif RUN_NAME == "Random-Forest-PCA":
        print("Applying PCA to raw data...")
        X_train_rf, X_val_rf = apply_pca_to_raw(X_train, X_val, n_components=30)
    elif RUN_NAME == "Random-Forest-Raw":
        print("Using raw data...")
        N_train = X_train.shape[0]
        N_val = X_val.shape[0]
        X_train_rf = X_train.reshape(N_train, -1)
        X_val_rf = X_val.reshape(N_val, -1)
    
    rf_regressor = RandomForestRegressor(
        n_estimators=1, # Start with 1 to use warm_start
        max_depth=config.max_depth,
        warm_start=True,
        n_jobs=-1,
        random_state=config.random_state
    )
    
    best_val_nll = float('inf')
    best_val_mse = float('inf')

    print(f"Starting RF Training up to {config.max_estimators} trees...")
    epochs = config.max_estimators // config.step_size

    for epoch in range(1, epochs + 1):
        num_trees = epoch * config.step_size
        rf_regressor.n_estimators = num_trees
        
        # Train (with warm_start=True, it just adds new trees)
        rf_regressor.fit(X_train_rf, Y_train)
        
        # --- Evaluate Train ---
        train_mu, train_var = compute_rf_predictions(rf_regressor, X_train_rf)
        train_mse = mean_squared_error(Y_train, train_mu)
        train_nll = calculate_nll(Y_train, train_mu, train_var)
        
        # --- Evaluate Val ---
        val_mu, val_var = compute_rf_predictions(rf_regressor, X_val_rf)
        val_mse = mean_squared_error(Y_val, val_mu)
        val_nll = calculate_nll(Y_val, val_mu, val_var)
        
        # Update best metrics
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_val_mse = val_mse
            wandb.run.summary["best_val_nll"] = best_val_nll
            wandb.run.summary["best_val_mse"] = best_val_mse
            
            # Save the random forest model using pickle
            with open(MODEL_SAVE_PATH, 'wb') as f:
                pickle.dump(rf_regressor, f)
            print(f"  --> Saved {RUN_NAME} to {MODEL_SAVE_PATH}")
            
        wandb.log({
            "epoch": epoch,
            "n_estimators": num_trees,
            "train_mse": train_mse,
            "train_nll_loss": train_nll,
            "val_mse": val_mse,
            "val_nll_loss": val_nll
        })
        
        print(f"Epoch {epoch:02d} (Trees: {num_trees:03d}) | "
              f"Train MSE: {train_mse:.4f} | Train NLL: {train_nll:.4f} | "
              f"Val MSE: {val_mse:.4f} | Val NLL: {val_nll:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
