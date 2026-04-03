import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from training_models import CNNLSTMModel
import pickle

# --- SETTINGS ---
DATA_PATH = "../data/processed_actuator_sysid_dataset.npz"
CHKPT_PATH = "../models/best_sysid_model.pth"
DEVICE = torch.device("cpu") # Use CPU for easier plotting

CONFIG = {
    "in_channels": 4,
    "learning_rate": 1e-4,
    "cnn1_dims": 128,
    "cnn2_dims": 64,
    "lstm_dims": 64,
    "weight_decay": 1e-3,
    "clip_value": 5.0
}

def get_internals(model, x):
    """ Manually run forward pass to grab intermediate tensors """
    model.eval()
    with torch.no_grad():
        # 1. Grab Raw CNN outputs (Pre-activation vs Post-activation)
        raw_low = model.cnn1_low_freq(x)
        act_low = model.bn1_low_freq(torch.nn.functional.mish(raw_low))
        
        raw_mid = model.cnn1_mid_freq(x)
        act_mid = model.bn1_mid_freq(torch.nn.functional.mish(raw_mid))
        
        raw_high = model.cnn1_high_freq(x)
        act_high = model.bn1_high_freq(torch.nn.functional.mish(raw_high))
        
        # 2. Grab the Mixer Output (CNN2)
        merged = torch.cat([act_low, act_mid, act_high], dim=1)
        mixer = model.bn2(torch.nn.functional.mish(model.cnn2(merged)))
        
        # 3. Grab Pooled state
        pooled = model.pool(mixer)
        
    return {
        "low": act_low.squeeze().numpy(), 
        "mid": act_mid.squeeze().numpy(), 
        "high": act_high.squeeze().numpy(),
        "mixer": mixer.squeeze().numpy(),
        "pooled": pooled.squeeze().numpy()
    }

def main():
    # 1. Load Model & Data
    model = CNNLSTMModel(CONFIG, n_params=3, chkpt_file_pth=CHKPT_PATH, device=DEVICE)
    model.load_model()
    
    data = np.load(DATA_PATH)
    X_test = data['X_test']
    
    # Pick a sample (Index 0)
    sample_idx = 0
    # Shape: (1, 4, 600)
    x_input = torch.tensor(X_test[sample_idx:sample_idx+1], dtype=torch.float32).permute(0, 2, 1)
    
    internals = get_internals(model, x_input)
    
    # 2. VISUALIZATION
    plt.figure(figsize=(16, 12))
    
    # Plot Original Signal (Theta)
    plt.subplot(5, 1, 1)
    plt.plot(X_test[sample_idx, :, 0], color='black', label="Raw Theta (Input)")
    plt.title("Original Input Trajectory (600 steps)")
    plt.legend()
    
    # Plot High Freq Features (Heatmap)
    # We show only first 32 channels of the 128 to keep it readable
    plt.subplot(5, 1, 2)
    sns.heatmap(internals['high'][:32, :], cmap='viridis', cbar=False)
    plt.ylabel(f"High Freq\n(K{model.cnn1_high_freq.kernel_size[0]}, D{model.cnn1_high_freq.dilation[0]})")
    
    # Plot Mid Freq Features
    plt.subplot(5, 1, 3)
    sns.heatmap(internals['mid'][:32, :], cmap='viridis', cbar=False)
    plt.ylabel(f"Mid Freq\n(K{model.cnn1_mid_freq.kernel_size[0]}, D{model.cnn1_mid_freq.dilation[0]})")
    
    # Plot Low Freq Features
    plt.subplot(5, 1, 4)
    sns.heatmap(internals['low'][:32, :], cmap='viridis', cbar=False)
    plt.ylabel(f"Low Freq\n(K{model.cnn1_low_freq.kernel_size[0]}, D{model.cnn1_low_freq.dilation[0]})")
    
    # Plot Pooled Output (Downsampled)
    plt.subplot(5, 1, 5)
    sns.heatmap(internals['pooled'][:32, :], cmap='magma', cbar=False)
    plt.ylabel("Mixed/Pooled\n(300 steps)")
    
    plt.tight_layout()
    plt.savefig("../figures/internal_activations.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()