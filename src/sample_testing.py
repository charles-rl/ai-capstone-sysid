import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from environment import SinglePendulumEnv
from training_models import CNNLSTMModel
from data_collection import MINMAX_PARAMS

# --- CONFIGURATION ---
SCALER_PATH = "../models/scalers.pkl"
CHKPT_PATH = "../models/best_sysid_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 50  # How many trajectories to draw from the Gaussian
MINIMUM_PARAMS = np.array([MINMAX_PARAMS["damping"][0], MINMAX_PARAMS["friction"][0], MINMAX_PARAMS["armature"][0]])

CONFIG = {
    "in_channels": 4, "learning_rate": 1e-4, "cnn1_dims": 128,
    "cnn2_dims": 64, "lstm_dims": 64, "weight_decay": 1e-3, "clip_value": 5.0
}

def get_clean_trajectory(env, params):
    """Runs a perfectly clean simulation for physical comparison."""
    obs, info = env.reset(seed=42, options={"parameters": params})
    trajectory = np.zeros((env.max_episode_steps, env.observation_space.shape[0]))
    for step in range(env.max_episode_steps):
        if 5 <= step < 15: action = np.ones(1)
        elif 305 <= step < 315: action = -np.ones(1)
        else: action = np.zeros(1)
        trajectory[step] = obs
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated: break
    return trajectory

def get_noisy_observation(env, params):
    """Generates the quantized/noisy features the AI expects."""
    traj = get_clean_trajectory(env, params)
    # Apply your specific noise/quantization logic here
    encoder_res = 4096.0
    theta_noisy = np.round(traj[:, 0] * (encoder_res / (2 * np.pi)))
    noise_ticks = 3
    theta_noisy += np.round(np.random.normal(0, noise_ticks, theta_noisy.shape))
    theta_noisy = theta_noisy * ((2 * np.pi) / encoder_res)
    
    dt = 0.02
    sigma_omega = (noise_ticks * 2 * np.pi) / (4096.0 * dt)
    # Add noise to omega (angular velocity)
    omega_noisy = traj[:, 1] + np.random.normal(0, sigma_omega, traj[:, 1].shape)
    
    X = np.zeros((env.max_episode_steps, 4))
    X[:, 0], X[:, 1], X[:, 2], X[:, 3] = theta_noisy, omega_noisy, np.cos(theta_noisy), np.sin(theta_noisy)
    return X, traj

def main():
    # 1. Setup
    with open(SCALER_PATH, 'rb') as f: scalers = pickle.load(f)
    x_scaler, y_scaler = scalers['x_scaler'], scalers['y_scaler']
    model = CNNLSTMModel(config=CONFIG, n_params=3, chkpt_file_pth=CHKPT_PATH, device=DEVICE)
    model.load_model()
    model.eval()
    env = SinglePendulumEnv(render_mode=None)

    # 2. Select Ground Truth (Change these to test ID vs OOD!)
    # ID Example: [2.49, 1.13, 0.77] | OOD Example: [0.51, 0.21, 0.01]  [4.99, 1.99, 0.99]
    true_params = [0.51, 0.21, 0.01]
    X_input, traj_true = get_noisy_observation(env, true_params)

    # 3. AI Inference
    X_scaled = x_scaler.transform(X_input).reshape(1, 600, 4)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    with torch.no_grad():
        mu_s, sigma_s = model(X_tensor)

    # 4. Plotting Initialization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Bug was found that the environment fixes the seed and I get the same trajectory every time.
    # This rng is independent from the one in the environment
    rng = np.random.RandomState(None) # 'None' means it stays truly random
    
    # 5. Stochastic Sampling Loop
    print(f"Sampling {NUM_SAMPLES} trajectories from Predicted Distribution...")
    for i in range(NUM_SAMPLES):
        # Sample D, F, A from the Gaussian N(mu, sigma)
        mu_s_cpu = mu_s.cpu().numpy()[0]
        sigma_s_cpu = sigma_s.cpu().numpy()[0]
        sample = rng.normal(mu_s_cpu, sigma_s_cpu)
        # MINMAX_PARAMS are [0.5, 0.2, 0.0] for [Damping, Friction, Armature]
        sample = np.clip(sample, y_scaler.transform(MINIMUM_PARAMS.reshape(1, -1)), None) # Ensure physical plausibility
        
        sample = y_scaler.inverse_transform(sample.reshape(1, -1))[0] # Scale back to physical units
        
        traj_sample = get_clean_trajectory(env, sample)
        
        # Plot with low alpha to create the "Uncertainty Cloud"
        ax1.plot(traj_sample[:, 0], color='red', alpha=0.15, linewidth=1, label='Sampled Trajectories' if i == 0 else None) # Only label the first one for legend
        ax2.plot(traj_sample[:, 1], color='red', alpha=0.15, linewidth=1, label='Sampled Trajectories' if i == 0 else None)

    # 6. Overlay Ground Truth
    ax1.plot(traj_true[:, 0], color='black', alpha=0.4, linewidth=2.5, label='True System Behavior')
    ax2.plot(traj_true[:, 1], color='black', alpha=0.4, linewidth=2.5, label='True System Behavior')

    # Formatting
    print(mu_s_cpu, sigma_s_cpu)
    mu_scaled = y_scaler.inverse_transform(mu_s_cpu.reshape(1, -1))[0]
    sigma_scaled = sigma_s_cpu / y_scaler.scale_  # Convert std dev from normalized space to physical units
    predicted_mean_text = r"Predicted Params $\mu$ (Scaled): " + f"{mu_scaled[0]:.2f}, {mu_scaled[1]:.2f}, {mu_scaled[2]:.2f}"
    true_mean_text = f"True Params: {true_params[0]:.2f}, {true_params[1]:.2f}, {true_params[2]:.2f}"
    predicted_sigma_text = r"Predicted $\sigma$ (Scaled): " + f"{sigma_scaled[0]:.3f}, {sigma_scaled[1]:.3f}, {sigma_scaled[2]:.3f}"

    ax1.set_title(r"Angle ($\theta$)", fontsize=12)
    ax1.set_ylabel("Radians")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.set_title(r"Angular Velocity ($\omega$)", fontsize=12)
    ax2.set_ylabel("Rad/s")
    ax2.set_xlabel("Timesteps (50Hz)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Bayesian Digital Twin Ensemble (N={NUM_SAMPLES})\nDistribution Sampling", fontsize=16, y=0.985)
    fig.text(0.5, 0.88, predicted_mean_text, ha='center', fontsize=14)
    fig.text(0.5, 0.85, predicted_sigma_text, ha='center', fontsize=14)
    fig.text(0.5, 0.82, true_mean_text, ha='center', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.84])
    plt.savefig("../figures/uncertainty_cloud_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()