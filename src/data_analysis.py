import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import os

plt.style.use('science')
DATA_PATH = "../data/raw_actuator_sysid_dataset.npz"

def verify_dataset():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # 1. Load Data
    data = np.load(DATA_PATH)
    trajectories = data['trajectories'] # (N, timesteps, 2) -> [theta, omega]
    parameters = data['parameters']     # (N, 3) -> [damping, friction, armature]

    num_episodes = trajectories.shape[0]
    timesteps = trajectories.shape[1]

    print(f"Dataset Loaded Successfully!")
    print(f"Total Episodes: {num_episodes}")
    print(f"Timesteps per Episode: {timesteps}")
    print(f"Observation Shape: {trajectories.shape[2]} (Theta, Omega)")

    # 2. Plot Parameter Distribution (Balance Check)
    # This is a requirement for the "Discussion" section of your report
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    param_names = ['Damping', 'Friction', 'Armature']
    for i in range(3):
        axes[i].hist(parameters[:, i], bins=30, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribution of {param_names[i]}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # 3. Plot Random Trajectories (Physics Check)
    num_samples = 3
    indices = np.random.choice(num_episodes, num_samples, replace=False)

    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(indices):
        traj = trajectories[idx]
        params = parameters[idx]
        
        # Plot Theta (Position)
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.plot(traj[:, 0], label=r"$\theta$ (Angle)")
        plt.title(f"Sample {idx} | D={params[0]:.2f}, F={params[1]:.2f}, A={params[2]:.2f}")
        plt.ylabel("Radians")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot Omega (Velocity)
        plt.subplot(num_samples, 2, 2*i + 2)
        plt.plot(traj[:, 1], color='orange', label=r"$\omega$ (Angular Velocity)")
        plt.ylabel("Rad/s")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.suptitle("Raw Trajectories Visualization (Random Sampling)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def find_pure_samples():
    data = np.load(DATA_PATH)
    trajectories = data['trajectories']
    parameters = data['parameters'] # [D, F, A]

    # 1. Define our physical targets
    # We want the sample closest to the actual statistical min, median, and max
    p_min = np.min(parameters, axis=0)
    p_max = np.max(parameters, axis=0)
    p_med = np.median(parameters, axis=0)

    # 2. Normalize parameters (0 to 1) for fair distance calculation
    # This prevents 'Damping' from dominating 'Armature' due to scale differences
    norm_params = (parameters - p_min) / (p_max - p_min)
    
    target_low = [0.0, 0.0, 0.0]
    target_med = [0.5, 0.5, 0.5]
    target_high = [1.0, 1.0, 1.0]

    # 3. Calculate Euclidean Distance from our 3 targets
    # This finds the 'purest' representative of each category
    dist_low = np.linalg.norm(norm_params - target_low, axis=1)
    dist_med = np.linalg.norm(norm_params - target_med, axis=1)
    dist_high = np.linalg.norm(norm_params - target_high, axis=1)

    idx_low = np.argmin(dist_low)
    idx_med = np.argmin(dist_med)
    idx_high = np.argmin(dist_high)

    indices = [idx_low, idx_med, idx_high]
    labels = ['Low (Minimum D/F/A)', 'Balanced (Median D/F/A)', 'High (Maximum D/F/A)']
    colors = ['tab:blue', 'tab:green', 'tab:red']

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for i, idx in enumerate(indices):
        p = parameters[idx]
        t = trajectories[idx]
        
        legend_label = f"{labels[i]}\n[D:{p[0]:.2f}, F:{p[1]:.2f}, A:{p[2]:.2f}]"
        
        ax1.plot(t[:, 0], color=colors[i], linewidth=2.5, label=legend_label)
        ax2.plot(t[:, 1], color=colors[i], linewidth=2.5)

    ax1.set_title(r"$\theta$ (Angle) Comparison", fontsize=14)
    ax1.set_ylabel("Radians")
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    ax2.set_title(r"$\omega$ (Angular Velocity) Comparison", fontsize=14)
    ax2.set_ylabel("Rad/s")
    ax2.set_xlabel("Timesteps (50Hz)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    find_pure_samples()  # to find the low, mid, high
    # verify_dataset()  # to sample from the dataset and see the distribution of parameters as well as their plots
    