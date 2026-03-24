from environment import SinglePendulumEnv
import numpy as np
from multiprocessing import Pool, cpu_count
import os
from tqdm import tqdm

# TODO: include versions of packages in requirements.txt

NUM_CORES = cpu_count() - 1 # Uses all available cores (Server/Colab friendly)
SAVE_PATH = "../data/raw_actuator_sysid_dataset.npz"
NUM_EPISODES = 20_000
# Absolute minimum and maximum
MINMAX_PARAMS = {
    "damping": [0.5, 5.0],
    "friction": [0.2, 2.0],
    "armature": [0.0, 1.0]
}

def collect_one_episode(episode_idx):
    """
    This function runs in its own process.
    It creates its own environment instance.
    """
    # Create env inside the worker (MuJoCo objects aren't picklable)
    env = SinglePendulumEnv(render_mode=None)
    half_max_steps = env.max_episode_steps // 2
    
    # Randomly sample parameters for this trajectory
    params = [
        np.random.uniform(MINMAX_PARAMS["damping"][0], MINMAX_PARAMS["damping"][1]),
        np.random.uniform(MINMAX_PARAMS["friction"][0], MINMAX_PARAMS["friction"][1]),
        np.random.uniform(MINMAX_PARAMS["armature"][0], MINMAX_PARAMS["armature"][1])
    ]
    
    obs, info = env.reset(seed=episode_idx, options={"parameters": params})
    
    trajectory = np.zeros((env.max_episode_steps, env.observation_space.shape[0]))
    
    done = False
    while not done:
        # Control logic
        # Pulse after a few timesteps has passed
        if 5 <= env.timesteps < 15:
            # Torque pulse positive at 100ms upto 300ms
            action = np.ones(env.action_space.shape)
        elif half_max_steps + 5 <= env.timesteps < half_max_steps + 15:
            # Torque pulse negative at 6.1s upto 6.3s
            action = -np.ones(env.action_space.shape)
        else:
            # let it settle for about 10 seconds
            action = np.zeros(env.action_space.shape)
        
        # Record the CURRENT observation before stepping
        trajectory[env.timesteps] = obs
        
        # Step the environment
        obs_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs = obs_
    
    env.close()
    return trajectory, params


if __name__ == "__main__":
    # 1. Ensure the directory exists
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print(f"Starting data collection on {NUM_CORES} cores...")
    
    # 2. Use a Pool to run episodes in parallel
    with Pool(processes=NUM_CORES) as pool:
        results = list(tqdm(pool.imap(collect_one_episode, range(NUM_EPISODES)), 
                            total=NUM_EPISODES, 
                            desc="Collecting Trajectories"))

    # 3. Unpack results
    # results is a list of (trajectory, params) tuples
    all_trajectories = np.array([r[0] for r in results])
    all_parameters = np.array([r[1] for r in results])

    # 4. Save to compressed NPZ
    print(f"Saving dataset to {SAVE_PATH}...")
    np.savez_compressed(SAVE_PATH, 
                        trajectories=all_trajectories, 
                        parameters=all_parameters)
    
    print("Done! Dataset shape:", all_trajectories.shape)
