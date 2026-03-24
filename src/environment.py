import mujoco
import mujoco.viewer
import gymnasium
import numpy as np

def simp_angle(a):
    """
    Wrap angle to [-pi, pi]
    """
    return (a + np.pi) % (2 * np.pi) - np.pi

class SinglePendulumEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    FRAME_SKIP = 20
    
    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        # MuJoCo Setup
        self.model = mujoco.MjModel.from_xml_path("../simulation/scene.xml")
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.renderer = None
        self.timesteps = 0
        self.max_episode_steps = 600  # 12 seconds
        
        # Environment
        self.actuator_scale = 3.0
        # Let's keep it at 64 bits
        # raw theta (goes beyond -pi and pi), angular velocity
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.timesteps = 0
        if seed is not None:
            np.random.seed(seed)
        
        # If options are provided then update parameters otherwise use default
        if options is not None:
            self.model.dof_damping[0] = options["parameters"][0]
            self.model.dof_frictionloss[0] = options["parameters"][1]
            self.model.dof_armature[0] = options["parameters"][2]

        # Reset Physics
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = np.pi  # pole starts at the bottom (equilibrium)
        self.data.qvel[0] = 0.0
        
        return self._get_obs_info()
    
    def _get_obs_info(self):
        current_parameters = np.array([
            self.model.dof_damping[0], self.model.dof_frictionloss[0], self.model.dof_armature[0]
        ], dtype=np.float64)
        
        info = {"qpos": self.data.qpos[0], "qvel": self.data.qvel[0], "parameters": current_parameters}
        obs = np.array([self.data.qpos[0], self.data.qvel[0]], dtype=np.float64)
        
        return obs, info
    
    def step(self, action):
        self.data.ctrl = action * self.actuator_scale

        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        
        self.timesteps += 1
        
        truncated = bool(self.timesteps >= self.max_episode_steps)
        # Reward is useless for this application
        reward = 0.0

        terminated = False

        obs, info = self._get_obs_info()
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            # 1. Launch Viewer if it doesn't exist yet
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data, show_left_ui=False, show_right_ui=False)

            if self.viewer.is_running():
                self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
                self.renderer.update_scene(self.data, camera="main_cam")
            else:
                self.renderer.update_scene(self.data, camera="main_cam")

            return self.renderer.render()
        
    def close(self):
        if self.viewer is not None:
            self.viewer.close()



if __name__ == "__main__":
    # ONLY for testing
    import time
    
    env = SinglePendulumEnv(render_mode="human")
    # Test out the min and max parameters
    parameters = [[0.5, 0.2, 0], [5.0, 2.0, 1.0]]
    for param in parameters:
        obs, info = env.reset(seed=0, options={"parameters": param})
        done = False
        while not done:
            start_time = time.time() # Track start of the frame
            if 10 <= env.timesteps < 20:
                # Torque pulse positive at 200ms upto 400ms
                action = np.ones(env.action_space.shape)
            elif 510 <= env.timesteps < 520:
                # Torque pulse negative at 10.2s upto 10.4s
                action = -np.ones(env.action_space.shape)
            else:
                # let it settle
                action = np.zeros(env.action_space.shape)
            print(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()
            
            # --- SYNC PHYSICS TO REAL TIME ---
            # Calculate how much time the CPU took to do the work
            elapsed = time.time() - start_time
            # Wait for the remainder of the 0.02s timestep
            if elapsed < env.model.opt.timestep * env.FRAME_SKIP:
                time.sleep(env.model.opt.timestep * env.FRAME_SKIP - elapsed)
    env.close()    
