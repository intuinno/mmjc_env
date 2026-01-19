"""Taxi wrapper for MMJCENV.

This wrapper makes the environment behave like the TaxiNavigationEnv,
providing high-level commands (Forward, CW, CCW) and rewarding velocity tracking.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class TaxiWrapper(gym.Wrapper):
    """Wrapper to make mmjc_env behave like TaxiNavigationEnv.
    
    Observations:
        - goal: 3-dim one-hot vector (Forward, CW, CCW)
        - proprioception: from NavigationWrapper
        
    Rewards:
        - Based on matching target linear and angular velocities.
    """
    def __init__(self, env, goal_switch_interval=20):
        super().__init__(env)
        self.goal_switch_interval = goal_switch_interval
        
        # Check if wrapped env has proprioception
        if not isinstance(env.observation_space, spaces.Dict) or "proprioception" not in env.observation_space.spaces:
            raise ValueError("TaxiWrapper expects an environment with 'proprioception' in observation space (e.g. NavigationWrapper).")
            
        self.proprioception_dim = env.observation_space["proprioception"].shape[0]
        
        self.observation_space = spaces.Dict({
            "goal": spaces.Box(0, 1, shape=(3,), dtype=np.float32),
            "proprioception": spaces.Box(-np.inf, np.inf, shape=(self.proprioception_dim,), dtype=np.float32)
        })
        
        self.current_goal_idx = 0
        self.steps_since_switch = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_goal_idx = np.random.randint(3)
        self.steps_since_switch = 0
        return self._get_obs(obs), info
        
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        self.steps_since_switch += 1
        if self.steps_since_switch >= self.goal_switch_interval:
            self.current_goal_idx = np.random.randint(3)
            self.steps_since_switch = 0
            
        reward = self._calculate_reward(info)
        
        info["goal_idx"] = self.current_goal_idx
        info["goal_name"] = ["Forward", "CW", "CCW"][self.current_goal_idx]
        
        return self._get_obs(obs), reward, terminated, truncated, info
        
    def _get_obs(self, obs):
        goal = np.zeros(3, dtype=np.float32)
        goal[self.current_goal_idx] = 1.0
        return {
            "goal": goal,
            "proprioception": obs["proprioception"]
        }
        
    def _calculate_reward(self, info):
        forward_vel = 0.0
        if "velocimeter" in info:
            forward_vel = info["velocimeter"][0]
            
        angular_vel = 0.0
        if "gyro" in info:
            angular_vel = info["gyro"][2] # Z-axis
            
        reward = 0.0
        
        if self.current_goal_idx == 0: # Forward
            reward = forward_vel
        elif self.current_goal_idx == 1: # CW (Right)
            reward = -angular_vel
        elif self.current_goal_idx == 2: # CCW (Left)
            reward = angular_vel
            
        return reward

    def render(self):
        self.env.render()
        # Overlay command
        base_env = self.env.unwrapped
        if base_env.window is not None and base_env.render_mode == "human":
            font = pygame.font.SysFont("Arial", 24)
            cmd_name = ["Forward", "CW", "CCW"][self.current_goal_idx]
            text = font.render(f"Command: {cmd_name}", True, (255, 0, 0))
            # Draw at top center or somewhere visible
            base_env.window.blit(text, (base_env.window_size + 10, 50))
            pygame.display.flip()