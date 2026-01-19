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
    def __init__(
        self,
        env,
        goal_switch_interval: int = 20,
        target_forward_velocity: float = 1.0,
        target_angular_velocity: float = 0.8,
        velocity_tolerance: float = 0.4,
        penalty_scale: float = 0.3,
    ):
        super().__init__(env)
        self.goal_switch_interval = goal_switch_interval
        self.target_forward_velocity = target_forward_velocity
        self.target_angular_velocity = target_angular_velocity
        self.velocity_tolerance = velocity_tolerance
        self.penalty_scale = penalty_scale
        
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
        
        goal_names = ["FORWARD", "ROTATE_CW", "ROTATE_CCW"]
        info["current_goal"] = goal_names[self.current_goal_idx]
        
        return self._get_obs(obs), reward, terminated, truncated, info
        
    def _get_obs(self, obs):
        goal = np.zeros(3, dtype=np.float32)
        goal[self.current_goal_idx] = 1.0
        return {
            "goal": goal,
            "proprioception": obs["proprioception"]
        }
        
    def _calculate_reward(self, info):
        forward_vel = info.get("velocimeter", [0.0])[0]
        angular_vel = info.get("gyro", [0.0, 0.0, 0.0])[2]  # Z-axis

        reward = 0.0
        
        if self.current_goal_idx == 0: # Forward
            # Linear reward for forward velocity, penalize angular velocity
            reward = forward_vel  - self.penalty_scale * abs(angular_vel)
            reward *= 2.0

        elif self.current_goal_idx == 1: # CW (Right)
            # Linear reward for angular velocity, penalize forward velocity
            reward = -angular_vel - self.penalty_scale * abs(forward_vel)

        elif self.current_goal_idx == 2: # CCW (Left)
            # Linear reward for angular velocity, penalize forward velocity
            reward = angular_vel - self.penalty_scale * abs(forward_vel)
            
        return reward

    def render(self):
        self.env.render()
        # The rendering of the taxi goal is now handled by the play script.
        # The previous overlay logic here conflicted with the underlying
        # NavigationWrapper's render loop.