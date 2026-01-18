"""Test script for TaxiNavigationEnv.

This script tests the TaxiNavigationEnv environment with:
- Human rendering showing top-down view with goal indicators
- Random actions to observe goal switching behavior
- Reward tracking for each goal type
"""

import gymnasium
import mmjc_env
from mmjc_env.envs import GoalType

# Create environment with human rendering
env = gymnasium.make(
    "TaxiNavigation-v0",
    render_mode="human",
    goal_switch_interval=50,  # Switch goal every 50 steps
    target_forward_velocity=1.0,
    target_angular_velocity=0.5,
)

print("TaxiNavigationEnv Test")
print("=" * 50)
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print("=" * 50)

obs, info = env.reset()
print(f"\nInitial observation:")
print(f"  Goal (one-hot): {obs['goal']}")
print(f"  Proprioception shape: {obs['proprioception'].shape}")
print(f"  Current goal: {info['current_goal']}")

total_reward = 0.0
step_count = 0
goal_rewards = {GoalType.FORWARD: 0.0, GoalType.ROTATE_CW: 0.0, GoalType.ROTATE_CCW: 0.0}

try:
    while True:
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        # Track reward by goal type
        goal_idx = obs['goal'].argmax()
        goal_type = GoalType(goal_idx)
        goal_rewards[goal_type] += reward

        # Print info every 50 steps
        if step_count % 50 == 0:
            print(f"\nStep {step_count}:")
            print(f"  Goal: {info['current_goal']}")
            print(f"  Forward vel: {info['forward_velocity']:.2f} m/s")
            print(f"  Angular vel: {info['angular_velocity']:.2f} rad/s")
            print(f"  Step reward: {reward:.4f}")
            print(f"  Total reward: {total_reward:.2f}")

        if terminated or truncated:
            print(f"\n{'=' * 50}")
            print(f"Episode finished at step {step_count}")
            print(f"Total reward: {total_reward:.2f}")
            print(f"\nRewards by goal type:")
            for goal_type, goal_reward in goal_rewards.items():
                print(f"  {goal_type.name}: {goal_reward:.2f}")
            print(f"{'=' * 50}")

            # Reset for next episode
            obs, info = env.reset()
            total_reward = 0.0
            step_count = 0
            goal_rewards = {GoalType.FORWARD: 0.0, GoalType.ROTATE_CW: 0.0, GoalType.ROTATE_CCW: 0.0}

except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    env.close()
    print("Environment closed")
