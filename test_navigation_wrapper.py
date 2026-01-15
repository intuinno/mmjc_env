import gymnasium
import mmjc_env
from mmjc_env.wrappers import NavigationWrapper

# Create base environment
env = gymnasium.make("mmjc-low-navigation", render_mode="human")

# Wrap with NavigationWrapper (dense_reward=True by default)
env = NavigationWrapper(env, heading_bins=12, dense_reward=True)

obs, info = env.reset()
print(f"Observation keys: {obs.keys()}")
print(f"Goal position: {info['goal_pos']}")
print(f"Heading shape: {obs['heading'].shape}")
print(f"Goal pos x shape: {obs['goal_pos_x'].shape}")
print(f"Proprioception shape: {obs['proprioception'].shape}")

done = False
total_reward = 0.0
step_count = 0
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    step_count += 1

    if step_count % 100 == 0:
        print(f"Step {step_count}, Reward: {reward:.4f}, Total: {total_reward:.4f}")

    if done:
        print(f"Episode finished: Total Reward = {total_reward:.4f}")
        obs, info = env.reset()
        print(f"New goal position: {info['goal_pos']}")
        total_reward = 0.0
        step_count = 0

env.close()
