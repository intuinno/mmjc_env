import gymnasium
import mmjc_env

env = gymnasium.make("mmjc-easy", render_mode="human")

obs, info = env.reset()
done = False
total_reward = 0.0
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    if done:
        obs, info = env.reset()
print(f"Episode {i + 1}: Total Reward = {total_reward}")


env.close()
