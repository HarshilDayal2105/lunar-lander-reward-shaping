import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
import numpy as np

def evaluate_and_record(model_path, video_folder):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda ep: ep == 0,
        name_prefix="lander"
    )

    model = PPO.load(model_path)
    scores = []

    for ep in range(10):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        scores.append(total_reward)
        print(f"Episode {ep + 1}: Score = {total_reward:.2f}")

    env.close()
    print(f"\nAverage Score over 10 attempts: {np.mean(scores):.2f}")

if __name__ == "__main__":
    evaluate_and_record("ppo_lunar_baseline", "./videos_baseline")
