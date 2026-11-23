import gymnasium as gym
from stable_baselines3 import PPO
from record_video import evaluate_and_record

class HoverHackWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        vertical_speed = abs(obs[3])
        tilt = abs(obs[4])

        # Flawed static bonus: +1 every frame it stays slow and upright
        if vertical_speed < 0.2 and tilt < 0.2:
            reward += 1.0

        return obs, reward, terminated, truncated, info

env = HoverHackWrapper(gym.make("LunarLander-v3"))
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("ppo_lunar_hacked")
env.close()

evaluate_and_record("ppo_lunar_hacked", "./videos_hacked")
