import gymnasium as gym
from stable_baselines3 import PPO
from record_video import evaluate_and_record

class FixedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_shaping = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_shaping = self._state_score(obs)
        return obs, info

    def _state_score(self, obs):
        dist = (obs[0]**2 + obs[1]**2) ** 0.5
        speed = (obs[2]**2 + obs[3]**2) ** 0.5
        tilt = abs(obs[4])
        return -10.0 * (dist + speed + tilt)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_shaping = self._state_score(obs)

        delta_bonus = current_shaping - self.prev_shaping
        time_penalty = -0.05

        reward = reward + delta_bonus + time_penalty
        self.prev_shaping = current_shaping
        return obs, reward, terminated, truncated, info

env = FixedRewardWrapper(gym.make("LunarLander-v3"))
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("ppo_lunar_fixed")
env.close()

evaluate_and_record("ppo_lunar_fixed", "./videos_fixed")
