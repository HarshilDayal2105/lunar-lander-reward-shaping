import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3")

# Initialize PPO with the standard two-layer 64-unit neural network
model = PPO("MlpPolicy", env, verbose=1)

# Train for 1,000,000 steps (~15-25 minutes on a laptop)
model.learn(total_timesteps=1_000_000)

# Save the trained weights
model.save("ppo_lunar_baseline")
env.close()
print("Baseline training complete and saved!")
