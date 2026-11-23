# Autonomous Rocket Landing via PPO & Custom Reward Shaping

A Reinforcement Learning project built with **Gymnasium (`LunarLander-v3`)**, **Box2D**, and **Stable-Baselines3** that trains a Proximal Policy Optimization (PPO) agent to land a lunar module—and demonstrates how flawed reward functions trigger **reward hacking** and how delta-based shaping resolves it.

---

## Project Overview

In `LunarLander-v3`, the agent observes an 8-dimensional state vector at 50 Hz (x/y coordinates, horizontal/vertical velocity, tilt angle, angular velocity, and left/right leg ground contact) and selects one of four discrete thruster actions per frame:
- `0`: Do nothing
- `1`: Fire left orientation engine
- `2`: Fire main engine
- `3`: Fire right orientation engine

An episode is considered **solved** when the agent achieves a score of **200+** on the unmodified environment.

### Experiments Included
1. **Baseline Agent (`train_baseline.py`)**: Trains a standard 2-layer MLP (`64 × 64`) PPO policy for 1,000,000 timesteps using the built-in Box2D reward function.
2. **Reward Hacking (`reward_hacking.py`)**: Wraps the environment to award a static `+1.0` bonus every frame the rocket remains upright and slow (`|vy| < 0.2`, `|angle| < 0.2`). Because episodes cap at 1,000 frames, the agent learns to exploit the objective by hovering indefinitely in mid-air to farm points rather than landing.
3. **Fixed Reward Shaping (`reward_fixed.py`)**: Replaces static state bonuses with **delta-based potential shaping** (rewarding frame-over-frame improvement in distance, speed, and tilt) alongside a `-0.05` per-frame time penalty, driving fast, controlled touchdowns.

---

## Repository Structure

```text
lunar-lander-reward-shaping/
├── test_env.py            # Verifies Box2D rendering and inspects the 8-element observation space
├── train_baseline.py      # Trains baseline PPO agent for 1M timesteps
├── record_video.py        # Evaluates saved models over 10 unwrapped episodes and exports MP4s
├── reward_hacking.py      # Custom Gymnasium Wrapper demonstrating hovering reward hack
├── reward_fixed.py        # Custom Gymnasium Wrapper fixing reward hacking via delta shaping + time penalty
└── README.md
