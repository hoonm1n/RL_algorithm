import gym
import gymnasium as gym
import numpy as np
import torch
from agent import TRPO


def main():
    env = gym.make("Hopper-v4", render_mode="rgb_array")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = TRPO(env, device, gamma=0.99)

    agent.update(total_update_steps=550000)

    env.close()

if __name__ == "__main__":
    main()