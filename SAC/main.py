import gym
import gymnasium as gym
import numpy as np
import torch
from agent import SAC


def main():
    env = gym.make("HalfCheetah-v4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SAC(env, device, gamma=0.99)

    agent.update(total_update_steps=2000000)

    env.close()

if __name__ == "__main__":
    main()