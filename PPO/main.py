import gym
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import torch
from agent import PPO

import torch.multiprocessing as mp


def main():
    num_envs = 4
    envs = gym.make_vec("HalfCheetah-v4", num_envs=num_envs, vectorization_mode="sync")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = PPO(envs, device, num_envs, gamma=0.99)

    agent.update(total_update_steps=2000000)

    envs.close()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

