import gymnasium as gym
import numpy as np
import torch
from agent import A2C


def main():
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = A2C(env, device, gamma=0.99)

    agent.update(total_update_steps=550000)

    env.close()

if __name__ == "__main__":
    main()