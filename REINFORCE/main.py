import gymnasium as gym
import numpy as np
import torch
from agent import REINFORCE


def main():
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = REINFORCE(env, device, gamma=0.99)

    agent.update(num_episodes=2000)

    #print(env.render())

    env.close()

if __name__ == "__main__":
    main()