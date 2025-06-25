import gymnasium as gym
import numpy as np
import torch
from agent import DQN


def main():
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQN(env, device, gamma=0.99)

    Q = agent.update(num_episodes=1000)


    #print(env.render())

    env.close()

if __name__ == "__main__":
    main()