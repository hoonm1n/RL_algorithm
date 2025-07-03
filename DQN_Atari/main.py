import gymnasium as gym
import numpy as np
import torch
from agent import DQN

from gymnasium.wrappers import FrameStack

def main():
    env = gym.make("ALE/Breakout-v5", obs_type="grayscale")
    env = FrameStack(env, num_stack=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQN(env, device, gamma=0.99)

    Q = agent.update(total_update_steps=2500000)


    #print(env.render())

    env.close()

if __name__ == "__main__":
    main()