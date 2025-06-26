import gymnasium as gym
import numpy as np
from collections import defaultdict
from agent import MCcontrol


def main():
    env = gym.make("Taxi-v3")
    MCagent = MCcontrol(env, gamma=0.99)

    MC_Q = MCagent.update(num_episodes=100000)

    print(f"MC Policy average reward (100 episodes): {MCagent.evaluate_policy()}")

    print(env.render())

if __name__ == "__main__":
    main()
