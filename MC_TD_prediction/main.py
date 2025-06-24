import gymnasium as gym
import numpy as np
from collections import defaultdict
from agent import MCPrediction
from agent import TDPrediction


def main():
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="ansi")
    MCagent = MCPrediction(env, gamma=0.99)
    TDagent = TDPrediction(env, gamma=0.99, learning_rate=0.1)


    MC_V = MCagent.evaluate(num_episodes=10000)
    TD_V = TDagent.evaluate(num_episodes=10000)

    
    for s in range(env.observation_space.n):
        print(f"MC -> state: {s}, v: {MC_V[s]}")
    print(f"------------------------------------")
    for s in range(env.observation_space.n):
        print(f"TD -> state: {s}, v: {TD_V[s]}")

    print(env.render())

if __name__ == "__main__":
    main()
