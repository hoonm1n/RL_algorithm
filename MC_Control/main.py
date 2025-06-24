import gymnasium as gym
import numpy as np
from collections import defaultdict
from agent import MCcontrol


def main():
    env = gym.make("Blackjack-v1")
    MCagent = MCcontrol(env, gamma=0.99)

    MC_Q = MCagent.update(num_episodes=500000)

    
    # for s in range(env.observation_space.n):
    #     print(f"state: {s}, Q: {MC_Q[s]}")
    #     print(f"state: {s}, maxQ: {np.argmax(MC_Q[s])}")

    # Print optimal policy
    print("Final Policy (player sum 12~21):")
    for player_sum in range(12, 22):
        state = (player_sum, 1, True)  # usable ace, dealer shows 1
        if state in MC_Q:
            print(f"{state} → {'Stick' if MC_Q[state][0] > MC_Q[state][1] else 'Hit'}")



    print(env.render())

if __name__ == "__main__":
    main()
