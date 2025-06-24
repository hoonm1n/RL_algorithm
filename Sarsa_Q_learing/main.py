import gymnasium as gym
import numpy as np
from agent import Sarsa
from agent import Q_learning


def main():
    env = gym.make("Blackjack-v1")
    Sarsa_agent = Sarsa(env, gamma=0.99)
    Q_learning_agent = Q_learning(env, gamma=0.99)


    Sarsa_Q = Sarsa_agent.update(num_episodes=500000)
    Q_learning_Q = Q_learning_agent.update(num_episodes=500000)


    # Print optimal policy : Sarsa
    print("Final Policy (player sum 12~21):  Sarsa")
    for player_sum in range(12, 22):
        state = (player_sum, 1, True)  # usable ace, dealer shows 1
        if state in Sarsa_Q:
            print(f"{state} → {'Stick' if Sarsa_Q[state][0] > Sarsa_Q[state][1] else 'Hit'}")


    # Print optimal policy : Q-learing
    print("Final Policy (player sum 12~21):  Q_learning")
    for player_sum in range(12, 22):
        state = (player_sum, 1, True)  # usable ace, dealer shows 1
        if state in Q_learning_Q:
            print(f"{state} → {'Stick' if Q_learning_Q[state][0] > Q_learning_Q[state][1] else 'Hit'}")


    print(env.render())

if __name__ == "__main__":
    main()
