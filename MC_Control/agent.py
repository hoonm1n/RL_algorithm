import numpy as np
from collections import defaultdict


class MCcontrol:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.ac_dim = env.action_space.n
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(self.ac_dim))
        self.returns_sum = defaultdict(lambda: np.zeros(self.ac_dim))
        self.returns_count = defaultdict(lambda: np.zeros(self.ac_dim))
        self.epsilon = 0.09
        
    def random_policy(self, state):
        return np.random.randint(self.ac_dim)
    
    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            action = self.random_policy(state)
        else:
            action = np.argmax(self.Q[state])

        return action
    
    def run_episode(self):
        state, _ = self.env.reset()
        done = False
        episode = []

        while not done:
            action = self.epsilon_greedy_policy(state)
            next_state, reward, done, truncated, info = self.env.step(action)
            self.env.render()
            episode.append((state, action, reward))
            state = next_state
        
        return episode


    
    def update(self, num_episodes):
        for i in range(num_episodes):
            episode = self.run_episode()
            G = 0
            visited = set()

            for t in range(len(episode)-1,-1,-1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                if (state, action) not in visited:
                    self.returns_sum[state][action] += G
                    self.returns_count[state][action] += 1
                    self.Q[state][action] += (self.returns_sum[state][action]-self.Q[state][action]) / self.returns_count[state][action] 
                    visited.add((state,action))
            # if i % 10000 == 0:
            #     for player_sum in range(12, 22):
            #         s = (player_sum, 1, False)
            #         if s in self.Q:
            #             print(f"{state} → {'Stick' if self.Q[s][0] > self.Q[s][1] else 'Hit'}")


        return self.Q

    