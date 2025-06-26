import numpy as np
from collections import defaultdict


class Sarsa:
    def __init__(self, env, gamma=0.99, learning_rate=0.1):
        self.env = env
        self.ac_dim = env.action_space.n
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.Q = defaultdict(lambda: np.zeros(self.ac_dim))
        self.epsilon = 0.1
        
    def random_policy(self, state):
        return np.random.randint(self.ac_dim)
    
    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            action = self.random_policy(state)
        else:
            action = np.argmax(self.Q[state])

        return action
    

    
    def update(self, num_episodes):
        for i in range(num_episodes):
            state, _ = self.env.reset()
            action = self.epsilon_greedy_policy(state)
            done = False

            while not done:
                next_state, reward, done, truncated, info = self.env.step(action)
                next_action = self.epsilon_greedy_policy(next_state)
                self.Q[state][action] = self.Q[state][action] + self.learning_rate*(reward + self.gamma*self.Q[next_state][next_action] - self.Q[state][action])
                state = next_state
                action = next_action

        return self.Q




class Q_learning:
    def __init__(self, env, gamma=0.99, learning_rate=0.1):
        self.env = env
        self.ac_dim = env.action_space.n
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.Q = defaultdict(lambda: np.zeros(self.ac_dim))
        self.epsilon = 0.1
        
    def random_policy(self, state):
        return np.random.randint(self.ac_dim)
    
    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            action = self.random_policy(state)
        else:
            action = np.argmax(self.Q[state])

        return action
    

    
    def update(self, num_episodes):
        for i in range(num_episodes):
            state, _ = self.env.reset()
            done = False

            while not done:
                action = self.epsilon_greedy_policy(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                self.Q[state][action] = self.Q[state][action] + self.learning_rate*(reward + self.gamma*np.max(self.Q[next_state]) - self.Q[state][action])
                state = next_state
  

        return self.Q