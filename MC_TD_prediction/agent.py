import numpy as np
from collections import defaultdict


class MCPrediction:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.ac_dim = env.action_space.n
        self.ob_dim = env.observation_space.n
        self.gamma = gamma
        self.V = defaultdict(float)
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(int)
        
    def random_policy(self, state):
        return np.random.choice(self.ac_dim)
    
    def run_episode(self):
        state, _ = self.env.reset()
        done = False
        episode = []

        while not done:
            action = self.random_policy(state)
            next_state, reward, done, truncated, info = self.env.step(action)
            self.env.render()
            print(f"State: {state}, Action: {action}, Reward: {reward}, Next: {next_state}")
            episode.append((state, action, reward))
            state = next_state
        
        return episode


    
    def evaluate(self, num_episodes):
        for i in range(num_episodes):
            episode = self.run_episode()
            G = 0
            vistied = set()
            for t in range(len(episode)-1,-1,-1):
                G = self.gamma * G + episode[t][2]
                if episode[t][0] is not vistied:
                    vistied.add(episode[t][0])
                    self.returns_sum[episode[t][0]] += G
                    self.returns_count[episode[t][0]] += 1
                    self.V[episode[t][0]] = self.returns_sum[episode[t][0]]  / self.returns_count[episode[t][0]] 

        return self.V

    


class TDPrediction:
    def __init__(self, env, gamma=0.99, learning_rate = 0.1):
        self.env = env
        self.ac_dim = env.action_space.n
        self.ob_dim = env.observation_space.n
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.V = defaultdict(float)

        
    def random_policy(self, state):
        return np.random.choice(self.ac_dim)


    
    def evaluate(self, num_episodes):
        for i in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.random_policy(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                self.env.render()
                print(f"State: {state}, Action: {action}, Reward: {reward}, Next: {next_state}")
                self.V[state] = self.V[state] + self.learning_rate*(reward + self.gamma*self.V[next_state] - self.V[state])
                state = next_state

        return self.V