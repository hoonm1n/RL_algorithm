import numpy as np
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter("runs/mc_taxi")


class MCcontrol:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.ac_dim = env.action_space.n
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(self.ac_dim))
        self.returns_sum = defaultdict(lambda: np.zeros(self.ac_dim))
        self.returns_count = defaultdict(lambda: np.zeros(self.ac_dim))
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        
    def random_policy(self, state):
        return np.random.randint(self.ac_dim)
    
    def epsilon_greedy_policy(self, state):

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if np.random.rand() < self.epsilon:
            action = self.random_policy(state)
        else:
            action = np.argmax(self.Q[state])

        return action
    
    def run_episode(self):
        state, _ = self.env.reset()
        done = False
        episode = []
        total_reward = 0

        while not done:
            action = self.epsilon_greedy_policy(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state
            total_reward += reward
        
        return episode, total_reward


    
    def update(self, num_episodes):
        for i in range(num_episodes):
            episode, total_reward = self.run_episode()
            G = 0
            visited = set()
            writer.add_scalar("TotalReward/train", total_reward, i)

            for t in range(len(episode)-1,-1,-1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                if (state, action) not in visited:
                    self.returns_sum[state][action] += G
                    self.returns_count[state][action] += 1
                    self.Q[state][action] = self.returns_sum[state][action] / self.returns_count[state][action]
                    visited.add((state,action))
        
                    
        writer.close()
        return self.Q
    
    def evaluate_policy(self, episodes=100):
        total_rewards = []
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = np.argmax(self.Q[state])
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
            total_rewards.append(total_reward)
            
        return np.mean(total_rewards)


    