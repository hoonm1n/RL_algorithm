import numpy as np
from collections import defaultdict
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

from model import PolicyNetwork


writer = SummaryWriter(log_dir=f"runs/cartpole_REINFORCE_{int(time.time())}")

class REINFORCE:
    def __init__(self, env, device, gamma=0.99):
        self.device = device
        self.env = env
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.n
        self.gamma = gamma
        self.policy = PolicyNetwork(self.ob_dim, self.ac_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.total_step = 0
        




    def get_action(self, state):
        state_ = torch.FloatTensor(state).to(self.device)
        probs = self.policy(state_)
        dist = Categorical(probs=probs)
        action = dist.sample()
        return action.item()




    def run_episode(self):
        state, _ = self.env.reset()
        done = False
        episode = []
        total_reward = 0

        while not done:
            action = self.get_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state
            total_reward += reward
            self.total_step += 1
        
        return episode, total_reward



    
    def update(self, num_episodes):
        for i in range(num_episodes):
            episode, total_reward = self.run_episode()
            G = 0
            visited = set()
            writer.add_scalar("TotalReward/train", total_reward, self.total_step)
     

            for t in range(len(episode)-1,-1,-1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                state_ = torch.FloatTensor(state).to(self.device)
                log_probs = torch.log(self.policy(state_))
                log_prob = log_probs[action]

                loss = - log_prob * G

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            print(f"Episode {i}, Total Reward: {total_reward:.2f}")

        torch.save(self.policy.state_dict(), './checkpoints/model_state_dict_2.pth')

        writer.close()
    