import numpy as np
from collections import defaultdict
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import QNetwork
from utils import replay_buffer

writer = SummaryWriter(log_dir=f"runs/cartpole_dqn_{int(time.time())}")

class DQN:
    def __init__(self, env, device, gamma=0.99):
        self.device = device
        self.env = env
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.n
        self.gamma = gamma
        self.epsilon = 1.0
        self.Q = QNetwork(self.ob_dim, self.ac_dim).to(device)
        self.targetQ = QNetwork(self.ob_dim, self.ac_dim).to(device)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=1e-4)
        self.rb = replay_buffer(capacity=10000)
        self.batch_size = 64
        self.target_update_freq = 10
        self.total_step = 0
        

    def random_policy(self, state):
        return np.random.randint(self.ac_dim)
    
    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            action = self.random_policy(state)
        else:
            state_ = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.Q(state_)
                action = q_values.argmax(dim=1).item()
        return action
    

    
    def update(self, num_episodes):
        for i in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
     

            while not done:
                action = self.epsilon_greedy_policy(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                self.rb.insert((state, action, reward, next_state, done))
                self.train_step()

                self.env.render()
                
                state = next_state
                total_reward += reward
                self.total_step += 1



            if self.total_step > 1000:
                self.epsilon = 0.1        

            if i % self.target_update_freq == 0:
                self.update_targetNetwork()

            print(f"Episode {i}, Total Reward: {total_reward:.2f}")

            writer.add_scalar("Total_Reward", total_reward, i)

        writer.close()
        return self.Q
    

    def train_step(self):
        if self.rb.curr_size() < self.batch_size:
            return 
        
        batch = self.rb.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)


        q_values = self.Q(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.targetQ(next_states).max(1, keepdim=True)[0]
            q_target = rewards + self.gamma * (1 - dones) * q_next

        loss = nn.MSELoss()(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_targetNetwork(self):
        self.targetQ.load_state_dict(self.Q.state_dict())