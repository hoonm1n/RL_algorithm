import numpy as np
from collections import defaultdict
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils as nn_utils


from model import QNetwork
from utils import ReplayBuffer
from utils import preprocess

writer = SummaryWriter(log_dir=f"runs/atari_dqn_{int(time.time())}")

class DQN:
    def __init__(self, env, device, gamma=0.99):
        self.device = device
        self.env = env
        self.ob_dim = env.observation_space.shape
        self.ac_dim = env.action_space.n
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_start = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 1000000
        self.Q = QNetwork(self.ob_dim, self.ac_dim).to(device)
        self.targetQ = QNetwork(self.ob_dim, self.ac_dim).to(device)
        self.optimizer = optim.RMSprop(self.Q.parameters(), lr=0.00025, alpha=0.95, eps=1e-8, momentum=0, centered=False)
        self.rb = ReplayBuffer(capacity=1000000, device=self.device)
        self.batch_size = 32
        self.start_rb_size = 50000
        self.target_update_freq = 10000
        self.total_step = 0
        self.max_grad_norm = 1.0
        

    def random_policy(self, state):
        return np.random.randint(self.ac_dim)
    
    def epsilon_greedy_policy(self, state, ep):
        if np.random.rand() < ep:
            action = self.random_policy(state)
        else:
            state_ = torch.from_numpy(state).to(self.device).float().unsqueeze(0) / 255.0
            with torch.no_grad():
                q_values = self.Q(state_)
                action = q_values.argmax(dim=1).item()
        return action
    

    
    def update(self, total_update_steps):
        episodes = 0
        while 1:
            state, _ = self.env.reset()
            obs = preprocess(state)

            done = False
            total_reward = 0
            episodes += 1
     

            while not done:
                action = self.epsilon_greedy_policy(obs, self.epsilon)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                reward = np.clip(reward, -1, 1)
                done = terminated or truncated
                next_obs = preprocess(next_state)

                self.rb.insert(obs, action, reward, next_obs, done)
                
                if self.rb.curr_size() > self.start_rb_size:
                    self.train_step()
                
                obs = next_obs
                total_reward += reward
                self.total_step += 1
                
                self.epsilon = max(self.epsilon_min, self.epsilon_start - (self.total_step / self.epsilon_decay) * (self.epsilon_start - self.epsilon_min))


                if self.total_step % self.target_update_freq == 0:
                    self.update_targetNetwork()
                    print("target Q update...")
                if self.total_step % 100000 == 0:
                    torch.save(self.Q.state_dict(), './checkpoints/model_state_dict_6.pth')

            print(f"Episode {episodes}, Total step {self.total_step}, Total Reward: {total_reward:.2f}")

            writer.add_scalar("Total_Reward", total_reward, episodes)

            if self.total_step >= total_update_steps:
                break

        torch.save(self.Q.state_dict(), './checkpoints/model_state_dict_6.pth')

        writer.close()
        return self.Q
    

    def train_step(self):
        if self.rb.curr_size() < self.batch_size:
            return 
        
        # batch = self.rb.sample(self.batch_size)

        # states, actions, rewards, next_states, dones = zip(*batch)

        states, actions, rewards, next_states, dones = self.rb.sample(self.batch_size)


        # states = np.array(states)
        # actions = np.array(actions)
        # rewards = np.array(rewards)
        # next_states = np.array(next_states)
        # dones = np.array(dones)

        # states = torch.FloatTensor(states).to(self.device)
        # actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        # rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        # next_states = torch.FloatTensor(next_states).to(self.device)
        # dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)


        q_values = self.Q(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.targetQ(next_states).max(1, keepdim=True)[0]
            q_target = rewards + self.gamma * (1 - dones) * q_next

        loss = nn.MSELoss()(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.Q.parameters(), self.max_grad_norm)

        self.optimizer.step()


    def update_targetNetwork(self):
        self.targetQ.load_state_dict(self.Q.state_dict())