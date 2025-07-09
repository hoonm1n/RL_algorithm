import numpy as np
from collections import defaultdict
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

from model import PolicyNetwork
from model import ValueNetwork



writer = SummaryWriter(log_dir=f"runs/cartpole_A2C_{int(time.time())}")

class A2C:
    def __init__(self, env, device, gamma=0.99):
        self.device = device
        self.env = env
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.n
        self.gamma = gamma
        self.policy = PolicyNetwork(self.ob_dim, self.ac_dim).to(device)
        self.value = ValueNetwork(self.ob_dim, self.ac_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=5e-4)
        self.total_step = 0
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.n_TD = 3
        




    def get_action(self, state):
        state_ = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            probs = self.policy(state_)
        dist = Categorical(probs=probs)
        action = dist.sample()
        return action.item()



    def n_step_TD(self,state):
        n_step_trajectory = []
        sum_reward = 0
        for i in range(self.n_TD):
            action = self.get_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            n_step_trajectory.append((state,action,next_state,reward,done))

            state = next_state
            sum_reward += reward
            if done:
                break

        return n_step_trajectory , done, sum_reward

    
    def update(self, total_update_steps):
        episodes = 0
        while 1:
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            episodes += 1

            while not done:
                n_step_trajectory, done, sum_reward = self.n_step_TD(state)

                self.train_step(n_step_trajectory)

                state = n_step_trajectory[-1][2]
                total_reward += sum_reward
                self.total_step += len(n_step_trajectory)

                if self.total_step % 100000 == 0:
                    torch.save(self.policy.state_dict(), './checkpoints/model_state_dict_2.pth')


            writer.add_scalar("TotalReward/train", total_reward, self.total_step)
            print(f"Episode {episodes}, Total Reward: {total_reward:.2f}")

            if self.total_step >= total_update_steps:
                break
        
        torch.save(self.policy.state_dict(), './checkpoints/model_state_dict_2.pth')

        writer.close()


        
    


    def train_step(self, n_step_trajectory):

        states, actions, next_states, rewards, dones = zip(*n_step_trajectory)

        state = torch.FloatTensor(states[0]).to(self.device)
        action = torch.tensor(actions[0]).to(self.device)

        n_state = torch.FloatTensor(next_states[-1]).to(self.device)

        with torch.no_grad():
            target = 0
            for i in range(len(n_step_trajectory)):
                target += self.gamma**i * rewards[i]
            target += self.gamma**len(n_step_trajectory) * self.value(n_state) * (1-float(dones[-1]))

            advantage = target - self.value(state)


        probs = self.policy(state)
        dist = Categorical(probs)
        log_prob = dist.log_prob(torch.tensor(action).to(self.device))
        
        entropy = dist.entropy().mean()

        policy_loss = - log_prob * advantage
        value_loss = nn.MSELoss()(self.value(state), target)

        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        loss.backward()

        self.policy_optimizer.step()
        self.value_optimizer.step()

