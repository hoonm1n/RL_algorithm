import os
import numpy as np
from collections import defaultdict
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

from model import ActorCritic




writer = SummaryWriter(log_dir=f"runs/cartpole_A3C_{int(time.time())}")

class A3C:
    def __init__(self, env, device, global_network, optimizer, global_step, lock, worker_idx, gamma=0.99):
        self.device = device
        self.env = env
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.n
        self.gamma = gamma
        self.global_network = global_network
        self.local_network = copy.deepcopy(global_network)
        self.global_optimizer = optimizer
        self.local_optimizer = optim.RMSprop(self.local_network.parameters(), lr=7e-4, alpha=0.99, eps=1e-5, momentum=0, centered=False)
        self.global_step = global_step
        self.lock = lock
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.n_TD = 20
        self.t_max = 20
        self.max_grad_norm = 40.0
        self.worker_idx = worker_idx
        

    def set_seed(self):
        base_seed = 1234

        seed = base_seed + self.worker_idx

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)




    def get_action(self, state):
        state_ = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            probs, _ = self.local_network(state_)
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
        self.set_seed()

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
                with self.lock:
                    self.global_step.value += len(n_step_trajectory)

                if self.global_step.value % 100000 == 0:
                    torch.save(self.global_network.state_dict(), './checkpoints/model_state_dict_2.pth')


            writer.add_scalar("TotalReward/train", total_reward, self.global_step.value)
            writer.add_scalar(f"TotalReward/train_{self.worker_idx}", total_reward, self.global_step.value)
            print(f"Episode {episodes}, Global Step: {self.global_step.value}, My process ID: {os.getpid()}, Total Reward: {total_reward:.2f}")

            if self.global_step.value >= total_update_steps:
                break
        
        # torch.save(self.global_network.state_dict(), './checkpoints/model_state_dict_1.pth')

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
            _, next_value = self.local_network(n_state)
            target += self.gamma**len(n_step_trajectory) * next_value * (1-float(dones[-1]))
            
            _, current_value = self.local_network(state)
            advantage = target - current_value


        probs, current_V = self.local_network(state)
        dist = Categorical(probs)
        log_prob = dist.log_prob(torch.tensor(action).to(self.device))
        
        entropy = dist.entropy().mean()

        policy_loss = - log_prob * advantage
        value_loss = nn.MSELoss()(current_V, target)

        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        self.global_optimizer.zero_grad()
        self.local_optimizer.zero_grad()


        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), self.max_grad_norm)

        for local_param, global_param in zip(self.local_network.parameters(), self.global_network.parameters()):
            global_param._grad = local_param.grad.clone()

        self.global_optimizer.step()

        self.local_network.load_state_dict(self.global_network.state_dict())






