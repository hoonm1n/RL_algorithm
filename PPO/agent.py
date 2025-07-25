import gym
import numpy as np
from collections import defaultdict
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import MultivariateNormal
import torch.multiprocessing as mp

# from model import ActorCritic
from model import PolicyNetwork
from model import ValueNetwork

from torch.utils.data import DataLoader, TensorDataset


writer = SummaryWriter(log_dir=f"runs/halfcheetah_PPO_{int(time.time())}")

class PPO:
    def __init__(self, envs, device, num_envs, gamma=0.99):
        self.device = device
        self.envs = envs
        self.ob_dim = envs.single_observation_space.shape[0]
        self.ac_dim = envs.single_action_space.shape[0]
        self.gamma = gamma
        # self.actor_critic = ActorCritic(self.ob_dim, self.ac_dim).to(device)
        # self.actor_critic_old = ActorCritic(self.ob_dim, self.ac_dim).to(device)
        self.policy = PolicyNetwork(self.ob_dim, self.ac_dim).to(device)
        self.critic = ValueNetwork(self.ob_dim).to(device)
        # self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=3e-4)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.total_step = 0
        self.rollout_len = 512
        self.total_episode = 0
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 1.0
        self.clip_eps = 0.2
        self.num_envs = num_envs
        self.minibatch_size = 64


        



    def get_action(self, states):
        states_ = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            mu, std = self.policy(states_)
        cov = torch.diag_embed(std ** 2)
        dist = MultivariateNormal(mu, cov)
        z = dist.sample()
        action = torch.tanh(z)

        log_prob = dist.log_prob(z)
        log_prob -= torch.sum(torch.log(1 - torch.tanh(z) ** 2 + 1e-6), dim=1)
        # log_prob = dist.log_prob(action)
        return action, log_prob





    def rollout(self):
        states, _ = self.envs.reset()
 
        states_all = [] 
        actions_all = [] 
        next_states_all = []
        rewards_all = [] 
        dones_all = [] 
        log_probs_all = []

        episode_rewards = [0] * self.num_envs
        completed_episode_rewards = []
        

        for _ in range(self.rollout_len):
            actions, log_probs = self.get_action(states)
            actions_for_env = actions.clamp(-1,1).cpu().numpy()
            log_probs_numpy = log_probs.detach().cpu().numpy()
            next_states, rewards, terminateds, truncateds, infos = self.envs.step(actions_for_env)
            dones = np.logical_or(terminateds, truncateds)

            states_all.append(states)
            actions_all.append(actions.cpu().numpy())
            next_states_all.append(next_states)
            rewards_all.append(rewards)
            dones_all.append(dones)
            log_probs_all.append(log_probs_numpy)


            for i, reward in enumerate(rewards):
                episode_rewards[i] += reward


            for i, done in enumerate(dones):
                if done:
                    completed_episode_rewards.append(episode_rewards[i])

                    reset_state, _ = self.envs.envs[i].reset()
                    next_states[i] = reset_state

                    episode_rewards[i] = 0


            states = next_states
        if completed_episode_rewards:
            mean_reward = np.mean(completed_episode_rewards)
        else:
            print("not done")
            mean_reward = np.mean(episode_rewards)
        print(f"Total Step {self.total_step}, Mean Reward: {mean_reward:.2f}")

        return states_all, actions_all, next_states_all, rewards_all, dones_all, log_probs_all, mean_reward







    def compute_gae(self, states, next_states, rewards, dones, gamma=0.99, lam=0.95):
        N, T, _ = rewards.shape
        advantages = torch.zeros(N, T, device=self.device)
        with torch.no_grad():
            values = self.critic(states)
            last_values = self.critic(next_states)
            for n in range(N):
                gae = 0.0
                for t in reversed(range(T)):
                    if dones[n][t]:
                        gae = 0.0
                    if t == T - 1:
                        next_value = last_values[n][t] * (1 - dones[n][t])
                    else:
                        next_value = values[n][t + 1] * (1 - dones[n][t])
                    delta = rewards[n][t] + gamma * next_value - values[n][t]
                    gae = delta + gamma * lam * gae
                    advantages[n][t] = gae
            returns = advantages.unsqueeze(2) + values
            
        return advantages, returns

 

    



    def update(self, total_update_steps):
        episodes = 0
        while 1:
            states, actions, next_states, rewards, dones, log_probs, mean_reward = self.rollout()
            
            print("rollout done")
            self.train_step(states, actions, next_states, rewards, dones, log_probs)

            self.total_step += (len(states) * self.num_envs)

            print("update done")

            print(self.total_step)

            if self.total_step % 100000 == 0:
                torch.save(self.policy.state_dict(), './checkpoints/model_state_dict_3.pth')

            writer.add_scalar("MeanReward/train", mean_reward, self.total_step)
            # print(f"Episode {episodes}, Total Reward: {total_reward:.2f}")

            if self.total_step >= total_update_steps:
                break
            

        torch.save(self.policy.state_dict(), './checkpoints/model_state_dict_3.pth')

        writer.close()



        


    def train_step(self, states, actions, next_states, rewards, dones, log_probs):

        states = np.array(states)
        actions = np.array(actions) 
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones) 
        log_probs = np.array(log_probs) 



        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(2).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(2).to(self.device)
        log_probs = torch.FloatTensor(log_probs).unsqueeze(2).to(self.device)


        states = states.permute(1, 0, 2)
        actions = actions.permute(1, 0, 2)
        rewards = rewards.permute(1, 0, 2)
        next_states = next_states.permute(1, 0, 2)
        dones = dones.permute(1, 0, 2)
        log_probs = log_probs.permute(1, 0, 2)


        advantages, returns = self.compute_gae(states, next_states, rewards, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) 

        

        states = states.reshape(-1, states.shape[2])
        actions = actions.reshape(-1, actions.shape[2])
        rewards = rewards.reshape(-1, rewards.shape[2])
        next_states = next_states.reshape(-1, next_states.shape[2])
        dones = dones.reshape(-1, dones.shape[2])
        log_probs = log_probs.reshape(-1, log_probs.shape[2])
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1, returns.shape[2])



        total_sample_size, _ = states.shape


        dataset = TensorDataset(states, actions, log_probs, advantages, returns)
        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)


        for epoch in range(10):
            for batch_states, batch_actions, batch_log_probs, batch_advantages, batch_returns in loader:

                mu_new, std_new = self.policy(batch_states)
                values = self.critic(batch_states)
                value_loss = nn.MSELoss()(values, batch_returns.detach())

                cov_new = torch.diag_embed(std_new**2)
                dist_new = MultivariateNormal(mu_new, cov_new)
                log_probs_new = dist_new.log_prob(batch_actions)
                
                ratio = torch.exp(log_probs_new - batch_log_probs.squeeze(1))
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

                surrogate_loss = ratio * batch_advantages
                clipped_surrogate_loss = clipped_ratio * batch_advantages

                surrogate_loss = - torch.min(surrogate_loss, clipped_surrogate_loss).mean()
                
                entropy = dist_new.entropy().mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # self.optimizer.zero_grad()
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)


                # self.optimizer.step()
                self.policy_optimizer.step()
                self.value_optimizer.step()

            print(f"Total Step {self.total_step}, value loss: {value_loss:.2f}")


        # self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())