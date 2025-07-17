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

from model import ActorCritic



writer = SummaryWriter(log_dir=f"runs/hopper_PPO_{int(time.time())}")

class PPO:
    def __init__(self, envs, device, num_envs, gamma=0.99):
        self.device = device
        self.envs = envs
        self.ob_dim = envs.single_observation_space.shape[0]
        self.ac_dim = envs.single_action_space.shape[0]
        self.gamma = gamma
        self.actor_critic = ActorCritic(self.ob_dim, self.ac_dim).to(device)
        self.actor_critic_old = ActorCritic(self.ob_dim, self.ac_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=3e-4)
        self.total_step = 0
        self.rollout_len = 128
        self.total_episode = 0
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 40.0
        self.clip_eps = 0.2
        self.num_envs = num_envs
        self.minibatch_size = 64


        



    def get_action(self, states):
        states_ = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            mu, std, value = self.actor_critic_old(states_)
        cov = torch.diag_embed(std ** 2)
        dist = MultivariateNormal(mu, cov)
        action = dist.sample()
        log_prob = dist.log_prob(action)
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
        

        for _ in range(self.rollout_len):
            actions, log_probs = self.get_action(states)
            actions_for_env = actions.detach().cpu().numpy()
            log_probs_numpy = log_probs.detach().cpu().numpy()
            next_states, rewards, terminateds, truncateds, infos = self.envs.step(actions_for_env)
            dones = np.logical_or(terminateds, truncateds)

            states_all.append(states)
            actions_all.append(actions_for_env)
            next_states_all.append(next_states)
            rewards_all.append(rewards)
            dones_all.append(dones)
            log_probs_all.append(log_probs_numpy)


            for i, reward in enumerate(rewards):
                episode_rewards[i] += reward


            for i, done in enumerate(dones):
                if done:
                    reset_state, _ = self.envs.envs[i].reset()
                    next_states[i] = reset_state

                    print(f"Total Step {self.total_step}, Total Reward: {episode_rewards[i]:.2f}")
                    episode_rewards[i] = 0


            states = next_states

        return states_all, actions_all, next_states_all, rewards_all, dones_all, log_probs_all






    def compute_gae(self, states, rewards, dones, gamma=0.99, lam=0.95):
        N, T, _ = rewards.shape
        advantages = torch.zeros(N, T, device=self.device)
        with torch.no_grad():
            _, _, values = self.actor_critic_old(states)
            for n in range(N):
                gae = 0.0
                for t in reversed(range(T)):
                    if t == T - 1:
                        next_value = 0.0
                    else:
                        next_value = values[n][t + 1] * (1 - dones[n][t])
                    delta = rewards[n][t] + gamma * next_value - values[n][t]
                    gae = delta + gamma * lam * gae * (1 - dones[n][t])
                    advantages[n][t] = gae
        return advantages

     

    



    def update(self, total_update_steps):
        episodes = 0
        while 1:
            states, actions, next_states, rewards, dones, log_probs = self.rollout()
            
            print("rollout done")
            self.train_step(states, actions, next_states, rewards, dones, log_probs)

            self.total_step += (len(states) * self.num_envs)

            print("update done")

            print(self.total_step)

            if self.total_step % 100000 == 0:
                torch.save(self.actor_critic.state_dict(), './checkpoints/model_state_dict_2.pth')

            # writer.add_scalar("TotalReward/train", total_reward, self.total_step)
            # print(f"Episode {episodes}, Total Reward: {total_reward:.2f}")

            if self.total_step >= total_update_steps:
                break
            

        torch.save(self.actor_critic.state_dict(), './checkpoints/model_state_dict_2.pth')

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


        # print(states.shape)
        # print(rewards.shape)
        


        advantages = self.compute_gae(states, rewards, dones)
        advantages = advantages.to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) 

        


        states = states.reshape(-1, states.shape[2])
        actions = actions.reshape(-1, actions.shape[2])
        rewards = rewards.reshape(-1, rewards.shape[2])
        next_states = next_states.reshape(-1, next_states.shape[2])
        dones = dones.reshape(-1, dones.shape[2])
        log_probs = log_probs.reshape(-1, log_probs.shape[2])
        advantages = advantages.reshape(-1)


        total_sample_size, _ = states.shape


        for epoch in range(10):
            indices = np.arange(total_sample_size)
            np.random.shuffle(indices)

            for start in range(0, total_sample_size, self.minibatch_size):
                end = start + self.minibatch_size
                batch_idxs = indices[start:end]

                batch_states = states[batch_idxs]
                batch_actions = actions[batch_idxs]
                batch_rewards = rewards[batch_idxs]
                batch_log_probs = log_probs[batch_idxs]
                batch_advantages = advantages[batch_idxs]


                with torch.no_grad():
                    _, _, values = self.actor_critic(batch_states)
                    returns = []
                    ret = 0
                    for r in reversed(batch_rewards.squeeze().cpu().numpy()):
                        ret = r + self.gamma * ret
                        returns.insert(0, ret)
                    returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)  

                mu_new, std_new, values = self.actor_critic(batch_states)
                value_loss = nn.MSELoss()(values, batch_advantages.unsqueeze(1))

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

                self.optimizer.zero_grad()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

                self.optimizer.step()

        self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())