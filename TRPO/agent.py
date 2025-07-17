import numpy as np
from collections import defaultdict
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import MultivariateNormal

from model import PolicyNetwork
from model import ValueNetwork



writer = SummaryWriter(log_dir=f"runs/hopper_TRPO_{int(time.time())}")

class TRPO:
    def __init__(self, env, device, gamma=0.99):
        self.device = device
        self.env = env
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        self.gamma = gamma
        self.policy = PolicyNetwork(self.ob_dim, self.ac_dim).to(device)
        self.policy_old = PolicyNetwork(self.ob_dim, self.ac_dim).to(device)
        self.value = ValueNetwork(self.ob_dim).to(device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=1e-3)
        self.total_step = 0
        self.max_kl = 0.05

        




    def get_action(self, state):
        state_ = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mu, std = self.policy_old(state_)
        cov = torch.diag_embed(std ** 2)
        dist = MultivariateNormal(mu, cov)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob



    def single_path(self,state):
        trajectory = []
        sum_reward = 0
        done = False
        while not done:
            action, log_prob = self.get_action(state)
            action_for_env = action.detach().cpu().numpy()
            next_state, reward, terminated, truncated, info = self.env.step(action_for_env)
            done = terminated or truncated

            trajectory.append((state, action, next_state, reward, done, log_prob))

            state = next_state
            sum_reward += reward

        return trajectory, sum_reward



    def compute_gae(self, states, rewards, gamma=0.99, lam=0.95):
        T = len(rewards)
        with torch.no_grad():
            values = self.value(states)
            advantages = torch.zeros(T)
            gae = 0.0
            for t in reversed(range(T)):
                if t == T-1:
                    delta = rewards[t] - values[t]
                else:
                    delta = rewards[t] + gamma * values[t + 1] - values[t]
                gae = delta + gamma * lam * gae
                advantages[t] = gae
        return advantages

    
    def update(self, total_update_steps):
        episodes = 0
        while 1:
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            episodes += 1

            trajectory, sum_reward = self.single_path(state)

            self.train_step(trajectory)

            
            total_reward = sum_reward
            self.total_step += len(trajectory)

            if self.total_step % 100000 == 0:
                torch.save(self.policy.state_dict(), './checkpoints/model_state_dict_2.pth')


            writer.add_scalar("TotalReward/train", total_reward, self.total_step)
            print(f"Episode {episodes}, Total Reward: {total_reward:.2f}")

            if self.total_step >= total_update_steps:
                break
        
        torch.save(self.policy.state_dict(), './checkpoints/model_state_dict_2.pth')

        writer.close()


        
    


    def train_step(self, trajectory):

        states, actions, next_states, rewards, dones, log_probs = zip(*trajectory)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        log_probs = torch.stack(log_probs).detach().unsqueeze(1).to(self.device)


        advantages = self.compute_gae(states, rewards)
        advantages = advantages.to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) 
        # print("Adv mean:", advantages.mean().item(), "std:", advantages.std().item())


        with torch.no_grad():
            values = self.value(states)
            returns = []
            ret = 0
            for r in reversed(rewards.squeeze().cpu().numpy()):
                ret = r + self.gamma * ret
                returns.insert(0, ret)
            returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)  

        value_preds = self.value(states)
        value_loss = nn.MSELoss()(value_preds, returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()



        surrogate_loss, kl = self.get_loss_kl(states, actions, log_probs, advantages)


        gradient = torch.autograd.grad(surrogate_loss, self.policy.parameters(), retain_graph=True)
        gradient_flat = torch.cat([g.contiguous().view(-1) for g in gradient])


        

        #2.Fisher Information Matrix


        #3.Conjugate Gradient
        direction = self.conjugate_gradient(gradient_flat, kl)




        #4.Line Search
        step_direction = direction
        shs = 0.5 * (direction @ self.fim(kl, direction))
        step_size = torch.sqrt(self.max_kl / (shs + 1e-8))
        full_step = step_size * step_direction



        old_parameters = torch.cat([p.view(-1) for p in self.policy.parameters()])

        new_parameters = self.line_search(self.policy, states, actions, log_probs, advantages, old_parameters, full_step, gradient_flat, self.max_kl)


        self.insert_parameters_to_model(self.policy, new_parameters)


        self.policy_old.load_state_dict(self.policy.state_dict())






    def fim(self, kl, v):
        kl_grad = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True, retain_graph=True)
        kl_grad_flat = torch.cat([g.contiguous().view(-1) for g in kl_grad])

        grad_vector_product = torch.sum(kl_grad_flat * v)

        hvp = torch.autograd.grad(grad_vector_product, self.policy.parameters(), retain_graph=True)
        hvp_flat = torch.cat([h.contiguous().view(-1) for h in hvp])

        damping = 1e-2

        return hvp_flat + damping * v


    def conjugate_gradient(self, g, kl):
        x = torch.zeros_like(g)
        r = g.clone()
        p = r.clone()
        for _ in range(10):
            Ap = self.fim(kl, p)
            al = (r @ r)/(p @ Ap)
            x += al * p
            r_next = r - al * Ap
            be = (r_next @ r_next) / (r @ r)
            p = r_next + be * p
            r = r_next
            # print((p @ Ap).item())
            # print((r_next @ r_next).sqrt())

            if (r_next @ r_next).sqrt() < 1e-6:
                break
        
        print((r_next @ r_next).sqrt().item())

        return x



    def line_search(self, policy, states, actions, log_probs, advantages, old_params, full_step, gradient_flat, max_kl=0.01, max_backtracks=10, accept_ratio=0.1):
        fval, kl_val = self.get_loss_kl(states, actions, log_probs, advantages)
        step_frac = 1.0

        for _ in range(max_backtracks):
            new_params = old_params + step_frac * full_step
            self.insert_parameters_to_model(policy, new_params)

            new_fval, new_kl = self.get_loss_kl(states, actions, log_probs, advantages)

            actual_improve = fval - new_fval
            expected_improve = (gradient_flat @ full_step) * step_frac  # 근사적 예상 향상
            improve_ratio = actual_improve / (expected_improve + 1e-8)

            # 조건: 손실 개선 + KL 제한 + 기대 개선 비율 만족
            if actual_improve > 0 and new_kl <= max_kl and improve_ratio >= accept_ratio:
                print(f"[Line Search] Step accepted at frac={step_frac:.3f}, KL={new_kl:.5f}")
                return new_params

            step_frac *= 0.5  # 실패 → step size 절반으로 줄임

        print("[Line Search] No acceptable step found; using old parameters.")
        self.insert_parameters_to_model(policy, old_params)
        return old_params





    def get_loss_kl(self, states, actions, log_probs, advantages):
        mu_new, std_new = self.policy(states)
        cov_new = torch.diag_embed(std_new**2)
        dist_new = MultivariateNormal(mu_new, cov_new)
        log_probs_new = dist_new.log_prob(actions)

        mu_old, std_old = self.policy_old(states)
        cov_old = torch.diag_embed(std_old**2)
        dist_old = MultivariateNormal(mu_old, cov_old)


        print("---------------------------")
        print(log_probs.shape)
        print(log_probs_new.shape)
        

        
        ratio = torch.exp(log_probs_new - log_probs.squeeze(1))
        surrogate_loss = (ratio * advantages).mean()

        print(ratio.shape)
        print(surrogate_loss.shape)


        kl = torch.distributions.kl_divergence(dist_old, dist_new).mean()

        return -surrogate_loss, kl





    def insert_parameters_to_model(self, model, params):
        prev_ind = 0
        for param in model.parameters():
            flat_size = param.numel()
            param.data.copy_(params[prev_ind:prev_ind + flat_size].view_as(param))
            prev_ind += flat_size