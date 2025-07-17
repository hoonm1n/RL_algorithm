import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu_output = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.critic(x)
        mu = self.mu_output(x)
        std = torch.exp(self.log_std)
        return mu, std, value