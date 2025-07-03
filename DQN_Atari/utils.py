import numpy as np
from collections import deque
import random
import cv2
import torch


class ReplayBuffer:
    def __init__(self, capacity, device="cpu", dtype=np.uint8):
        self.capacity = capacity
        self.device = device
        self.state_shape = (4, 84, 84)

        self.states = np.empty((capacity, *self.state_shape), dtype=dtype)
        self.next_states = np.empty((capacity, *self.state_shape), dtype=dtype)
        self.actions = np.empty((capacity,), dtype=np.int64)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=bool)

        self.idx = 0
        self.size = 0

    def insert(self, state, action, reward, next_state, done):
        self.states[self.idx] = np.copy(state)
        self.actions[self.idx] = np.copy(action)
        self.rewards[self.idx] = np.copy(reward)
        self.next_states[self.idx] = np.copy(next_state)
        self.dones[self.idx] = np.copy(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[indices]).to(self.device).float() / 255.0
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device).float() / 255.0
        actions = torch.from_numpy(self.actions[indices]).to(self.device).long().unsqueeze(1)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device).float().unsqueeze(1)
        dones = torch.from_numpy(self.dones[indices]).to(self.device).float().unsqueeze(1)


        return states, actions, rewards, next_states, dones

    def curr_size(self):
        return self.size







def preprocess(_state):
    processed = np.zeros((4, 84, 84), dtype=np.uint8) 

    for i in range(4):
        resized = cv2.resize(_state[i], (84, 110), interpolation=cv2.INTER_AREA)
        cropped = resized[18:102, :]
        processed[i] = cropped 
    return processed


