import numpy as np
from collections import deque
import random



class replay_buffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.size = 0
        self.rb = deque()

    
    def insert(self, transition):
        if self.size == self.capacity:
            self.rb.popleft()
            self.rb.append(transition)
        
        else:
            self.size += 1
            self.rb.append(transition)

    
    def sample(self, batch_size):
        if self.size >= batch_size:
            batch = random.sample(self.rb, batch_size)
        return batch

    def curr_size(self):
        return self.size