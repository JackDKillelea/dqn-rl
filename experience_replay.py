from collections import deque
import random

class ReplayMemory:
    def __init__(self, max_len, seed=None):
        self.memory = deque([], maxlen=max_len)

        # Add option for seed for reproducibility
        if seed is not None:
            random.seed(seed)

    def append(self, transaction):
        self.memory.append(transaction)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)