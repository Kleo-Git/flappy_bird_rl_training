import random
from collections import deque, namedtuple
from typing import List, Tuple, Any

# Defining a namedtuple for structured data storage
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    """
    A class used to store and sample transitions for Reinforcement Learning.
    This helps in breaking the correlation between consecutive experiences.
    """
    def __init__(self, capacity: int, seed: int = None):
        """
        Initializes the replay memory.
        
        Args:
            capacity (int): Maximum number of transitions to store.
            seed (int, optional): Seed for the random number generator for reproducibility.
        """
        self.memory = deque([], maxlen=capacity)
        
        if seed is not None:
            random.seed(seed)

    def append(self, *args: Any):
        """
        Saves a transition. Usage: memory.append(state, action, next_state, reward, done)
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Randomly samples a batch of transitions from memory.
        
        Args:
            batch_size (int): Number of transitions to sample.
            
        Returns:
            List[Transition]: A list of sampled Transition objects.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Returns the current size of the memory."""
        return len(self.memory)