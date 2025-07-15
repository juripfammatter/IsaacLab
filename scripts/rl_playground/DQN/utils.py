import random
from collections import namedtuple, deque


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        """Returns a random batch of transition samples"""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
