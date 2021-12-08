from collections import namedtuple
import random
import math
from collections import deque
import numpy as np
import torch

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, done):
        # state = np.expand_dims(state, 0)
        # next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = map(
            np.stack, zip(*random.sample(self.buffer, batch_size))
        )
        return state, action, reward, next_state, done
        # return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class HERBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def push(self, goal, obs_t, action, reward, obs_tp1, done):
        data = (goal, obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        goals, obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            goal, obs_t, action, reward, obs_tp1, done = data
            goals.append(np.array(goal.cpu(), copy=False))
            obses_t.append(np.array(obs_t.cpu(), copy=False))
            actions.append(np.array(action.cpu(), copy=False))
            rewards.append(reward.cpu())
            obses_tp1.append(np.array(obs_tp1.cpu(), copy=False))
            dones.append(done.cpu())
        return (
            np.array(goals),
            np.array(obses_t),
            np.array(actions),
            np.array(rewards),
            np.array(obses_tp1),
            np.array(dones),
        )

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        goals: np.array
        obses_t: np.array
        actions: np.array
        rewards: np.array
        obses_tp1: np.array
        dones: np.array
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
