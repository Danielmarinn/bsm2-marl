"""
replay_buffer.py — Replay buffer for SAC on BSM2.
Stores experiences (s, a, r, s', done) for off-policy training.
"""

import numpy as np


class ReplayBuffer:
    """
    Fixed-size circular buffer.
    Each experience is a tuple (state, action, reward, next_state, done).
    """

    def __init__(self, state_dim, action_dim, max_size=50_000):
        self.max_size  = max_size
        self.ptr       = 0      # write position
        self.size      = 0      # number of stored experiences

        self.states      = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((max_size, 1),          dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((max_size, 1),          dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = done

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size
