"""
replay_buffer.py — Replay Buffer para SAC no BSM2
Armazena experiências (s, a, r, s', done) para treino off-policy.
"""

import numpy as np


class ReplayBuffer:
    """
    Buffer circular de tamanho fixo.
    Cada experiência é um tuplo (state, action, reward, next_state, done).
    """

    def __init__(self, state_dim, action_dim, max_size=50_000):
        self.max_size  = max_size
        self.ptr       = 0      # posição de escrita
        self.size      = 0      # número de experiências armazenadas

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
