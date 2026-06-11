"""
sac_networks.py — SAC neural networks for BSM2 control.
Architecture:
  - Actor  : Gaussian policy (mean + log_std)
  - Critic : two Q-networks (to reduce overestimation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reproducibility
torch.manual_seed(42)

LOG_STD_MIN = -5
LOG_STD_MAX = 2


# =====================================================
# Shared MLP block
# =====================================================

def mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU):
    layers = []
    dims = [input_dim] + list(hidden_dims)
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i+1]), activation()]
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


# =====================================================
# ACTOR — Gaussian policy
# Output: mean and log_std of the action distribution
# =====================================================

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim,
                 hidden=(256, 256),
                 action_low=-1.0, action_high=1.0):
        super().__init__()

        self.net     = mlp(state_dim, hidden[:-1], hidden[-1])
        self.mu_head = nn.Linear(hidden[-1], action_dim)
        self.ls_head = nn.Linear(hidden[-1], action_dim)

        # Scale to denormalize action [-1,1] → [low, high]
        self.register_buffer("scale",
            torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("bias",
            torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32))

    def forward(self, state):
        h       = F.relu(self.net(state))
        mu      = self.mu_head(h)
        log_std = self.ls_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        """
        Sample an action using the reparametrization trick.
        Returns:
          action_env : action in real space [action_low, action_high]
          log_prob   : log probability (for the entropy term)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)

        # Sample in the unbounded space
        x_t = dist.rsample()

        # Squash to [-1, 1] with tanh
        y_t = torch.tanh(x_t)

        # Log prob with tanh correction (SAC paper, eq. 21)
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Denormalize to real space
        action_env = y_t * self.scale + self.bias

        return action_env, log_prob

    def deterministic(self, state):
        """Deterministic action for evaluation (no exploration)."""
        mu, _ = self.forward(state)
        y_t   = torch.tanh(mu)
        return y_t * self.scale + self.bias


# =====================================================
# CRITIC — two Q-networks
# =====================================================

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden=(256, 256)):
        super().__init__()
        # Independent Q1 and Q2
        self.q1 = mlp(state_dim + action_dim, hidden, 1)
        self.q2 = mlp(state_dim + action_dim, hidden, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_only(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)
