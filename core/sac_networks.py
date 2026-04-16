"""
sac_networks.py — Redes neuronais do SAC para controlo do BSM2
Arquitectura:
  - Actor  : política Gaussiana (média + log_std)
  - Critic : dois Q-networks (para reduzir overestimation)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reprodutibilidade
torch.manual_seed(42)

LOG_STD_MIN = -5
LOG_STD_MAX = 2


# =====================================================
# BLOCO MLP partilhado
# =====================================================

def mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU):
    layers = []
    dims = [input_dim] + list(hidden_dims)
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i+1]), activation()]
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


# =====================================================
# ACTOR — política Gaussiana
# Saída: média e log_std da distribuição da ação
# =====================================================

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim,
                 hidden=(256, 256),
                 action_low=-1.0, action_high=1.0):
        super().__init__()

        self.net     = mlp(state_dim, hidden[:-1], hidden[-1])
        self.mu_head = nn.Linear(hidden[-1], action_dim)
        self.ls_head = nn.Linear(hidden[-1], action_dim)

        # Escala para desnormalizar ação [-1,1] → [low, high]
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
        Amostra uma ação usando o reparametrization trick.
        Retorna:
          action_env : ação no espaço real [action_low, action_high]
          log_prob   : log probabilidade (para cálculo de entropia)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)

        # Amostra no espaço não limitado
        x_t = dist.rsample()

        # Squash para [-1, 1] com tanh
        y_t = torch.tanh(x_t)

        # Log prob com correcção do tanh (SAC paper, eq. 21)
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Desnormalizar para espaço real
        action_env = y_t * self.scale + self.bias

        return action_env, log_prob

    def deterministic(self, state):
        """Ação determinística para avaliação (sem exploração)."""
        mu, _ = self.forward(state)
        y_t   = torch.tanh(mu)
        return y_t * self.scale + self.bias


# =====================================================
# CRITIC — dois Q-networks
# =====================================================

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden=(256, 256)):
        super().__init__()
        # Q1 e Q2 independentes
        self.q1 = mlp(state_dim + action_dim, hidden, 1)
        self.q2 = mlp(state_dim + action_dim, hidden, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_only(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)
