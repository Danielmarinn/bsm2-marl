"""
G2ANet-inspired neural blocks for the BSM2 multi-agent controller.

This module is intentionally isolated from the current SAC controllers. It
implements the architectural pieces we need next: per-agent encoding,
hard-attention communication gates, soft-attention weights, graph/message
aggregation, and a central critic over the joint action.

It is not yet a complete training algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class G2ANetConfig:
    obs_dims: tuple[int, ...]
    action_dims: tuple[int, ...]
    embed_dim: int = 64
    hidden_dim: int = 128
    temperature: float = 1.0

    @property
    def n_agents(self) -> int:
        return len(self.obs_dims)

    @property
    def joint_action_dim(self) -> int:
        return sum(self.action_dims)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class AgentEncoder(nn.Module):
    """Encode one agent observation-action pair into a compact embedding."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        # Simplified to a robust MLP to avoid the 'stateless GRU' issue
        # identified in the architectural review.
        self.net = _mlp(obs_dim + action_dim, hidden_dim, embed_dim)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class HardAttentionGate(nn.Module):
    """
    Learn directed communication links between agents.

    Output shape is [batch, n_agents, n_agents]. Entry i,j is the relaxed gate
    for information flowing from source agent j to target agent i. Self-links
    are forced to zero.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.link_net = _mlp(embed_dim * 2, hidden_dim, 2)

    def forward(self, embeddings: torch.Tensor, hard: bool = False) -> torch.Tensor:
        batch, n_agents, embed_dim = embeddings.shape
        target = embeddings.unsqueeze(2).expand(batch, n_agents, n_agents, embed_dim)
        source = embeddings.unsqueeze(1).expand(batch, n_agents, n_agents, embed_dim)
        pair = torch.cat([target, source], dim=-1)
        logits = self.link_net(pair)

        gates = F.gumbel_softmax(
            logits,
            tau=self.temperature,
            hard=hard,
            dim=-1,
        )[..., 1]

        eye = torch.eye(n_agents, device=embeddings.device, dtype=gates.dtype)
        return gates * (1.0 - eye.unsqueeze(0))


class SoftAttention(nn.Module):
    """Scaled dot-product attention between selected agent embeddings."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(
        self,
        embeddings: torch.Tensor,
        gates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query(embeddings)
        k = self.key(embeddings)
        v = self.value(embeddings)

        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale
        mask = gates <= 1e-6
        scores = scores.masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=-1) * gates
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        messages = torch.matmul(weights, v)
        return self.out(messages), weights


class G2ANetCentralCritic(nn.Module):
    """
    Central critic with G2ANet-style communication.

    Actors can remain individual; this critic consumes all observations and
    actions, learns inter-agent links, aggregates messages, then predicts Q_tot.
    """

    def __init__(self, config: G2ANetConfig):
        super().__init__()
        if len(config.obs_dims) != len(config.action_dims):
            raise ValueError("obs_dims and action_dims must have the same length.")
        self.config = config
        self.encoders = nn.ModuleList(
            AgentEncoder(obs_dim, action_dim, config.hidden_dim, config.embed_dim)
            for obs_dim, action_dim in zip(config.obs_dims, config.action_dims)
        )
        self.hard_attention = HardAttentionGate(
            config.embed_dim,
            config.hidden_dim,
            config.temperature,
        )
        self.soft_attention = SoftAttention(config.embed_dim)
        # Removed redundant joint_action_dim to ensure gradients flow
        # primarily through the attention-based embeddings.
        critic_input_dim = config.n_agents * config.embed_dim
        self.q_head = _mlp(critic_input_dim, config.hidden_dim, 1)

    def forward(
        self,
        observations: Sequence[torch.Tensor],
        actions: Sequence[torch.Tensor],
        hard: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if len(observations) != self.config.n_agents or len(actions) != self.config.n_agents:
            raise ValueError("Expected one observation and one action tensor per agent.")

        embeddings = torch.stack(
            [
                encoder(obs, action)
                for encoder, obs, action in zip(self.encoders, observations, actions)
            ],
            dim=1,
        )
        hard_gates = self.hard_attention(embeddings, hard=hard)
        messages, soft_weights = self.soft_attention(embeddings, hard_gates)
        # Joint embedding now contains the fully attended information
        joint_embedding = (embeddings + messages).reshape(embeddings.shape[0], -1)
        q_tot = self.q_head(joint_embedding)

        info = {
            "embeddings": embeddings,
            "hard_gates": hard_gates,
            "soft_weights": soft_weights,
            "messages": messages,
        }
        return q_tot, info
