"""
Recurrent attention critic blocks for the Game2 BSM2 controller.

This is a G2ANet-inspired stack adapted for continuous-action CTDE-SAC:
per-agent GRU memory, actor-side soft/hard peer attention, critic-side soft
inter-agent attention, and a scalar Q_tot head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RecurrentG2ANetConfig:
    obs_dims: tuple[int, ...]
    action_dims: tuple[int, ...]
    embed_dim: int = 64
    hidden_dim: int = 128
    use_prev_action_memory: bool = True
    layer_norm: bool = True

    @property
    def n_agents(self) -> int:
        return len(self.obs_dims)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class RecurrentAgentEncoder(nn.Module):
    """
    Encode one agent's observation history and current action.

    The GRU receives the observation and previous action. The current action is
    injected after the temporal encoder so actor gradients can evaluate
    "what is this action worth in this process context?" without rewriting the
    memory state with the same action being scored.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        embed_dim: int,
        use_prev_action_memory: bool = True,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.use_prev_action_memory = use_prev_action_memory
        memory_input_dim = obs_dim + (action_dim if use_prev_action_memory else 0)
        self.input_proj = nn.Linear(memory_input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, embed_dim, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()
        self.action_embed = _mlp(embed_dim + action_dim, hidden_dim, embed_dim)

    def _previous_actions(self, actions: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(actions[:, :1, :])
        return torch.cat([zeros, actions[:, :-1, :]], dim=1)

    def forward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        if obs_seq.ndim != 3 or action_seq.ndim != 3:
            raise ValueError("Expected [batch, seq, dim] tensors.")

        if self.use_prev_action_memory:
            prev_action = self._previous_actions(action_seq.detach())
            memory_in = torch.cat([obs_seq, prev_action], dim=-1)
        else:
            memory_in = obs_seq

        z = torch.relu(self.input_proj(memory_in))
        h, _ = self.gru(z)
        h = self.norm(h)
        return self.action_embed(torch.cat([h, action_seq], dim=-1))


class ActorRecurrentEncoder(nn.Module):
    """
    Per-agent recurrent observation encoder for the decentralized actor.

    Mirrors the shape of RecurrentAgentEncoder (Linear -> GRU -> LayerNorm)
    but produces an action-free embedding so the actor head can be
    conditioned on temporal context without leakage of the current action
    being scored.

    Supports two call modes:
      * forward(obs_seq, prev_action_seq=None, h0=None) for training over
        contiguous sequence chunks; returns (emb_seq, h_n).
      * step(obs, prev_action=None, h_prev=...) for streaming inference at
        the coordinator; returns (emb, h_next).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        embed_dim: int,
        use_prev_action_memory: bool = False,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.use_prev_action_memory = use_prev_action_memory
        memory_input_dim = obs_dim + (action_dim if use_prev_action_memory else 0)
        self.input_proj = nn.Linear(memory_input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, embed_dim, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()

    def init_hidden(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.embed_dim, device=device)

    def _pad_prev_actions(self, action_seq: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(action_seq[:, :1, :])
        return torch.cat([zeros, action_seq[:, :-1, :]], dim=1)

    def forward(
        self,
        obs_seq: torch.Tensor,
        prev_action_seq: torch.Tensor | None = None,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if obs_seq.ndim != 3:
            raise ValueError("Expected obs_seq shaped [batch, seq, obs_dim].")
        if self.use_prev_action_memory:
            if prev_action_seq is None:
                raise ValueError(
                    "ActorRecurrentEncoder.use_prev_action_memory=True requires prev_action_seq."
                )
            if prev_action_seq.ndim != 3:
                raise ValueError("Expected prev_action_seq shaped [batch, seq, action_dim].")
            memory_in = torch.cat([obs_seq, prev_action_seq.detach()], dim=-1)
        else:
            memory_in = obs_seq
        z = torch.relu(self.input_proj(memory_in))
        h_out, h_n = self.gru(z, h0)
        return self.norm(h_out), h_n

    def step(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor | None,
        h_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim != 2:
            raise ValueError("Expected obs shaped [batch, obs_dim].")
        obs_seq = obs.unsqueeze(1)
        if self.use_prev_action_memory:
            if prev_action is None:
                prev_action = torch.zeros(
                    obs.shape[0], self.action_dim, device=obs.device, dtype=obs.dtype
                )
            action_seq = prev_action.unsqueeze(1)
        else:
            action_seq = None
        emb_seq, h_next = self.forward(obs_seq, action_seq, h_prev)
        return emb_seq.squeeze(1), h_next


class ActorSoftAttention(nn.Module):
    """Soft attention over peer embeddings for one decentralized actor.

    Given per-agent embeddings of shape [..., n_agents, embed_dim], computes a
    context vector for the designated self agent by attending over the other
    agents. Self-attention is masked out. An optional `gate` tensor of shape
    [..., n_agents] (values in [0, 1]) multiplies the attention weights before
    renormalization, providing the integration point for hard attention.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(
        self,
        peer_embs: torch.Tensor,
        self_index: int,
        gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_agents = peer_embs.shape[-2]
        if not 0 <= self_index < n_agents:
            raise ValueError("self_index out of range.")
        self_emb = peer_embs[..., self_index, :]
        q = self.query(self_emb).unsqueeze(-2)
        k = self.key(peer_embs)
        v = torch.relu(self.value(peer_embs))
        scores = (q @ k.transpose(-1, -2)) / self.scale
        self_mask = torch.zeros(n_agents, dtype=torch.bool, device=peer_embs.device)
        self_mask[self_index] = True
        scores = scores.masked_fill(self_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        if gate is not None:
            weights = weights * gate.unsqueeze(-2)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        weights = weights.masked_fill(self_mask, 0.0)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        ctx = (weights @ v).squeeze(-2)
        return self.out(ctx), weights.squeeze(-2)


class ActorHardAttention(nn.Module):
    """Straight-through binary peer gate for one decentralized actor."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.threshold = threshold
        self.scorer = nn.Sequential(
            nn.Linear(4 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        peer_embs: torch.Tensor,
        self_index: int,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if peer_embs.ndim < 2:
            raise ValueError("Expected peer_embs shaped [..., n_agents, embed_dim].")
        n_agents = peer_embs.shape[-2]
        if not 0 <= self_index < n_agents:
            raise ValueError("self_index out of range.")

        self_emb = peer_embs[..., self_index : self_index + 1, :]
        self_expanded = self_emb.expand_as(peer_embs)
        pair_features = torch.cat(
            [
                self_expanded,
                peer_embs,
                torch.abs(self_expanded - peer_embs),
                self_expanded * peer_embs,
            ],
            dim=-1,
        )

        logits = self.scorer(pair_features).squeeze(-1)
        logits = logits / max(self.temperature, 1e-6)
        self_mask = torch.zeros(n_agents, dtype=torch.bool, device=peer_embs.device)
        self_mask[self_index] = True

        probs = torch.sigmoid(logits).masked_fill(self_mask, 0.0)
        if self.training and not deterministic:
            u = torch.rand_like(probs).clamp(1e-6, 1.0 - 1e-6)
            logistic_noise = torch.log(u) - torch.log1p(-u)
            soft_gate = torch.sigmoid((logits + logistic_noise) / max(self.temperature, 1e-6))
            soft_gate = soft_gate.masked_fill(self_mask, 0.0)
        else:
            soft_gate = probs

        hard_gate = (soft_gate >= self.threshold).to(soft_gate.dtype)
        hard_gate = hard_gate.masked_fill(self_mask, 0.0)
        gate = hard_gate.detach() - soft_gate.detach() + soft_gate

        active = hard_gate.sum(dim=-1, keepdim=True) > 0.0
        best_idx = probs.masked_fill(self_mask, -1.0).argmax(dim=-1, keepdim=True)
        fallback_hard = torch.zeros_like(probs).scatter(-1, best_idx, 1.0)
        fallback_hard = fallback_hard.masked_fill(self_mask, 0.0)
        fallback_gate = fallback_hard.detach() - probs.detach() + probs
        gate = torch.where(active, gate, fallback_gate)

        if n_agents > 2:
            gate_hard = (gate.detach() >= self.threshold).to(gate.dtype)
            too_many = gate_hard.sum(dim=-1, keepdim=True) >= float(n_agents - 1)
            weakest_idx = probs.masked_fill(self_mask, 2.0).argmin(dim=-1, keepdim=True)
            capped_hard = gate_hard.scatter(-1, weakest_idx, 0.0)
            capped_gate = capped_hard.detach() - probs.detach() + probs
            gate = torch.where(too_many, capped_gate, gate)

        gate = gate.masked_fill(self_mask, 0.0)
        return gate, probs


class TemporalSoftAttention(nn.Module):
    """Soft attention over agents at each timestep.

    Accepts an optional pairwise gate that masks the attention weights before
    renormalization. The current central critic is soft-only and calls this
    without a gate; the parameter is retained as a general capability. (A hard
    gate on a value-regression critic has no sparsity incentive and collapses
    to fully-connected -- the faithful G2ANet hard gate is on the actor.)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(
        self,
        embeddings: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 4:
            raise ValueError("Expected embeddings shaped [batch, seq, agents, embed].")
        batch, seq_len, n_agents, _ = embeddings.shape

        q = self.query(embeddings)
        k = self.key(embeddings)
        v = torch.relu(self.value(embeddings))

        scores = torch.einsum("btie,btje->btij", q, k) / self.scale
        eye = torch.eye(n_agents, device=embeddings.device, dtype=torch.bool)
        scores = scores.masked_fill(eye.view(1, 1, n_agents, n_agents), -1e9)
        weights = torch.softmax(scores, dim=-1)
        if gate is not None:
            weights = weights * gate
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        weights = weights.masked_fill(eye.view(1, 1, n_agents, n_agents), 0.0)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        messages = torch.einsum("btij,btje->btie", weights, v)
        return self.out(messages), weights


class RecurrentAttentionCentralCritic(nn.Module):
    """
    Central recurrent attention critic returning Q_tot for every timestep.
    """

    def __init__(self, config: RecurrentG2ANetConfig):
        super().__init__()
        if len(config.obs_dims) != len(config.action_dims):
            raise ValueError("obs_dims and action_dims must have the same length.")
        self.config = config
        self.encoders = nn.ModuleList(
            RecurrentAgentEncoder(
                obs_dim,
                action_dim,
                config.hidden_dim,
                config.embed_dim,
                config.use_prev_action_memory,
                config.layer_norm,
            )
            for obs_dim, action_dim in zip(config.obs_dims, config.action_dims)
        )
        self.attention = TemporalSoftAttention(config.embed_dim)
        self.q_head = _mlp(config.n_agents * config.embed_dim, config.hidden_dim, 1)

    def forward(
        self,
        observations: Sequence[torch.Tensor],
        actions: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if len(observations) != self.config.n_agents or len(actions) != self.config.n_agents:
            raise ValueError("Expected one observation and action sequence per agent.")

        embeddings = torch.stack(
            [
                encoder(obs_seq, act_seq)
                for encoder, obs_seq, act_seq in zip(self.encoders, observations, actions)
            ],
            dim=2,
        )
        # Soft-only central critic. Hard (discrete) gating belongs on the ACTOR
        # in G2ANet (ActorHardAttention); a hard gate on a value-regression
        # critic has no sparsity incentive and collapses to fully-connected, so
        # it is intentionally not used here.
        messages, soft_weights = self.attention(embeddings)
        joint = (embeddings + messages).reshape(embeddings.shape[0], embeddings.shape[1], -1)
        q_tot = self.q_head(joint)
        return q_tot, {
            "embeddings": embeddings,
            "messages": messages,
            "soft_weights": soft_weights,
        }
