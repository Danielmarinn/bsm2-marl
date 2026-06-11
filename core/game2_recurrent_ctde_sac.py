"""
Recurrent Attention CTDE-SAC for the BSM2 Game2 controller.

This keeps decentralized continuous SAC actors and replaces the feed-forward
central critic with a recurrent soft-attention critic trained on contiguous
sequence chunks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from game2_abstraction import ACTION_DELTA_BOUNDS, AGENT_ORDER
from game2_sac import AGENT_LOCAL_DIMS, build_normalisation
from recurrent_g2anet import (
    ActorHardAttention,
    ActorRecurrentEncoder,
    ActorSoftAttention,
    RecurrentAttentionCentralCritic,
    RecurrentG2ANetConfig,
)
from sac_networks import Actor


AGENT_NAMES: list[str] = list(AGENT_ORDER)


@dataclass(frozen=True)
class RecurrentCTDESACConfig:
    hidden: tuple[int, int] = (256, 256)
    embed_dim: int = 64
    hidden_dim: int = 128

    # Asymmetric rates: critic faster than actor. This is the two-timescale
    # actor-critic principle (Konda & Tsitsiklis 2000) -- a slower actor lets
    # the centralized critic track a more stable target. NOT from Nam et al.
    # (he reports no learning rates and used Central-V, not SAC).
    lr_actor: float = 5e-5
    lr_critic: float = 3e-4
    # (no lr_alpha: alpha is fixed, see config.fixed_alpha -- no alpha optimizer)

    gamma: float = 0.99
    tau: float = 0.005

    batch_size: int = 32          # sequences, not individual rows
    buffer_size: int = 200_000
    # 5_000 steps (~52 sim days) of uniform random delta exploration before
    # training kicks in. The critic needs a wide experience window to form
    # stable Q-targets before the actor starts following its gradient.
    warmup_steps: int = 5_000
    seq_len: int = 32
    burn_in: int = 8

    grad_clip: float = 5.0
    updates_per_step: int = 1

    # Disable SAC alpha auto-tuning. Per-agent auto-tune on a shared joint
    # reward can collapse alpha and kill exploration. Fixed alpha keeps
    # entropy regularization stable throughout training and is the standard
    # choice in cooperative MARL with a shared reward.
    fixed_alpha: float = 0.2


class SequenceReplayBuffer:
    """Ring buffer sampled as contiguous chunks. Designed for one long BSM2 run."""

    def __init__(self, obs_dims: list[int], n_agents: int, buffer_size: int):
        self.obs_dims = obs_dims
        self.n_agents = n_agents
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        self.has_wrapped = False
        self._obs = [np.zeros((buffer_size, d), dtype=np.float32) for d in obs_dims]
        self._next_obs = [np.zeros((buffer_size, d), dtype=np.float32) for d in obs_dims]
        self._actions = [np.zeros((buffer_size, 1), dtype=np.float32) for _ in range(n_agents)]
        self._reward = np.zeros((buffer_size, 1), dtype=np.float32)
        self._done = np.zeros((buffer_size, 1), dtype=np.float32)

    def add(
        self,
        obs_list: list[np.ndarray],
        action_list: list[float],
        reward: float,
        next_obs_list: list[np.ndarray],
        done: bool,
    ) -> None:
        idx = self.ptr
        for i in range(self.n_agents):
            self._obs[i][idx] = obs_list[i]
            self._next_obs[i][idx] = next_obs_list[i]
            self._actions[i][idx, 0] = action_list[i]
        self._reward[idx, 0] = reward
        self._done[idx, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.has_wrapped = self.has_wrapped or self.ptr == 0
        self.size = min(self.size + 1, self.buffer_size)

    def sample_sequences(self, batch_size: int, seq_len: int) -> tuple[
        list[np.ndarray], list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray
    ]:
        if self.has_wrapped:
            raise RuntimeError("SequenceReplayBuffer wrapping is not supported for recurrent sampling yet.")
        if self.size < seq_len + 1:
            raise ValueError("Not enough contiguous rows for sequence sampling.")
        max_start = self.size - seq_len
        starts = np.random.randint(0, max_start + 1, size=batch_size)
        idx = starts[:, None] + np.arange(seq_len)[None, :]
        obs = [self._obs[i][idx] for i in range(self.n_agents)]
        actions = [self._actions[i][idx] for i in range(self.n_agents)]
        rewards = self._reward[idx]
        next_obs = [self._next_obs[i][idx] for i in range(self.n_agents)]
        done = self._done[idx]
        return obs, actions, rewards, next_obs, done

    def __len__(self) -> int:
        return self.size


class TwinRecurrentAttentionCritic(nn.Module):
    def __init__(self, config: RecurrentG2ANetConfig):
        super().__init__()
        self.q1 = RecurrentAttentionCentralCritic(config)
        self.q2 = RecurrentAttentionCentralCritic(config)

    def forward(
        self,
        observations: Sequence[torch.Tensor],
        actions: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1, _ = self.q1(observations, actions)
        q2, _ = self.q2(observations, actions)
        return q1, q2


class RecurrentCTDEMultiAgentSAC:
    """Decentralized MLP actors trained by a recurrent centralized critic."""

    def __init__(self, config: RecurrentCTDESACConfig):
        if config.seq_len <= config.burn_in + 1:
            raise ValueError("seq_len must be greater than burn_in + 1.")
        self.config = config
        self.agent_names = AGENT_NAMES
        self.actors: dict[str, Actor] = {}
        self.actor_encoders: dict[str, ActorRecurrentEncoder] = {}
        self.actor_attentions: dict[str, ActorSoftAttention] = {}
        self.actor_hard_attentions: dict[str, ActorHardAttention] = {}
        self.opt_actors: dict[str, torch.optim.Adam] = {}
        self.log_alphas: dict[str, torch.Tensor] = {}
        self.state_means: dict[str, torch.Tensor] = {}
        self.state_stds: dict[str, torch.Tensor] = {}
        self._actor_h: dict[str, torch.Tensor | None] = {n: None for n in self.agent_names}

        for name in self.agent_names:
            obs_dim = AGENT_LOCAL_DIMS[name]
            low, high = ACTION_DELTA_BOUNDS[name]
            self.actor_encoders[name] = ActorRecurrentEncoder(
                obs_dim=obs_dim,
                action_dim=1,
                hidden_dim=config.hidden_dim,
                embed_dim=config.embed_dim,
                use_prev_action_memory=False,
            )
            self.actor_attentions[name] = ActorSoftAttention(embed_dim=config.embed_dim)
            self.actor_hard_attentions[name] = ActorHardAttention(
                embed_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
            )
            self.actors[name] = Actor(2 * config.embed_dim, 1, config.hidden, low, high)
            self.opt_actors[name] = torch.optim.Adam(
                list(self.actor_encoders[name].parameters())
                + list(self.actor_attentions[name].parameters())
                + list(self.actor_hard_attentions[name].parameters())
                + list(self.actors[name].parameters()),
                lr=config.lr_actor,
            )
            # Fixed alpha (no auto-tune). log_alpha is a non-trainable buffer.
            # See config.fixed_alpha.
            log_alpha = torch.tensor(
                math.log(config.fixed_alpha), requires_grad=False
            )
            self.log_alphas[name] = log_alpha
            mean, std = build_normalisation(name, obs_dim)
            self.state_means[name] = torch.as_tensor(mean, dtype=torch.float32)
            self.state_stds[name] = torch.as_tensor(std + 1e-8, dtype=torch.float32)

        critic_cfg = RecurrentG2ANetConfig(
            obs_dims=tuple(AGENT_LOCAL_DIMS[n] for n in self.agent_names),
            action_dims=tuple(1 for _ in self.agent_names),
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
        )
        self.critic = TwinRecurrentAttentionCritic(critic_cfg)
        self.critic_tgt = TwinRecurrentAttentionCritic(critic_cfg)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        self.critic_tgt.requires_grad_(False)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        self._act_low = torch.tensor([ACTION_DELTA_BOUNDS[n][0] for n in self.agent_names], dtype=torch.float32)
        self._act_high = torch.tensor([ACTION_DELTA_BOUNDS[n][1] for n in self.agent_names], dtype=torch.float32)

        self.buffer = SequenceReplayBuffer(
            obs_dims=[AGENT_LOCAL_DIMS[n] for n in self.agent_names],
            n_agents=len(self.agent_names),
            buffer_size=config.buffer_size,
        )
        self.total_steps = 0

    def _norm(self, name: str, obs: np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        return (t - self.state_means[name]) / self.state_stds[name]

    def _norm_batch_seq(self, i: int, obs: np.ndarray) -> torch.Tensor:
        name = self.agent_names[i]
        t = torch.as_tensor(obs, dtype=torch.float32)
        return (t - self.state_means[name]) / self.state_stds[name]

    def _norm_actions(self, acts: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [
            2.0 * (acts[i] - self._act_low[i]) / (self._act_high[i] - self._act_low[i] + 1e-8) - 1.0
            for i in range(len(self.agent_names))
        ]

    def _ensure_actor_hidden(self, name: str, device: torch.device) -> torch.Tensor:
        h = self._actor_h[name]
        if h is None:
            h = self.actor_encoders[name].init_hidden(batch_size=1, device=device)
            self._actor_h[name] = h
        return h

    def reset_actor_state(self) -> None:
        for name in self.agent_names:
            self._actor_h[name] = None

    def select_actions(
        self,
        obs_dict: Mapping[str, np.ndarray],
        rng: np.random.Generator,
        deterministic: bool = False,
    ) -> dict[str, float]:
        if self.total_steps < self.config.warmup_steps and not deterministic:
            return {name: float(rng.uniform(*ACTION_DELTA_BOUNDS[name])) for name in self.agent_names}
        actions: dict[str, float] = {}
        with torch.no_grad():
            embs: list[torch.Tensor] = []
            for name in self.agent_names:
                s = self._norm(name, obs_dict[name]).unsqueeze(0)
                h_prev = self._ensure_actor_hidden(name, s.device)
                emb, h_next = self.actor_encoders[name].step(s, None, h_prev)
                self._actor_h[name] = h_next.detach()
                embs.append(emb)
            peer_embs = torch.stack(embs, dim=1)
            for i, name in enumerate(self.agent_names):
                gate, _ = self.actor_hard_attentions[name](
                    peer_embs,
                    i,
                    deterministic=True,
                )
                context, _ = self.actor_attentions[name](peer_embs, i, gate=gate)
                actor_in = torch.cat([embs[i], context], dim=-1)
                if deterministic:
                    a = self.actors[name].deterministic(actor_in)
                else:
                    a, _ = self.actors[name].sample(actor_in)
                actions[name] = float(a.squeeze().item())
        return actions

    def add_transition(
        self,
        obs_dict: Mapping[str, np.ndarray],
        actions_dict: Mapping[str, float],
        reward_total: float,
        next_obs_dict: Mapping[str, np.ndarray],
        done: bool,
    ) -> None:
        self.buffer.add(
            [obs_dict[n] for n in self.agent_names],
            [actions_dict[n] for n in self.agent_names],
            reward_total,
            [next_obs_dict[n] for n in self.agent_names],
            done,
        )
        self.total_steps += 1
        if done:
            self.reset_actor_state()

    def _sample_actor_with_attention(
        self,
        agent_idx: int,
        all_obs_seq: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.agent_names[agent_idx]
        batch, seq_len, _ = all_obs_seq[agent_idx].shape
        embs: list[torch.Tensor] = []
        for j, n in enumerate(self.agent_names):
            emb_j, _ = self.actor_encoders[n](all_obs_seq[j])
            if j != agent_idx:
                emb_j = emb_j.detach()
            embs.append(emb_j)
        peer_embs = torch.stack(embs, dim=2)
        gate, _ = self.actor_hard_attentions[name](peer_embs, agent_idx)
        context, _ = self.actor_attentions[name](peer_embs, agent_idx, gate=gate)
        actor_in = torch.cat([embs[agent_idx], context], dim=-1)
        flat = actor_in.reshape(batch * seq_len, 2 * self.config.embed_dim)
        action, logp = self.actors[name].sample(flat)
        return action.reshape(batch, seq_len, 1), logp.reshape(batch, seq_len, 1)

    def _valid_mask(self, done_t: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(done_t)
        mask[:, : self.config.burn_in, :] = 0.0
        return mask * (1.0 - done_t)

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (((pred - target) ** 2) * mask).sum() / (mask.sum() + 1e-8)

    @torch.no_grad()
    def _actor_attention_stats(
        self,
        all_obs_seq: list[torch.Tensor],
        mask: torch.Tensor,
    ) -> dict[str, float]:
        """Compact scalars that make the ACTOR's peer attention auditable in
        the CSV. This is where the faithful G2ANet hard attention lives
        (ActorHardAttention, capped so it cannot collapse) -- the place where
        success criteria 1-2 must be verified. The central critic is soft-only
        (no hard gate), so nothing critic-side is logged here.

        Pure logging -- no gradient, never touches training. The deterministic
        gate (the deployed graph) is used. Stats are averaged only over *valid*
        timesteps (``mask`` excludes the burn-in prefix and terminal steps).

        Returned scalars:
          * actor_hard_open_frac -- fraction of directed agent-pairs the hard
            gate keeps open. Capped to ~1-2 peers/agent, so healthy is well
            below 1.0 (collapsed) and above 0 (dead): a learned sparse graph.
          * actor_hard_std -- spread of the hard gate across pairs (>0 = some
            pairs open, others closed = structured graph).
          * actor_soft_entropy -- mean entropy (nats) of the soft attention
            over the gated peers; lower => more focused.
          * actor_soft_max -- mean of the largest peer weight (higher => more
            focused).
        """
        embs = [
            self.actor_encoders[n](all_obs_seq[j])[0]
            for j, n in enumerate(self.agent_names)
        ]
        peer = torch.stack(embs, dim=2)  # [b, s, n_agents, embed]
        b, s, n, _ = peer.shape
        gates, softs = [], []
        for i, name in enumerate(self.agent_names):
            gate, _ = self.actor_hard_attentions[name](peer, i, deterministic=True)
            _, w = self.actor_attentions[name](peer, i, gate=gate)
            gates.append((gate >= 0.5).to(peer.dtype))  # [b, s, n]
            softs.append(w)                              # [b, s, n]
        G = torch.stack(gates, dim=2)  # [b, s, i, j]
        W = torch.stack(softs, dim=2)  # [b, s, i, j]

        eye = torch.eye(n, device=peer.device, dtype=peer.dtype)
        offdiag = (1.0 - eye).view(1, 1, n, n)
        n_pairs = float(n * (n - 1))
        wm = mask.detach().reshape(b, s)
        w4 = wm.reshape(b, s, 1, 1)
        wt = wm.sum().clamp_min(1e-8)

        G_off = G * offdiag
        open_frac = (G_off * w4).sum() / (wt * n_pairs)
        var = (((G_off - open_frac) ** 2) * offdiag * w4).sum() / (wt * n_pairs)
        hard_std = var.clamp_min(0.0).sqrt()

        ent = -(W * torch.log(W + 1e-8)).sum(dim=-1).mean(dim=-1)  # [b, s]
        soft_entropy = (ent * wm).sum() / wt
        peak = W.max(dim=-1).values.mean(dim=-1)  # [b, s]
        soft_max = (peak * wm).sum() / wt

        return {
            "actor_hard_open_frac": float(open_frac),
            "actor_hard_std": float(hard_std),
            "actor_soft_entropy": float(soft_entropy),
            "actor_soft_max": float(soft_max),
        }

    def train_step(self) -> Mapping[str, float] | None:
        if self.total_steps < self.config.warmup_steps:
            return None
        if len(self.buffer) < max(self.config.seq_len + 1, self.config.batch_size):
            return None

        obs_b, act_b, rew_b, nobs_b, done_b = self.buffer.sample_sequences(
            self.config.batch_size,
            self.config.seq_len,
        )

        obs_n = [self._norm_batch_seq(i, obs_b[i]) for i in range(len(self.agent_names))]
        nobs_n = [self._norm_batch_seq(i, nobs_b[i]) for i in range(len(self.agent_names))]
        acts_t = [torch.as_tensor(act_b[i], dtype=torch.float32) for i in range(len(self.agent_names))]
        rew_t = torch.as_tensor(rew_b, dtype=torch.float32)
        done_t = torch.as_tensor(done_b, dtype=torch.float32)
        mask = self._valid_mask(done_t)

        with torch.no_grad():
            next_acts, next_logps = [], []
            for i, name in enumerate(self.agent_names):
                a_next, lp_next = self._sample_actor_with_attention(i, nobs_n)
                next_acts.append(a_next)
                next_logps.append(lp_next)

            entropy_term = sum(
                self.log_alphas[name].exp().detach() * next_logps[i]
                for i, name in enumerate(self.agent_names)
            )
            q1_t, q2_t = self.critic_tgt(nobs_n, self._norm_actions(next_acts))
            q_tgt = torch.min(q1_t, q2_t) - entropy_term
            target = rew_t + self.config.gamma * (1.0 - done_t) * q_tgt

        self.opt_critic.zero_grad()
        q1, q2 = self.critic(obs_n, self._norm_actions(acts_t))
        critic_loss = self._masked_mse(q1, target, mask) + self._masked_mse(q2, target, mask)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.opt_critic.step()

        result: dict[str, float] = {"critic_loss": float(critic_loss.item())}
        # Log the ACTOR's peer attention (criteria 1-2). Logging only, no grad.
        result.update(self._actor_attention_stats(obs_n, mask))

        for p in self.critic.parameters():
            p.requires_grad_(False)
        for i, name in enumerate(self.agent_names):
            a_pi, logp = self._sample_actor_with_attention(i, obs_n)
            joint = list(acts_t)
            joint[i] = a_pi

            q1_pi, q2_pi = self.critic(obs_n, self._norm_actions(joint))
            q_pi = torch.min(q1_pi, q2_pi)

            # Per-agent MASAC actor update with a centralized critic. Agent i
            # differentiates the joint Q only w.r.t. its own (reparameterised)
            # action while the other agents' replay actions stay fixed, so it
            # gets an individual credit signal (its own dQ/da_i) even though
            # the reward is shared. This is the decentralized-actor /
            # centralized-critic gradient of MADDPG / MASAC (Lowe et al. 2017).
            alpha = self.log_alphas[name].exp()
            actor_loss = ((alpha.detach() * logp - q_pi) * mask).sum() / (mask.sum() + 1e-8)

            self.opt_actors[name].zero_grad()
            actor_loss.backward()
            # Clip ALL actor-side parameters the optimizer updates (GRU encoder +
            # soft/hard attention + policy head), not just the policy head. The GRU
            # is the component most prone to exploding gradients under BPTT.
            actor_params = [p for grp in self.opt_actors[name].param_groups for p in grp["params"]]
            torch.nn.utils.clip_grad_norm_(actor_params, self.config.grad_clip)
            self.opt_actors[name].step()

            # Fixed-alpha mode: no auto-tune. log_alpha has requires_grad=False,
            # so there is no alpha gradient step.

            result[f"{name}_loss_actor"] = float(actor_loss.item())
            result[f"{name}_alpha"] = float(alpha.item())

        for p in self.critic.parameters():
            p.requires_grad_(True)
        for p, tp in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            tp.data.copy_(self.config.tau * p.data + (1.0 - self.config.tau) * tp.data)

        return result

    def state_dict(self) -> dict:
        return {
            # schema_version 6: fixed alpha + critic hard attention + per-agent
            # MASAC actor update (centralized critic, decentralized actors).
            "schema_version": 6,
            "config": self.config,
            "total_steps": self.total_steps,
            "actors": {n: self.actors[n].state_dict() for n in self.agent_names},
            "actor_encoders": {n: self.actor_encoders[n].state_dict() for n in self.agent_names},
            "actor_attentions": {n: self.actor_attentions[n].state_dict() for n in self.agent_names},
            "actor_hard_attentions": {
                n: self.actor_hard_attentions[n].state_dict() for n in self.agent_names
            },
            "opt_actors": {n: self.opt_actors[n].state_dict() for n in self.agent_names},
            "log_alphas": {n: self.log_alphas[n].detach().cpu() for n in self.agent_names},
            "critic": self.critic.state_dict(),
            "critic_tgt": self.critic_tgt.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.total_steps = int(state.get("total_steps", 0))
        if (
            "actor_encoders" not in state
            or "actor_attentions" not in state
            or "actor_hard_attentions" not in state
        ):
            raise KeyError(
                "Checkpoint is missing 'actor_encoders', 'actor_attentions', "
                "or 'actor_hard_attentions'; pre-TP-013 checkpoints are not "
                "compatible with the actor-side encoder + soft/hard-attention stack."
            )
        for name in self.agent_names:
            self.actors[name].load_state_dict(state["actors"][name])
            self.actor_encoders[name].load_state_dict(state["actor_encoders"][name])
            self.actor_attentions[name].load_state_dict(state["actor_attentions"][name])
            self.actor_hard_attentions[name].load_state_dict(
                state["actor_hard_attentions"][name]
            )
            self.opt_actors[name].load_state_dict(state["opt_actors"][name])
            self.log_alphas[name].data.copy_(state["log_alphas"][name])
            # Fixed-alpha mode: no alpha optimizer to restore.
        self.critic.load_state_dict(state["critic"])
        self.critic_tgt.load_state_dict(state["critic_tgt"])
        self.opt_critic.load_state_dict(state["opt_critic"])
        self.reset_actor_state()
