"""
CTDE multi-agent SAC with G2ANet central critic for BSM2.

Actors receive only local observations — identical input space to sac-local.
The twin G2ANet central critic receives ALL agents' local observations and
actions during training, so inter-agent coordination is learned by the critic
via attention without inflating actor input dimensions.

This is the centralized-training decentralized-execution (CTDE) approach
used in Nam et al. (2023): agents act on local information; the shared
critic provides the training signal that accounts for joint behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game2_abstraction import ACTION_DELTA_BOUNDS, AGENT_ORDER
from game2_sac import AGENT_LOCAL_DIMS, build_normalisation
from g2anet import G2ANetCentralCritic, G2ANetConfig
from sac_networks import Actor


AGENT_NAMES: list[str] = list(AGENT_ORDER)


@dataclass(frozen=True)
class CTDESACConfig:
    # -- Networks (G2ANet architecture) --------------------------------------
    hidden: tuple[int, int] = (256, 256)   # actors (MLP)
    embed_dim: int = 64                    # G2ANet default
    hidden_dim: int = 128                  # G2ANet default

    # -- Learning rates (Asymmetric for CTDE stability) ----------------------
    lr_actor: float = 1e-4                 # actor learns slower than critic
    lr_critic: float = 3e-4                # critic leads the learning
    lr_alpha: float = 3e-4                # entropy adjustment
    
    # -- RL parameters -------------------------------------------------------
    gamma: float = 0.99                   # long-horizon BSM2 dynamics
    tau: float = 0.01                     # faster target propagation for noisy rewards

    # -- Buffer / batching (Scaled for BSM2 thermal delays) ------------------
    batch_size: int = 512                 # reduced gradient noise
    buffer_size: int = 200_000            # covers ~150 episodes of BSM2 dynamics
    warmup_steps: int = 10_000           # thorough exploration before training

    # -- Stability -----------------------------------------------------------
    grad_clip: float = 0.5               # tighter clip for G2ANet backprop
    updates_per_step: int = 2            # standard SAC efficiency


# ---------------------------------------------------------------------------
# Joint replay buffer
# ---------------------------------------------------------------------------

class JointReplayBuffer:
    """
    Stores one entry per environment step containing all agents' observations,
    actions, the scalar joint reward, and next observations.
    """

    def __init__(self, obs_dims: list[int], n_agents: int, buffer_size: int):
        self.obs_dims = obs_dims
        self.n_agents = n_agents
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
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
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> tuple[
        list[np.ndarray], list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray
    ]:
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = [self._obs[i][idx] for i in range(self.n_agents)]
        next_obs = [self._next_obs[i][idx] for i in range(self.n_agents)]
        actions = [self._actions[i][idx] for i in range(self.n_agents)]
        return obs, actions, self._reward[idx], next_obs, self._done[idx]

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# Twin G2ANet critic
# ---------------------------------------------------------------------------

class TwinG2ANetCritic(nn.Module):
    """Two independent G2ANet critics for clipped double-Q estimation."""

    def __init__(self, config: G2ANetConfig):
        super().__init__()
        self.q1 = G2ANetCentralCritic(config)
        self.q2 = G2ANetCentralCritic(config)

    def forward(
        self,
        observations: Sequence[torch.Tensor],
        actions: Sequence[torch.Tensor],
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1, _ = self.q1(observations, actions, hard=hard)
        q2, _ = self.q2(observations, actions, hard=hard)
        return q1, q2


# ---------------------------------------------------------------------------
# CTDE multi-agent SAC
# ---------------------------------------------------------------------------

class CTDEMultiAgentSAC:
    """
    Multi-agent SAC: decentralized actors, centralized twin G2ANet critic.

    Actor update for agent i: hold all other agents' sampled actions fixed
    (from the replay buffer), replace action i with a fresh actor sample,
    and differentiate Q_tot w.r.t. actor i's parameters only.

    Critic update: standard SAC Bellman target using the joint next-actions
    sampled from all actors, with per-agent entropy terms.
    """

    def __init__(self, config: CTDESACConfig):
        self.config = config
        self.agent_names = AGENT_NAMES

        self.actors: dict[str, Actor] = {}
        self.opt_actors: dict[str, torch.optim.Adam] = {}
        self.log_alphas: dict[str, torch.Tensor] = {}
        self.opt_alphas: dict[str, torch.optim.Adam] = {}
        self.state_means: dict[str, torch.Tensor] = {}
        self.state_stds: dict[str, torch.Tensor] = {}

        for name in self.agent_names:
            obs_dim = AGENT_LOCAL_DIMS[name]
            low, high = ACTION_DELTA_BOUNDS[name]
            self.actors[name] = Actor(obs_dim, 1, config.hidden, low, high)
            self.opt_actors[name] = torch.optim.Adam(
                self.actors[name].parameters(), lr=config.lr_actor
            )
            log_alpha = torch.tensor(0.0, requires_grad=True)
            self.log_alphas[name] = log_alpha
            self.opt_alphas[name] = torch.optim.Adam([log_alpha], lr=config.lr_alpha)
            mean, std = build_normalisation(name, obs_dim)
            self.state_means[name] = torch.as_tensor(mean, dtype=torch.float32)
            self.state_stds[name] = torch.as_tensor(std + 1e-8, dtype=torch.float32)

        g2a_cfg = G2ANetConfig(
            obs_dims=tuple(AGENT_LOCAL_DIMS[n] for n in self.agent_names),
            action_dims=tuple(1 for _ in self.agent_names),
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
        )
        self.critic = TwinG2ANetCritic(g2a_cfg)
        self.critic_tgt = TwinG2ANetCritic(g2a_cfg)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        self.critic_tgt.requires_grad_(False)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        # -- Action Normalization for G2ANet Attention (Nam et al. 2023) -----
        # Prevents scale dominance (e.g. Qint=61944 vs Qec=5) in the encoder.
        self._act_low = torch.tensor([ACTION_DELTA_BOUNDS[n][0] for n in self.agent_names], dtype=torch.float32)
        self._act_high = torch.tensor([ACTION_DELTA_BOUNDS[n][1] for n in self.agent_names], dtype=torch.float32)

        self.buffer = JointReplayBuffer(
            obs_dims=[AGENT_LOCAL_DIMS[n] for n in self.agent_names],
            n_agents=len(self.agent_names),
            buffer_size=config.buffer_size,
        )
        self.total_steps = 0
        self.target_entropies = {name: -1.0 for name in self.agent_names}

    # ------------------------------------------------------------------

    def _norm(self, name: str, obs: np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        return (t - self.state_means[name]) / self.state_stds[name]

    def _norm_batch(self, i: int, obs: np.ndarray) -> torch.Tensor:
        name = self.agent_names[i]
        t = torch.as_tensor(obs, dtype=torch.float32)
        return (t - self.state_means[name]) / self.state_stds[name]

    # ------------------------------------------------------------------
    # Execution: each actor sees only its own local observation
    # ------------------------------------------------------------------

    def select_actions(
        self,
        obs_dict: Mapping[str, np.ndarray],
        rng: np.random.Generator,
        deterministic: bool = False,
    ) -> dict[str, float]:
        if self.total_steps < self.config.warmup_steps and not deterministic:
            return {
                name: float(rng.uniform(*ACTION_DELTA_BOUNDS[name]))
                for name in self.agent_names
            }
        actions: dict[str, float] = {}
        with torch.no_grad():
            for name in self.agent_names:
                s = self._norm(name, obs_dict[name]).unsqueeze(0)
                if deterministic:
                    a = self.actors[name].deterministic(s)
                else:
                    a, _ = self.actors[name].sample(s)
                actions[name] = float(a.squeeze().item())
        return actions

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

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

    def _norm_actions(self, acts: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Maps each agent's action from its physical range to [-1, 1]."""
        return [
            2.0 * (acts[i] - self._act_low[i]) / (self._act_high[i] - self._act_low[i] + 1e-8) - 1.0
            for i in range(len(self.agent_names))
        ]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> Mapping[str, float] | None:
        if len(self.buffer) < self.config.batch_size:
            return None

        obs_b, act_b, rew_b, nobs_b, done_b = self.buffer.sample(self.config.batch_size)

        obs_n = [self._norm_batch(i, obs_b[i]) for i in range(len(self.agent_names))]
        nobs_n = [self._norm_batch(i, nobs_b[i]) for i in range(len(self.agent_names))]
        acts_t = [torch.as_tensor(act_b[i], dtype=torch.float32) for i in range(len(self.agent_names))]
        rew_t = torch.as_tensor(rew_b, dtype=torch.float32)
        done_t = torch.as_tensor(done_b, dtype=torch.float32)

        # ---- Critic update ----
        with torch.no_grad():
            next_acts, next_logps = [], []
            for i, name in enumerate(self.agent_names):
                a_n, lp_n = self.actors[name].sample(nobs_n[i])
                next_acts.append(a_n)
                next_logps.append(lp_n)

            entropy_term = sum(
                self.log_alphas[name].exp().detach() * next_logps[i]
                for i, name in enumerate(self.agent_names)
            )
            # Use normalized actions for G2ANet attention stability
            q1_t, q2_t = self.critic_tgt(nobs_n, self._norm_actions(next_acts))
            q_tgt = torch.min(q1_t, q2_t) - entropy_term
            target = rew_t + self.config.gamma * (1.0 - done_t) * q_tgt

        self.opt_critic.zero_grad()
        # Use normalized actions for G2ANet attention stability
        q1, q2 = self.critic(obs_n, self._norm_actions(acts_t))
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.opt_critic.step()

        # ---- Per-agent actor + alpha updates ----
        result: dict[str, float] = {"critic_loss": float(critic_loss.item())}

        for i, name in enumerate(self.agent_names):
            a_pi, logp = self.actors[name].sample(obs_n[i])

            # AGENT COOPERATION: Sample fresh actions from all other actors
            # to ensure agent 'i' optimizes against current policies, not stale buffer data.
            joint = []
            for j, jname in enumerate(self.agent_names):
                if j == i:
                    joint.append(a_pi)
                else:
                    with torch.no_grad():
                        # We use fresh samples from other actors (no gradient)
                        a_j, _ = self.actors[jname].sample(obs_n[j])
                    joint.append(a_j)

            # Use normalized actions for G2ANet attention stability
            q1_pi, q2_pi = self.critic(obs_n, self._norm_actions(joint))
            q_pi = torch.min(q1_pi, q2_pi)

            alpha = self.log_alphas[name].exp()
            actor_loss = (alpha.detach() * logp - q_pi).mean()
            self.opt_actors[name].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[name].parameters(), self.config.grad_clip)
            self.opt_actors[name].step()

            alpha_loss = -(self.log_alphas[name] * (logp + self.target_entropies[name]).detach()).mean()
            self.opt_alphas[name].zero_grad()
            alpha_loss.backward()
            self.opt_alphas[name].step()

            result[f"{name}_loss_actor"] = float(actor_loss.item())
            result[f"{name}_alpha"] = float(alpha.item())

        # ---- Soft target update ----
        for p, tp in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            tp.data.copy_(self.config.tau * p.data + (1.0 - self.config.tau) * tp.data)

        return result

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "schema_version": 1,
            "config": self.config,
            "total_steps": self.total_steps,
            "actors": {n: self.actors[n].state_dict() for n in self.agent_names},
            "opt_actors": {n: self.opt_actors[n].state_dict() for n in self.agent_names},
            "log_alphas": {n: self.log_alphas[n].detach().cpu() for n in self.agent_names},
            "opt_alphas": {n: self.opt_alphas[n].state_dict() for n in self.agent_names},
            "critic": self.critic.state_dict(),
            "critic_tgt": self.critic_tgt.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.total_steps = int(state.get("total_steps", 0))
        for name in self.agent_names:
            self.actors[name].load_state_dict(state["actors"][name])
            self.opt_actors[name].load_state_dict(state["opt_actors"][name])
            self.log_alphas[name].data.copy_(state["log_alphas"][name])
            self.opt_alphas[name].load_state_dict(state["opt_alphas"][name])
        self.critic.load_state_dict(state["critic"])
        self.critic_tgt.load_state_dict(state["critic_tgt"])
        self.opt_critic.load_state_dict(state["opt_critic"])
