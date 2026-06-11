"""
Independent SAC components for the first Game2 multi-agent baseline.

This is the conservative baseline before connecting the G2ANet central critic:
four individual SAC policies share the same MATLAB bridge but do not yet use a
learned attention mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from replay_buffer import ReplayBuffer
from sac_networks import Actor, Critic


AGENT_LOCAL_DIMS = {
    "qec": 6,
    "qint": 6,
    "do": 6,
    "qw": 9,
}

LOCAL_STATE_MEAN = {
    "qec": np.array([3.8, 5.5, 7.1, 2.1, 7.27, 0.0], dtype=np.float32),
    "qint": np.array([3.8, 5.5, 7.1, 20648.0, 0.55, 0.0], dtype=np.float32),
    "do": np.array([1.99551, 1.34704, 0.557381, 5.69173, 4776.8, 0.0], dtype=np.float32),
    "qw": np.array([3762.93, 0.531962, 17.4439, 0.545891, 1671.58, 3260.0, 20648.0, 0.55, 0.0], dtype=np.float32),
}

LOCAL_STATE_STD = {
    "qec": np.array([1.8, 1.9, 1.7, 0.5, 4.0, 1.0], dtype=np.float32),
    "qint": np.array([1.8, 1.9, 1.7, 3000.0, 1.0, 1.0], dtype=np.float32),
    "do": np.array([0.10904, 1.64868, 1.09978, 1.54241, 518.046, 1.0], dtype=np.float32),
    "qw": np.array([254.892, 0.0396945, 2.324, 0.383135, 304.829, 500.0, 3000.0, 1.0, 1.0], dtype=np.float32),
}


@dataclass(frozen=True)
class SACConfig:
    state_dim: int
    action_low: float
    action_high: float
    hidden: tuple[int, int] = (256, 256)
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 50_000
    warmup_steps: int = 1_000
    grad_clip: float = 1.0


def build_normalisation(agent: str, state_dim: int) -> tuple[np.ndarray, np.ndarray]:
    mean = LOCAL_STATE_MEAN[agent]
    std = LOCAL_STATE_STD[agent]
    if state_dim == len(mean):
        return mean, std
    if state_dim < len(mean):
        raise ValueError(f"{agent} state_dim {state_dim} is smaller than local dim {len(mean)}")
    extra = state_dim - len(mean)
    return (
        np.concatenate([mean, np.zeros(extra, dtype=np.float32)]).astype(np.float32),
        np.concatenate([std, np.ones(extra, dtype=np.float32)]).astype(np.float32),
    )


class IndependentSACAgent:
    def __init__(self, name: str, config: SACConfig):
        self.name = name
        self.config = config
        self.actor = Actor(config.state_dim, 1, config.hidden, config.action_low, config.action_high)
        self.critic = Critic(config.state_dim, 1, config.hidden)
        self.critic_tgt = Critic(config.state_dim, 1, config.hidden)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        self.critic_tgt.requires_grad_(False)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)
        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=config.lr_alpha)
        self.buffer = ReplayBuffer(config.state_dim, 1, config.buffer_size)
        self.total_steps = 0
        self.target_entropy = -1.0

        mean, std = build_normalisation(name, config.state_dim)
        self.state_mean = torch.as_tensor(mean, dtype=torch.float32)
        self.state_std = torch.as_tensor(std + 1e-8, dtype=torch.float32)

    def state_dict(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "config": self.config,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_tgt": self.critic_tgt.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "opt_alpha": self.opt_alpha.state_dict(),
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_tgt.load_state_dict(state["critic_tgt"])
        self.opt_actor.load_state_dict(state["opt_actor"])
        self.opt_critic.load_state_dict(state["opt_critic"])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.log_alpha.device))
        self.opt_alpha.load_state_dict(state["opt_alpha"])
        self.total_steps = int(state.get("total_steps", 0))

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    def normalize(self, state: np.ndarray) -> torch.Tensor:
        state_t = torch.as_tensor(state, dtype=torch.float32)
        return (state_t - self.state_mean) / self.state_std

    def random_action(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.config.action_low, self.config.action_high))

    def select_action(
        self,
        state: np.ndarray,
        rng: np.random.Generator,
        deterministic: bool = False,
    ) -> float:
        if self.total_steps < self.config.warmup_steps and not deterministic:
            return self.random_action(rng)
        with torch.no_grad():
            s = self.normalize(state).unsqueeze(0)
            if deterministic:
                action = self.actor.deterministic(s)
            else:
                action, _ = self.actor.sample(s)
        return float(action.squeeze().item())

    def add_transition(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.add(
            np.asarray(state, dtype=np.float32),
            np.asarray([action], dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        )
        self.total_steps += 1

    def train_step(self) -> Mapping[str, float] | None:
        if len(self.buffer) < self.config.batch_size:
            return None

        s, a, r, s_next, d = self.buffer.sample(self.config.batch_size)
        s = (torch.as_tensor(s, dtype=torch.float32) - self.state_mean) / self.state_std
        s_next = (torch.as_tensor(s_next, dtype=torch.float32) - self.state_mean) / self.state_std
        a = torch.as_tensor(a, dtype=torch.float32)
        r = torch.as_tensor(r, dtype=torch.float32)
        d = torch.as_tensor(d, dtype=torch.float32)

        with torch.no_grad():
            a_next, logp_next = self.actor.sample(s_next)
            q1_next, q2_next = self.critic_tgt(s_next, a_next)
            q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp() * logp_next
            target = r + self.config.gamma * (1.0 - d) * q_next

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.opt_critic.step()

        a_pi, logp = self.actor.sample(s)
        q1_pi, q2_pi = self.critic(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.log_alpha.exp().detach() * logp - q_pi).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.opt_actor.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data
            )

        return {
            "loss_actor": float(actor_loss.item()),
            "loss_critic": float(critic_loss.item()),
            "loss_alpha": float(alpha_loss.item()),
            "alpha": self.alpha,
        }
