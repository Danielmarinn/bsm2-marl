"""
Shape and gradient check for core/g2anet.py.

This is not a training run. It only verifies that the G2ANet-inspired central
critic accepts the current four BSM2 agent dimensions and backpropagates.
"""

from __future__ import annotations

import os
import sys

import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CORE_DIR = os.path.join(ROOT, "core")
sys.path.insert(0, CORE_DIR)

from g2anet import G2ANetCentralCritic, G2ANetConfig  # noqa: E402


def main() -> int:
    torch.manual_seed(42)
    config = G2ANetConfig(
        obs_dims=(5, 4, 5, 5),
        action_dims=(1, 1, 1, 1),
        embed_dim=32,
        hidden_dim=64,
        temperature=1.0,
    )
    critic = G2ANetCentralCritic(config)

    batch_size = 8
    observations = [torch.randn(batch_size, dim) for dim in config.obs_dims]
    actions = [torch.randn(batch_size, dim) for dim in config.action_dims]

    q_tot, info = critic(observations, actions)
    if q_tot.shape != (batch_size, 1):
        raise AssertionError(f"q_tot shape {tuple(q_tot.shape)} != {(batch_size, 1)}")

    for name in ("hard_gates", "soft_weights"):
        value = info[name]
        expected = (batch_size, config.n_agents, config.n_agents)
        if value.shape != expected:
            raise AssertionError(f"{name} shape {tuple(value.shape)} != {expected}")

    diag = torch.diagonal(info["hard_gates"], dim1=1, dim2=2)
    if not torch.allclose(diag, torch.zeros_like(diag), atol=1e-6):
        raise AssertionError("hard attention self-links should be zero")

    loss = q_tot.mean()
    loss.backward()
    grad_norm = 0.0
    for param in critic.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.norm().item())
    if grad_norm <= 0.0:
        raise AssertionError("expected non-zero gradients")

    print("[G2ANET-CHECK] PASS")
    print(f"[G2ANET-CHECK] q_tot shape: {tuple(q_tot.shape)}")
    print(f"[G2ANET-CHECK] hard_gates shape: {tuple(info['hard_gates'].shape)}")
    print(f"[G2ANET-CHECK] soft_weights shape: {tuple(info['soft_weights'].shape)}")
    print(f"[G2ANET-CHECK] grad_norm: {grad_norm:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
