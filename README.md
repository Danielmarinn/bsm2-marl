# BSM2 MARL — Multi-Agent Reinforcement Learning for Wastewater Treatment

**Daniel Marin** — MSc Thesis, Engineering Physics, University of Coimbra

> 🚧 **Active Development** — Currently implementing CTRL-2 (SAC). G2ANet multi-agent framework in progress.

---

## Overview

This project applies **Multi-Agent Reinforcement Learning (MARL)** to control a wastewater treatment plant (WWTP) simulated in the **BSM2** (Benchmark Simulation Model No. 2) environment running in MATLAB/Simulink.

The goal is to train 4 cooperative RL agents to simultaneously optimise effluent quality and operational costs, replacing conventional PID controllers.

---

## Architecture

```
MATLAB/Simulink (BSM2)          Python (RL Agents)
─────────────────────           ──────────────────
  BSM2 closed-loop         ←→   SAC agents
  Simulink step (15 min)        Reward computation
  State export → CSV            Action → CSV
  Action import ← CSV           Checkpoint saving
```

Communication between MATLAB and Python is done via CSV file exchange with flag files for synchronisation, running at the BSM2 simulation timestep (15 min).

---

## The 4 Control Agents

| Agent | Variable | Range | Observations |
|-------|----------|-------|--------------|
| CTRL-1 | Qec (external carbon) | — | SNO₂, SNO₁, SNO₃, COD/TN |
| **CTRL-2** ✅ | **Qint (internal recirculation)** | **5,000–61,944 m³/d** | **SNO₂, SNO₁, SNO₃, COD/TN** |
| CTRL-3 | DOref (dissolved oxygen setpoint) | — | SO₅, SNH₄, SNH₅, SNO₃, TSS₃ |
| CTRL-4 | Qw (waste sludge flow) | — | TSS₅, SND₅ |

Observations selected via **Pearson correlation analysis** over 57,000 BSM2 samples (days 245–609).

---

## Reward Function

Based on **Nam et al. (2023)**:

```
J(t) = 200·EQI + 40·AE + 3·PE + 1·EC
r(t) = -5·(J_RL / J_manual) + 5,   clipped ≥ -1
```

Where:
- **EQI** — Effluent Quality Index (pollution load)
- **AE** — Aeration Energy
- **PE** — Pumping Energy (function of Qint)
- **EC** — External Carbon consumption

---

## Algorithm: SAC → G2ANet (planned)

**Current:** Single-agent **Soft Actor-Critic (SAC)** for CTRL-2

**Planned:** **G2ANet** (Graph-to-Agent Network, Liu et al.) — CTDE framework with:
- Hard attention (Gumbel-Softmax) → sparse binary communication graph
- Soft attention → weighted message aggregation between agents
- Centralised critic during training, decentralised execution

---

## Repository Structure

```
├── agents/
│   ├── ctrl_sac_qint.py        # CTRL-2: Qint SAC agent ✅
│   └── ctrl_proportional.py    # Proportional baseline controller
│
├── core/
│   ├── sac_networks.py         # Actor/Critic neural networks
│   ├── replay_buffer.py        # Experience replay buffer
│   └── reward.py               # Shared reward function (Nam et al. 2023)
│
├── matlab/
│   ├── RL_main_episodes.m      # Episode-based training orchestrator
│   ├── RL_main_simple.m        # Simple step-by-step orchestrator
│   ├── update_Qint_from_python.m
│   ├── save_last_sample_to_csv.m
│   └── filter_and_rename_csv.m
│
├── docs/
│   └── Correlação agentes/     # Pearson correlation analysis (MATLAB)
│
├── comms/                      # MATLAB↔Python CSV interface (runtime)
├── checkpoints/                # Saved model weights
└── logs/                       # Training logs (reward, J, episode stats)
```

---

## SAC Hyperparameters (CTRL-2)

| Parameter | Value |
|-----------|-------|
| Hidden layers | (256, 256) |
| Learning rate | 3×10⁻⁴ |
| Discount γ | 0.99 |
| Soft update τ | 0.005 |
| Batch size | 256 |
| Buffer size | 50,000 |
| Warmup steps | 1,000 |

---

## Setup

```bash
pip install torch numpy pandas
```

**Run training (CTRL-2):**
```bash
# Terminal 1 — Start Python agent
python agents/ctrl_sac_qint.py

# MATLAB — Start BSM2 simulation
run matlab/RL_main_episodes.m
```

Requires MATLAB/Simulink with BSM2 closed-loop model.

---

## Roadmap

- [x] MATLAB-Python communication bridge
- [x] Reward function (Nam et al. 2023)
- [x] SAC infrastructure (networks, replay buffer)
- [x] CTRL-2 (Qint) — SAC agent training
- [ ] CTRL-1 (Qec) — SAC agent
- [ ] CTRL-3 (DOref) — SAC agent
- [ ] CTRL-4 (Qw) — SAC agent
- [ ] G2ANet multi-agent communication framework
- [ ] Full 4-agent training and evaluation
- [ ] Comparison vs BSM2 baseline controllers

---

## References

- Nam, K. et al. (2023). *Reinforcement learning-based WWTP control*
- Liu, Y. et al. (2020). *G2ANet: Multi-Agent Communication with Graph Attention*
- Jeppsson, U. et al. (2011). *BSM2: Benchmark Simulation Model No. 2*
