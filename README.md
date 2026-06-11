# bsm2-marl

Multi-agent reinforcement learning for energy-efficient control of the BSM2 wastewater treatment benchmark.

This is the controller implementation from the MSc thesis *Multi-Agent Reinforcement Learning for Energy-Efficient Control of Wastewater Treatment Plants* (University of Coimbra). Four agents act on the dissolved-oxygen setpoint, the internal nitrate recycle, the external carbon dosage, and the waste sludge flow of the BSM2 plant. They are trained with a recurrent Soft Actor-Critic under centralized training and decentralized execution, and coordinate through the two-stage graph attention of G2ANet. The design recreates the G2ANet controller of Nam et al. (2023) and adapts it from a full-scale digital twin to the plant-wide BSM2 benchmark.

## Layout

- `core/` — shared building blocks: actor and critic networks, replay buffer, reward, G2ANet attention, and the CTDE-SAC variants.
- `agents/` — runnable controllers: the four single-agent diagnostic baselines and the multi-agent coordinator.
- `matlab/` — the Simulink/BSM2 side of the co-simulation and the per-step action hooks.
- `scripts/` — launch scripts and post-run analysis.

## Controller modes

The coordinator supports four modes of increasing capability:

- `sac-local` — independent SAC per actuator, no communication.
- `sac-game2` — single-step G2ANet attention between agents.
- `sac-ctde` — centralized critic with decentralized actors.
- `sac-ctde-rnn` — recurrent actors and critic with the full two-stage G2ANet attention. This is the controller evaluated in the thesis.

## How it runs

Training and evaluation are a file-based co-simulation. The MATLAB BSM2 model exports the plant state at every 15-minute control step, the Python coordinator returns the four setpoints, and the two sides synchronize through a CSV and flag-file handshake. The MATLAB side requires the official BSM2 distribution from the IWA Task Group on Benchmarking of Control Strategies, which is not redistributed here.

A full run is started from `scripts/` (for example `run_game2_clean609.ps1` on the Python side) together with `matlab/RL_main_game2.m` in the BSM2 Simulink project.

## Requirements

Python 3.11 with PyTorch, NumPy and Pandas (`requirements.txt`). MATLAB with Simulink and the BSM2 benchmark.

## License

MIT. See `LICENSE`.
