# Project Structure

## Repository map

```text
bsm2-marl/
|-- agents/
|   |-- ctrl_proportional.py
|   `-- ctrl_sac_qint.py
|-- core/
|   |-- replay_buffer.py
|   |-- reward.py
|   `-- sac_networks.py
|-- matlab/
|   |-- RL_main_episodes.m
|   |-- RL_main_simple.m
|   |-- update_Qint_from_python.m
|   |-- save_last_sample_to_csv.m
|   `-- filter_and_rename_csv.m
|-- comms/
|   |-- README.md
|   |-- action.csv
|   |-- episode_info.csv
|   `-- state.csv
|-- checkpoints/
|   `-- ctrl2_qint_sac.pt
|-- logs/
|   `-- ctrl2_qint_training.csv
|-- docs/
|   |-- README.md
|   |-- correlation-analysis/
|   |   |-- bsm2_corr_analysis_full.m
|   |   |-- bsm2_corr_analysis_full_prog_rl_snapshot.m
|   |   `-- open-loop-vs-closed-loop-correlation.md
|   `-- matlab-python-integration-notes.txt
|-- AGENTS.md
|-- PROJECT_STRUCTURE.md
|-- README.md
`-- requirements.txt
```

## Folder roles

- `agents/`: Python controllers and learning logic.
- `core/`: shared RL components used by the agents.
- `matlab/`: MATLAB scripts that drive BSM2 and exchange data with Python.
- `comms/`: representative runtime CSV files used by the bridge.
- `checkpoints/`: sample trained model artifact.
- `logs/`: sample training trace.
- `docs/`: analysis notes, correlation work, and integration notes.

## Current implementation status

- Implemented in the public repo:
  - MATLAB-Python communication bridge
  - proportional baseline controller
  - CTRL-2 SAC controller for `Qint`
  - shared reward, replay buffer, and actor-critic networks
  - sample checkpoint and training log artifacts

- Still in active development:
  - CTRL-1 `Qec`
  - CTRL-3 `DOref`
  - CTRL-4 `Qw`
  - multi-agent coordination

## Notes

- This file describes the current curated public repository.
- Runtime flag files such as `flag_state.run` and `flag_action.run` are intentionally not tracked.
