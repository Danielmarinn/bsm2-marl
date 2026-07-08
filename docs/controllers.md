# Controllers

The project separates the thesis controller into four single-agent diagnostics and one coordinated multi-agent controller.

## Single-agent diagnostics

| Controller | Action | Range | Purpose |
|---|---:|---:|---|
| CTRL-1 | `Qec` | 0 to 5 mg/L | External carbon dosing diagnostic. |
| CTRL-2 | `Qint` | 5000 to 61944 m3/d | Internal nitrate recycle diagnostic. |
| CTRL-3 | `SO4ref` / `DOref` | 0 to 10 mg/L | Dissolved oxygen setpoint diagnostic. |
| CTRL-4 | `Qw` | 0 to 450 m3/d | Waste sludge flow diagnostic. |

These controllers are useful because they isolate each actuator before the full multi-agent run. The validation figures in `docs/validation/` show that all four controllers produced bounded actions and usable SAC logs.

## Multi-agent controller

The multi-agent coordinator writes all four actions in one row: `Qec`, `Qint`, `SO4ref`, and `Qw`.

The final thesis path is `sac-ctde-rnn`:

- decentralized actors receive local observations;
- actor-side recurrence carries process history;
- actor-side peer attention uses a straight-through binary gate followed by soft attention;
- the centralized critic receives all agents' observation-action sequences during training;
- each actor update differentiates through its own sampled action while holding the other agents at replay values.

This is a MASAC-style CTDE design adapted to the BSM2 co-simulation loop.

## Files

- `agents/ctrl_game2_sac_coordinator.py` - main Python coordinator.
- `core/game2_recurrent_ctde_sac.py` - recurrent CTDE-SAC training loop.
- `core/recurrent_g2anet.py` - recurrent attention modules.
- `core/reward.py` - single-agent diagnostic rewards and the final shared joint reward.
- `matlab/RL_main_game2.m` - MATLAB orchestration for multi-agent BSM2 runs.
- `matlab/update_game2_actions_from_python.m` - four-action MATLAB bridge.
