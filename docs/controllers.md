# Controllers

The project separates the thesis controller into four single-agent diagnostics and one coordinated multi-agent controller.

## Single-agent diagnostics

Ranges and rate limits follow Table 4.1 of the dissertation.

| Controller | Action | Range | Rate limit | Purpose |
|---|---:|---:|---:|---|
| CTRL-1 | `Qec` (m3/d) | 0 to 5 | ±0.5 | External carbon dosing diagnostic. |
| CTRL-2 | `Qint` (m3/d) | 5000 to 61944 | ±5000 | Internal nitrate recycle diagnostic. |
| CTRL-3 | `SO4ref` / `DOref` (mg/L) | 0 to 10 | ±0.5 | Dissolved oxygen setpoint diagnostic. |
| CTRL-4 | `Qw` (m3/d) | 0 to 450 | ±5 | Waste sludge flow diagnostic. |

The rate limit applies at each variable's update interval: the 15 min simulator step for `Qec`, `Qint` and `DOref`, and the daily sludge update for `Qw`.

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
