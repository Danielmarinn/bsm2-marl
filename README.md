# bsm2-marl

Multi-agent reinforcement learning for energy-aware control of the BSM2 wastewater treatment benchmark.

This repository contains the controller implementation developed for the MSc thesis *Multi-Agent Reinforcement Learning for Energy-Efficient Control of Wastewater Treatment Plants* (University of Coimbra). The work studies four coordinated control actions in the BSM2 plant: external carbon dosing (`Qec`), internal nitrate recycle (`Qint`), dissolved oxygen setpoint (`SO4ref` / `DOref`), and waste sludge flow (`Qw`).

The final controller is a recurrent Soft Actor-Critic system trained under centralized training and decentralized execution. Each actor receives its own local observation. During training, a recurrent centralized critic sees the joint observation-action sequence and uses G2ANet-inspired attention to estimate a shared plant objective.

## What is included

- `agents/` - runnable Python controllers for the single-agent diagnostics and the multi-agent coordinator.
- `core/` - SAC networks, replay buffer, reward functions, Game2 state abstraction, and recurrent G2ANet/CTDE blocks.
- `matlab/` - MATLAB/Simulink orchestration and action-update hooks for the BSM2 co-simulation.
- `scripts/` - analysis, monitoring, validation, and run helpers.
- `docs/` - concise implementation and validation notes.
- `results/` - small exported tables and figures that document the thesis runs.

The official BSM2 distribution is not redistributed here. MATLAB, Simulink, and the BSM2 model files must be available locally to reproduce the full plant loop.

## Controller modes

The multi-agent coordinator supports four modes:

| Mode | Meaning |
|---|---|
| `sac-local` | Four independent SAC actors, each using only its local observation. |
| `sac-game2` | Four actors with shared Game2 plant context appended to their inputs. |
| `sac-ctde` | Decentralized actors with a centralized G2ANet critic. |
| `sac-ctde-rnn` | Recurrent actors and recurrent centralized critic with two-stage actor-side peer attention. |

The thesis focuses on `sac-ctde-rnn`. It uses a straight-through binary hard gate followed by soft attention on the actor side, and soft inter-agent attention in the centralized critic. The entropy temperature is fixed in the final controller rather than auto-tuned.

## Main result, stated carefully

The final multi-agent run shows that the controller learned to reduce several controllable resource terms, but it did not dominate the manual BSM2 baseline on the full objective.

On the official 245-609 day evaluation window, compared with the manual baseline:

- aeration energy decreased by about 24%;
- pumping energy decreased by about 32%;
- external carbon use decreased by about 49%;
- the partial controllable operating-cost components decreased by about 33%;
- the logged reward objective without safety penalties was about 6.9% worse;
- ammonium violations increased substantially.

So the result is not presented as a production-ready controller. It is evidence that the MARL stack, MATLAB/Python bridge, recurrent CTDE training loop, and attention-based coordination can be run end to end on BSM2, while also exposing the remaining water-quality tradeoff.

See `docs/validation.md` and `results/game2_final_audit/` for the exported summaries.

## Running the code

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Run the Python coordinator from the repository root, then start the matching MATLAB script in the BSM2 project:

```bash
python agents/ctrl_game2_sac_coordinator.py --mode sac-ctde-rnn
```

In MATLAB:

```matlab
run matlab/RL_main_game2.m
```

The co-simulation uses CSV and flag files under `comms/`. Runtime logs, checkpoints, `.mat` files, and local BSM2 assets are intentionally ignored by Git.

## Repository stance

This is a research codebase, not a packaged control product. The public version keeps the implementation, reproducibility hooks, small validation artifacts, and honest result summaries. Large training logs, model checkpoints, MATLAB `.mat` runtime files, and the proprietary/third-party BSM2 model are excluded.

## License

MIT. See `LICENSE`.
