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

The central finding is negative but informative: the coordinated multi-agent controller runs end to end on BSM2 and produces closed-loop plant-wide policies, but the tested reward, action bounds, and compliance penalty do **not** yield a policy that beats the manual BSM2 baseline while respecting the official effluent limits.

A lower operating cost is only meaningful if the effluent limits stay protected. The manual baseline is a demanding reference: it already runs near 2 mg/L dissolved oxygen and balances effluent quality (EQI), operating cost (OCI), and compliance well.

The table below is the thesis synthesis (Table 5.5), evaluated on the official 245-609 day window.

| Indicator | Manual baseline | Original ranges | Restricted ranges | Lower DO range | Lower DO + penalty |
|---|---:|---:|---:|---:|---:|
| EQI (kg poll. units / d) | 5576.7 | 31911.8 | 5756.2 | 5551.8 | 5714.0 |
| OCI (total, cost units) | 9450.0 | 4137.0 | 10631.1 | 11265.3 | 11550.0 |
| Aeration energy (kWh/d) | 4225.4 | 788.3 | 5032.4 | 3848.6 | 4368.5 |
| External carbon cost | 2400.0 | 3647.5 | 2766.9 | 4494.3 | 4251.4 |
| SNH violation (% of time) | 0.41 | 91.31 | 0.25 | 12.93 | 15.14 |
| TN violation (% of time) | 1.18 | 90.75 | 5.16 | 0.45 | 1.13 |
| SNH95 (mg N/L) | 1.54 | 50.57 | 1.42 | 5.80 | 6.19 |
| TN95 (mg N/L) | 16.75 | 52.82 | 18.03 | 15.47 | 16.40 |

How to read it:

- **Original action ranges** reach a much lower OCI (4137 vs 9450), but only by collapsing aeration to 788 kWh/d. This suppresses nitrification, so EQI rises about 5.7x and ammonia is over the limit for 91% of the window. The apparent cost saving is not a genuine efficiency gain; it is undertreatment.
- **Restricted (physically safe) action ranges** are the most reliable learned configuration. Ammonia compliance is restored (0.25% violation) and EQI returns close to the baseline, but the official OCI is about **12.5% higher** than the manual baseline. Once confined to a safe operating region, the learned controller treats more aggressively and spends more.
- **Lowering the dissolved-oxygen bound** recovers part of the aeration saving but reintroduces ammonia violations (12.93%). The simple compliance penalty tested here does not close that trade-off (15.14%).

So the contribution is the implementation plus the diagnosis: the work shows *when* an apparent cost reduction is caused by insufficient treatment effort rather than by genuine improvement, and it shows that effluent limits must shape learning as safety requirements before coordinated RL can reduce operating cost without degrading effluent quality.

See `docs/validation.md` for the per-configuration tables and `results/` for the exported official summaries.

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
