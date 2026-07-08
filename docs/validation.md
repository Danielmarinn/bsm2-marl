# Validation Summary

The public repository keeps only the small validation artifacts needed to understand the thesis result.

## Single-agent diagnostics

All four diagnostic controllers reached active SAC operation and produced bounded actions.

| Controller | Rows | SAC rows | Action range observed | Bounds OK |
|---|---:|---:|---:|---|
| CTRL-1 / Qec | 58401 | 57401 | 0.0 to 5.0 | yes |
| CTRL-2 / Qint | 58401 | 57401 | 5000.0 to 61944.0 | yes |
| CTRL-3 / DO | 47401 | 46401 | 0.0 to 10.0 | yes |
| CTRL-4 / Qw | 608 | 548 | 0.19 to 448.55 | yes |

See `docs/validation/single_agent_validation_overview.png` and the per-controller figures in the same folder.

## Multi-agent runs

Each coordinated configuration was trained once and evaluated on the official 245-609 day window against the manual BSM2 baseline. The table is the thesis synthesis (Table 5.5). Lower is better for cost and violation metrics, but a lower OCI only counts if the effluent limits stay protected.

| Indicator | Manual | Original ranges | Restricted ranges | Lower DO | Lower DO + penalty |
|---|---:|---:|---:|---:|---:|
| EQI (kg poll./d) | 5576.7 | 31911.8 | 5756.2 | 5551.8 | 5714.0 |
| OCI (total) | 9450.0 | 4137.0 | 10631.1 | 11265.3 | 11550.0 |
| Aeration (kWh/d) | 4225.4 | 788.3 | 5032.4 | 3848.6 | 4368.5 |
| External carbon cost | 2400.0 | 3647.5 | 2766.9 | 4494.3 | 4251.4 |
| SNH violation (% time) | 0.41 | 91.31 | 0.25 | 12.93 | 15.14 |
| TN violation (% time) | 1.18 | 90.75 | 5.16 | 0.45 | 1.13 |
| SNH95 (mg N/L) | 1.54 | 50.57 | 1.42 | 5.80 | 6.19 |
| TN95 (mg N/L) | 16.75 | 52.82 | 18.03 | 15.47 | 16.40 |

Reading:

- Original ranges reach the lowest OCI but through underaeration: EQI rises about 5.7x and ammonia is violated 91% of the time. Not a genuine saving.
- Restricted ranges are the most reliable learned configuration: ammonia compliant (0.25%), EQI near baseline, but OCI about 12.5% above the manual baseline.
- Lowering the dissolved-oxygen bound recovers some aeration saving but breaks ammonia compliance (12.93%); the tested soft penalty does not fix it (15.14%).

No learned configuration improves the manual baseline while respecting the official effluent limits. That is the intended result.

## Files

- `results/bsm2_manual_baseline_official_summary.csv` - manual baseline, official metrics.
- `results/game2_official_summary_20260620_091413.csv` - restricted-ranges run (the most reliable learned configuration).
- `results/game2_official_summary_20260623_080230.csv` - lower dissolved-oxygen run.
- `docs/validation/single_agent_validation_summary.csv` - single-agent diagnostic table.
