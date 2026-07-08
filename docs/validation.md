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

## Multi-agent run

The final run was evaluated on the official 245-609 day window against the manual BSM2 baseline.

| Metric | Run | Manual | Change |
|---|---:|---:|---:|
| Aeration energy AE | 3209 kWh/d | 4225 kWh/d | 24.0% lower |
| Pumping energy PE | 304 kWh/d | 445 kWh/d | 31.8% lower |
| External carbon EC | 404 kg COD/d | 800 kg COD/d | 49.5% lower |
| Partial controllable operating cost | 4726 | 7071 | 33.2% lower |
| EQI | 6226 | 5577 | 11.6% higher |
| Reward objective J without safety penalties | 1.375e6 | 1.286e6 | 6.9% higher |
| SNH violation time | 27.6% | 0.41% | worse |

Lower is better for the cost and violation metrics. The controller clearly reduced several resource terms, but the higher EQI and ammonium violation time mean it should be read as a research result rather than a finished control policy.

## Files

- `docs/validation/single_agent_validation_summary.csv` - single-agent diagnostic table.
- `results/game2_final_audit/official_245_609_comparison.csv` - final run vs manual baseline.
- `results/game2_final_audit/windows_summary.csv` - final run by time window.
- `results/bsm2_manual_baseline_official_summary.csv` - manual baseline official-style metrics.
