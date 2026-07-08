# Exploratory Multi-Agent Audit (not a thesis configuration)

These compact tables come from an additional exploratory recurrent CTDE-SAC run. They are kept for transparency, but this run is **not** one of the four configurations reported in the thesis (see Table 5.5 and `docs/validation.md`).

Like the original wide-range run in the thesis, this run reduces aeration energy and external carbon by operating below the manual dissolved-oxygen level. That apparent saving is an underaeration effect, not a genuine efficiency gain: effluent ammonia (SNH) is over the limit for about 27.6% of the official 245-609 day window, and the "controllable OCI (AE+PE+3EC)" reported in the CSV is a partial cost term, not the official BSM2 OCI.

For the authoritative results and the official OCI, use:

- `results/bsm2_manual_baseline_official_summary.csv` - manual baseline;
- `results/game2_official_summary_20260620_091413.csv` - restricted-ranges run (most reliable learned configuration);
- `results/game2_official_summary_20260623_080230.csv` - lower dissolved-oxygen run.
