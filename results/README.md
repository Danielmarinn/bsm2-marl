# Results

This folder keeps the exported summaries from the thesis work.

Included:

- manual BSM2 baseline summary tables (`bsm2_manual_baseline_official_summary.csv`);
- official per-configuration summaries for the learned runs reported in the thesis
  (`game2_official_summary_20260620_091413.csv` = restricted ranges,
  `game2_official_summary_20260623_080230.csv` = lower dissolved oxygen,
  `game2_official_summary_20260626_084942.csv` = lower dissolved oxygen with compliance penalty);
- `game2_original_ranges_from_thesis_tables.csv`, the original-ranges configuration. This file is
  transcribed from the dissertation tables rather than exported from a run, because that run's
  summary was not retained. Each field carries its source and unreported fields are blank;
- validation helpers used to check exported metrics.

Not included:

- full training logs;
- model checkpoints;
- MATLAB `.mat` runtime files;
- the BSM2 model distribution.

The most readable summary is `docs/validation.md`. The raw exported tables are here for transparency.
