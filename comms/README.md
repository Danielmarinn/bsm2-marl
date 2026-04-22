# Runtime Communication Files

This folder shows the file-based bridge between MATLAB/Simulink and Python.

## Files

- `state.csv`: latest process state exported by MATLAB.
- `action.csv`: latest controller action written by Python.
- `episode_info.csv`: episode metadata for the episode-based orchestrator.

## Notes

- The canonical action column is `Qint`.
- A legacy `Qec` column is also written for backward compatibility with older bridge code.
- Runtime flag files such as `flag_state.run` and `flag_action.run` are intentionally ignored by Git.
