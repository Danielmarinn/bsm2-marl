# bsm2-marl AGENT GUIDE

## Purpose
- This is Daniel Marin's MSc thesis repository.
- Main goal: build, debug, and extend the wastewater-treatment RL pipeline.
- Current priority: move from the current CTRL-2 SAC setup toward a multi-agent setup, without breaking the existing Python-MATLAB bridge.

## Working Style
- Act like an execution partner, not just a reviewer.
- If asked to implement, implement.
- If asked to debug, reproduce the issue as far as the environment allows, identify the cause, patch it, and report what was verified.
- Prefer small, reviewable diffs over broad rewrites.
- Preserve the current project structure unless there is a strong reason to change it.

## Project Facts
- Python side lives mainly in `agents/` and `core/`.
- MATLAB orchestration lives in `matlab/`.
- Communication between MATLAB and Python is file-based CSV/flag exchange.
- `agents/ctrl_sac_qint.py` is the current reference agent.
- The BSM2/Simulink runtime may not be available in cloud tasks.

## Priority Order
1. Keep existing Python training logic correct.
2. Keep MATLAB-Python communication compatibility intact.
3. Make implementation progress on RL agents, debugging, and infrastructure.
4. Improve docs only when they help current work.

## Guardrails
- Do not pretend Simulink or hardware validation happened if it did not.
- If MATLAB/Simulink is unavailable, do Python-side validation and clearly state the limitation.
- Do not delete checkpoints, logs, or docs unless the task explicitly asks for cleanup.
- Do not rename core files or folders casually because MATLAB and Python scripts may depend on fixed paths.
- Avoid changing reward semantics or control ranges unless the task explicitly calls for it.

## Validation
- After Python changes, run the strongest checks available in the environment.
- Minimum expected Python validation:
  - `python -m py_compile agents\\ctrl_proportional.py agents\\ctrl_sac_qint.py core\\replay_buffer.py core\\reward.py core\\sac_networks.py`
- If dependencies are available, prefer adding a short smoke check or targeted script run.
- If a change touches MATLAB integration, inspect related `.m` files for path/CSV/flag compatibility even if MATLAB cannot run.

## Output Expectations
- State:
  - what changed
  - what was verified
  - what could not be verified locally or in cloud
- When blocked by missing MATLAB/Simulink, propose the exact next validation step for Daniel to run locally.

