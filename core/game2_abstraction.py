"""
Game2/G2ANet shared-observation helpers.

This module defines the first concrete contract for the Game2 abstraction:
each agent can keep its own local state, then append a shared context vector
containing recent actions and plant outputs from the coupled BSM2 system.

The helpers are intentionally independent from SAC so they can be used by
new Game2 agents, analysis scripts, or a future MATLAB/Python multi-agent
runner without changing the single-agent controllers first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


FEATURE_SCHEMA_VERSION = 1

AGENT_ORDER = ("qec", "qint", "do", "qw")

ACTION_BOUNDS = {
    "qec": (0.0, 5.0),
    "qint": (5_000.0, 61_944.0),
    "do": (0.0, 10.0),
    "qw": (0.0, 450.0),
}

ACTION_DEFAULTS = {
    "qec": 2.0,
    "qint": 61_944.0,
    "do": 2.0,
    "qw": 300.0,
}

# Maximum allowed change per 15-minute step (Hard Constraints)
# Values chosen to allow sufficient reaction to rain while preventing
# biological shocks or 'bang-bang' control oscillations.
ACTION_RATE_LIMITS = {
    "qec": 0.5,      # m3/d per 15min
    "qint": 5_000.0, # m3/d per 15min
    "do": 0.5,       # mg/L per 15min
    "qw": 5.0,       # m3/d per 15min (Critical for SRT stability)
}

# SAC policies use rate-limited delta actions. The MATLAB bridge still
# receives absolute physical commands; the coordinator converts deltas through
# `apply_action_deltas` before writing action.csv.
ACTION_DELTA_BOUNDS = {
    agent: (-limit, limit)
    for agent, limit in ACTION_RATE_LIMITS.items()
}

# Shared process signals. These are deliberately compact: enough to expose
# cross-agent coupling without burying each policy in every BSM2 state.
OUTPUT_KEYS = (
    "SNO_1",
    "SNO_2",
    "SNO_3",
    "SNH_2",
    "SNH_4",
    "SNH_5",
    "SO_4",
    "SO_5",
    "TSS_3",
    "TSS_5",
    "Flow",
    "CODTN",
)

# Baseline-like centres/scales. These are not final thesis metrics; they are
# stable normalisation constants for a first Game2 implementation.
OUTPUT_MEAN = {
    "SNO_1": 5.54,
    "SNO_2": 3.93,
    "SNO_3": 7.20,
    "SNH_2": 7.27,
    "SNH_4": 1.35,
    "SNH_5": 0.55,
    "SO_4": 2.00,
    "SO_5": 1.57,
    "TSS_3": 4_700.0,
    "TSS_5": 3_260.0,
    "Flow": 20_648.0,
    "CODTN": 2.05,
}

OUTPUT_SCALE = {
    "SNO_1": 2.0,
    "SNO_2": 2.0,
    "SNO_3": 2.0,
    "SNH_2": 4.0,
    "SNH_4": 2.0,
    "SNH_5": 1.0,
    "SO_4": 1.0,
    "SO_5": 1.0,
    "TSS_3": 700.0,
    "TSS_5": 500.0,
    "Flow": 3_000.0,
    "CODTN": 0.5,
}


@dataclass(frozen=True)
class Game2Snapshot:
    """One compact view of the multi-agent plant/controller state."""

    actions: Mapping[str, float] = field(default_factory=dict)
    outputs: Mapping[str, float] = field(default_factory=dict)


def _as_float(value: object, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def normalise_action(agent: str, value: object | None = None) -> float:
    """Map an agent action to approximately [-1, 1]."""

    if agent not in ACTION_BOUNDS:
        raise KeyError(f"Unknown Game2 agent: {agent!r}")
    low, high = ACTION_BOUNDS[agent]
    default = ACTION_DEFAULTS[agent]
    raw = np.clip(_as_float(value, default), low, high)
    mid = (high + low) / 2.0
    scale = (high - low) / 2.0
    return float((raw - mid) / (scale + 1e-8))


def clip_action_deltas(deltas: Mapping[str, float]) -> dict[str, float]:
    """Clip policy outputs to the per-step physical rate limits."""

    clipped: dict[str, float] = {}
    for agent, value in deltas.items():
        if agent not in ACTION_DELTA_BOUNDS:
            raise KeyError(f"Unknown Game2 agent: {agent!r}")
        low, high = ACTION_DELTA_BOUNDS[agent]
        clipped[agent] = float(np.clip(_as_float(value, 0.0), low, high))
    return clipped


def apply_action_deltas(
    deltas: Mapping[str, float],
    previous_actions: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """
    Convert rate-limited policy deltas to absolute actuator commands.

    The previous absolute action is part of the local state, so a delta-action
    policy remains Markov while matching the actuator rate constraints.
    """

    previous = ACTION_DEFAULTS if previous_actions is None else previous_actions
    clipped_deltas = clip_action_deltas(deltas)
    actions: dict[str, float] = {}
    for agent, delta in clipped_deltas.items():
        if agent not in ACTION_BOUNDS:
            raise KeyError(f"Unknown Game2 agent: {agent!r}")
        low, high = ACTION_BOUNDS[agent]
        prev = _as_float(previous.get(agent), ACTION_DEFAULTS[agent])
        actions[agent] = float(np.clip(prev + delta, low, high))
    return actions


def actions_to_deltas(
    actions: Mapping[str, float],
    previous_actions: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Convert absolute actions to clipped policy deltas for replay/debugging."""

    previous = ACTION_DEFAULTS if previous_actions is None else previous_actions
    return clip_action_deltas({
        agent: _as_float(actions.get(agent), ACTION_DEFAULTS[agent])
        - _as_float(previous.get(agent), ACTION_DEFAULTS[agent])
        for agent in AGENT_ORDER
    })


def normalise_output(key: str, value: object | None = None) -> float:
    """Normalise one shared BSM2 process signal."""

    if key not in OUTPUT_KEYS:
        raise KeyError(f"Unknown Game2 output key: {key!r}")
    mean = OUTPUT_MEAN[key]
    scale = OUTPUT_SCALE[key]
    raw = _as_float(value, mean)
    return float((raw - mean) / (scale + 1e-8))


def build_game2_context(
    current: Game2Snapshot,
    previous: Game2Snapshot | None = None,
) -> np.ndarray:
    """
    Build the shared Game2 context vector.

    Feature order:
      1. current normalised actions for qec, qint, do, qw
      2. current normalised shared process outputs
      3. normalised output changes since the previous snapshot

    If no previous snapshot is provided, the delta block is zero.
    """

    features: list[float] = []

    for agent in AGENT_ORDER:
        features.append(normalise_action(agent, current.actions.get(agent)))

    for key in OUTPUT_KEYS:
        features.append(normalise_output(key, current.outputs.get(key)))

    for key in OUTPUT_KEYS:
        if previous is None:
            features.append(0.0)
        else:
            current_value = _as_float(current.outputs.get(key), OUTPUT_MEAN[key])
            previous_value = _as_float(previous.outputs.get(key), OUTPUT_MEAN[key])
            features.append(float((current_value - previous_value) / (OUTPUT_SCALE[key] + 1e-8)))

    return np.asarray(features, dtype=np.float32)


def augment_local_state(
    local_state: np.ndarray,
    current: Game2Snapshot,
    previous: Game2Snapshot | None = None,
) -> np.ndarray:
    """Append the shared Game2 context to an agent's existing local state."""

    local = np.asarray(local_state, dtype=np.float32).reshape(-1)
    context = build_game2_context(current, previous)
    return np.concatenate([local, context]).astype(np.float32)


def game2_context_dim() -> int:
    return len(AGENT_ORDER) + 2 * len(OUTPUT_KEYS)


def game2_feature_names() -> list[str]:
    names = [f"action_{agent}_norm" for agent in AGENT_ORDER]
    names.extend(f"output_{key}_norm" for key in OUTPUT_KEYS)
    names.extend(f"delta_{key}_norm" for key in OUTPUT_KEYS)
    return names


def snapshot_from_mapping(row: Mapping[str, object]) -> Game2Snapshot:
    """
    Build a snapshot from a CSV-like row.

    Accepted action aliases match the current codebase names.
    """

    actions = {
        "qec": row.get("Qec", row.get("Qec_new", ACTION_DEFAULTS["qec"])),
        "qint": row.get("Qint", row.get("Qint_new", ACTION_DEFAULTS["qint"])),
        "do": row.get("SO4ref", row.get("SO4ref_new", ACTION_DEFAULTS["do"])),
        "qw": row.get("Qw", row.get("Qw_new", ACTION_DEFAULTS["qw"])),
    }
    outputs = {key: row.get(key, OUTPUT_MEAN[key]) for key in OUTPUT_KEYS}
    return Game2Snapshot(actions=actions, outputs=outputs)
