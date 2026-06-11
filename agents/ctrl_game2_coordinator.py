"""
Game2 four-agent integration coordinator.

This script is the first Python-side bridge for the Game2 implementation. It
does not make a performance claim and does not load old single-agent
checkpoints, because Game2 augments each local state with 28 shared features.

Typical short integration run:
    python agents/ctrl_game2_coordinator.py --mode hold --max-steps 20
    % in MATLAB, run RL_main_game2

Start the coordinator before the MATLAB runner. By default it archives the
previous integration log and clears stale handshaking flags so a fresh test
does not accidentally consume an old state row.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

import numpy as np
import pandas as pd

_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_AGENTS_DIR, ".."))
_CORE_DIR = os.path.join(_ROOT, "core")
sys.path.insert(0, _CORE_DIR)

from game2_abstraction import (  # noqa: E402
    ACTION_BOUNDS,
    ACTION_DEFAULTS,
    ACTION_RATE_LIMITS,
    AGENT_ORDER,
    augment_local_state,
    game2_context_dim,
    snapshot_from_mapping,
)

COMMS_DIR = os.path.join(_ROOT, "comms")
LOG_DIR = os.path.join(_ROOT, "logs")

STATE_FILE = os.path.join(COMMS_DIR, "state.csv")
ACTION_FILE = os.path.join(COMMS_DIR, "action.csv")
FLAG_STATE = os.path.join(COMMS_DIR, "flag_state.run")
FLAG_ACTION = os.path.join(COMMS_DIR, "flag_action.run")
FLAG_ERROR = os.path.join(COMMS_DIR, "flag_error.run")
RUN_LOCK_FILE = os.path.join(COMMS_DIR, "game2_python_coordinator.lock")

LOG_FILE = os.path.join(LOG_DIR, "game2_integration_log.csv")


@dataclass
class LocalStates:
    qec: np.ndarray
    qint: np.ndarray
    do: np.ndarray
    qw: np.ndarray


def _float(row: Mapping[str, object], key: str, default: float) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        value = default
    if not np.isfinite(value):
        return float(default)
    return float(value)


def read_state_row() -> dict[str, object]:
    while True:
        try:
            df = pd.read_csv(STATE_FILE)
            return df.iloc[-1].to_dict()
        except Exception as exc:
            print(f"[GAME2] error reading state.csv: {exc}")
            time.sleep(0.05)


def build_local_states(
    row: Mapping[str, object],
    previous_actions: Mapping[str, float] | None = None,
) -> LocalStates:
    prev_qec = 0.0 if previous_actions is None else (float(previous_actions["qec"]) - 2.5) / 2.5
    prev_qint = 0.0 if previous_actions is None else (float(previous_actions["qint"]) - 33472.0) / 28472.0
    prev_do = 0.0 if previous_actions is None else (float(previous_actions["do"]) - 5.0) / 5.0
    prev_qw = 0.0 if previous_actions is None else (float(previous_actions["qw"]) - 225.0) / 225.0

    qec = np.array(
        [
            _float(row, "SNO_2", 3.8),
            _float(row, "SNO_1", 5.5),
            _float(row, "SNO_3", 7.1),
            _float(row, "CODTN", 2.1),
            _float(row, "SNH_2", 7.27),
            prev_qec,
        ],
        dtype=np.float32,
    )
    qint = np.array(
        [
            _float(row, "SNO_2", 3.8),
            _float(row, "SNO_1", 5.5),
            _float(row, "SNO_3", 7.1),
            _float(row, "Flow", 20648.0),
            _float(row, "SNH_5", 0.55),
            prev_qint,
        ],
        dtype=np.float32,
    )
    do = np.array(
        [
            _float(row, "SO_4", 2.0),
            _float(row, "SNH_4", 1.35),
            _float(row, "SNH_5", 0.55),
            _float(row, "SNO_3", 7.1),
            _float(row, "TSS_3", 4700.0),
            prev_do,
        ],
        dtype=np.float32,
    )
    qw = np.array(
        [
            _float(row, "TSS_5_1d", _float(row, "TSS_5", 3260.0)),
            _float(row, "SND_5_1d", 0.53),
            _float(row, "SRT_3d", 17.44),
            _float(row, "SNH_5_1d", _float(row, "SNH_5", 0.55)),
            _float(row, "EQI_1d", 1671.58),
            _float(row, "TSS_5", 3260.0),
            _float(row, "Flow", 20648.0),
            _float(row, "SNH_5", 0.55),
            prev_qw,
        ],
        dtype=np.float32,
    )
    return LocalStates(qec=qec, qint=qint, do=do, qw=qw)


def build_augmented_states(
    row: Mapping[str, object],
    previous_row: Mapping[str, object] | None,
    previous_actions: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    current = snapshot_from_mapping(row)
    previous = snapshot_from_mapping(previous_row) if previous_row else None
    local = build_local_states(row, previous_actions)
    return {
        "qec": augment_local_state(local.qec, current, previous),
        "qint": augment_local_state(local.qint, current, previous),
        "do": augment_local_state(local.do, current, previous),
        "qw": augment_local_state(local.qw, current, previous),
    }


def select_actions(mode: str, row: Mapping[str, object], rng: np.random.Generator) -> dict[str, float]:
    if mode == "defaults":
        return dict(ACTION_DEFAULTS)

    if mode == "random":
        return {
            agent: float(rng.uniform(*ACTION_BOUNDS[agent]))
            for agent in AGENT_ORDER
        }

    if mode == "hold":
        return {
            "qec": _float(row, "Qec", ACTION_DEFAULTS["qec"]),
            "qint": _float(row, "Qint", ACTION_DEFAULTS["qint"]),
            "do": _float(row, "SO4ref", ACTION_DEFAULTS["do"]),
            "qw": _float(row, "Qw", ACTION_DEFAULTS["qw"]),
        }

    raise ValueError(f"Unknown Game2 coordinator mode: {mode}")


def clip_actions(
    actions: Mapping[str, float],
    previous_actions: Mapping[str, float] | None = None,
) -> dict[str, float]:
    clipped = {}
    for agent, value in actions.items():
        low, high = ACTION_BOUNDS[agent]
        val = float(value)
        if previous_actions is not None and agent in previous_actions:
            limit = ACTION_RATE_LIMITS[agent]
            prev = float(previous_actions[agent])
            val = float(np.clip(val, prev - limit, prev + limit))
        clipped[agent] = float(np.clip(val, low, high))
    return clipped


def atomic_replace_with_retry(
    src: str,
    dst: str,
    max_retries: int = 20,
    retry_delay: float = 0.05,
    label: str = "atomic_replace",
) -> bool:
    last_err: Exception | None = None
    for _ in range(max_retries):
        try:
            os.replace(src, dst)
            return True
        except PermissionError as exc:
            last_err = exc
            time.sleep(retry_delay)
    print(f"[GAME2] warning: {label} failed after retries ({last_err}).")
    try:
        os.remove(src)
    except OSError:
        pass
    return False


def write_actions(actions: Mapping[str, float]) -> bool:
    frame = pd.DataFrame(
        {
            "Qec": [actions["qec"]],
            "Qint": [actions["qint"]],
            "SO4ref": [actions["do"]],
            "Qw": [actions["qw"]],
        }
    )
    last_err: Exception | None = None
    for attempt in range(20):
        tmp = f"{ACTION_FILE}.{os.getpid()}.{attempt}.tmp"
        try:
            frame.to_csv(tmp, index=False)
            os.replace(tmp, ACTION_FILE)
            return True
        except OSError as exc:
            last_err = exc
            try:
                os.remove(tmp)
            except OSError:
                pass
            time.sleep(0.05)
    print(f"[GAME2] warning: write_actions failed after retries ({last_err}).")
    return False


def archive_existing_file_for_fresh_run(path: str) -> None:
    if not os.path.exists(path):
        return
    root, ext = os.path.splitext(path)
    archived = f"{root}_archived_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
    os.replace(path, archived)
    print(f"[GAME2] existing log archived -> {archived}")


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        query_limited_info = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(query_limited_info, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_coordinator_lock(label: str, run_id: str) -> None:
    os.makedirs(COMMS_DIR, exist_ok=True)
    if os.path.exists(RUN_LOCK_FILE):
        try:
            raw = open(RUN_LOCK_FILE, "r").read().strip().splitlines()
            existing_pid = int(raw[0]) if raw else -1
        except (OSError, ValueError):
            existing_pid = -1
        if _pid_is_running(existing_pid):
            raise RuntimeError(
                f"Another Game2 Python coordinator is already running "
                f"(pid={existing_pid}). Stop it before starting {label}."
            )
        try:
            os.remove(RUN_LOCK_FILE)
            print(f"[GAME2] removed stale coordinator lock -> {RUN_LOCK_FILE}")
        except OSError:
            pass

    try:
        with open(RUN_LOCK_FILE, "x") as f:
            f.write(f"{os.getpid()}\n{label}\n{run_id}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except FileExistsError as exc:
        raise RuntimeError("Game2 coordinator lock appeared during startup; aborting.") from exc


def release_coordinator_lock() -> None:
    try:
        raw = open(RUN_LOCK_FILE, "r").read().strip().splitlines()
        existing_pid = int(raw[0]) if raw else -1
    except (OSError, ValueError):
        return
    if existing_pid == os.getpid():
        try:
            os.remove(RUN_LOCK_FILE)
        except OSError:
            pass


def clear_stale_flags() -> None:
    for path in (FLAG_STATE, FLAG_ACTION, FLAG_ERROR):
        if os.path.exists(path):
            os.remove(path)
            print(f"[GAME2] removed stale flag -> {path}")


def append_log(
    row: Mapping[str, object],
    augmented: Mapping[str, np.ndarray],
    actions: Mapping[str, float],
    step: int,
    mode: str,
    run_id: str,
) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "step": step,
        "mode": mode,
        "time": _float(row, "time", np.nan),
        "game2_context_dim": game2_context_dim(),
        "qec_aug_dim": len(augmented["qec"]),
        "qint_aug_dim": len(augmented["qint"]),
        "do_aug_dim": len(augmented["do"]),
        "qw_aug_dim": len(augmented["qw"]),
        "Qec": actions["qec"],
        "Qint": actions["qint"],
        "SO4ref": actions["do"],
        "Qw": actions["qw"],
    }
    out = pd.DataFrame([record])
    header = not os.path.exists(LOG_FILE)
    try:
        out.to_csv(LOG_FILE, mode="a", header=header, index=False)
    except PermissionError as exc:
        print(f"[GAME2] warning: log append delayed ({exc}).")


def self_test() -> None:
    row = {
        "time": 1.0,
        "Qec": 2.0,
        "Qint": 61944.0,
        "SO4ref": 2.0,
        "Qw": 300.0,
        "SNO_1": 5.5,
        "SNO_2": 3.8,
        "SNO_3": 7.1,
        "SNH_2": 7.0,
        "SNH_4": 1.3,
        "SNH_5": 0.55,
        "SO_4": 2.0,
        "SO_5": 1.6,
        "TSS_3": 4700.0,
        "TSS_5": 3260.0,
        "Flow": 20648.0,
        "CODTN": 2.1,
        "TSS_5_1d": 3260.0,
        "SND_5_1d": 0.53,
        "SRT_3d": 17.44,
        "SNH_5_1d": 0.55,
        "EQI_1d": 1671.58,
    }
    augmented = build_augmented_states(row, None, None)
    expected_dims = {"qec": 34, "qint": 34, "do": 34, "qw": 37}
    for agent, expected in expected_dims.items():
        actual = len(augmented[agent])
        if actual != expected:
            raise AssertionError(f"{agent} augmented dim {actual}, expected {expected}")
    print("[GAME2] self-test passed: augmented state dimensions are correct.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Game2 four-agent integration coordinator.")
    parser.add_argument(
        "--mode",
        choices=("hold", "defaults", "random"),
        default="hold",
        help="Action source for the integration bridge. 'hold' repeats current plant actions.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many action handshakes. 0 means run until interrupted.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--append-log",
        action="store_true",
        help="Append to the existing integration log instead of archiving it first.",
    )
    parser.add_argument(
        "--keep-stale-flags",
        action="store_true",
        help="Do not clear existing flag_state.run/flag_action.run at startup.",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=0.0,
        help=(
            "Exit after this many seconds without a new MATLAB state flag, "
            "once at least one handshake has completed. 0 disables it."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    os.makedirs(COMMS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    acquire_coordinator_lock(args.mode, run_id)
    if not args.append_log:
        archive_existing_file_for_fresh_run(LOG_FILE)
    if not args.keep_stale_flags:
        clear_stale_flags()

    rng = np.random.default_rng(args.seed)

    print("\n[GAME2] Coordinator started")
    print(f"  run id: {run_id}")
    print(f"  mode: {args.mode}")
    print(f"  context dim: {game2_context_dim()}")
    print(f"  comms dir: {COMMS_DIR}")
    print("  waiting for MATLAB flag_state.run...\n")

    previous_row: dict[str, object] | None = None
    previous_actions: dict[str, float] | None = None
    step = 0
    last_activity = time.monotonic()

    try:
        while True:
            while not os.path.exists(FLAG_STATE):
                if (
                    args.idle_timeout
                    and step > 0
                    and time.monotonic() - last_activity >= args.idle_timeout
                ):
                    print(f"[GAME2] idle timeout reached ({args.idle_timeout:.1f}s).")
                    return
                time.sleep(0.05)

            row = read_state_row()
            augmented = build_augmented_states(row, previous_row, previous_actions)
            actions = clip_actions(
                select_actions(args.mode, row, rng),
                previous_actions=previous_actions,
            )

            if not write_actions(actions):
                print("[GAME2] action write failed; not creating flag_action.run.", file=sys.stderr)
                continue

            if os.path.exists(FLAG_STATE):
                os.remove(FLAG_STATE)
            open(FLAG_ACTION, "w").close()

            append_log(row, augmented, actions, step, args.mode, run_id)

            print(
                f"[GAME2] step={step:05d} t={_float(row, 'time', np.nan):.4f} "
                f"Qec={actions['qec']:.3f} Qint={actions['qint']:.1f} "
                f"SO4ref={actions['do']:.3f} Qw={actions['qw']:.3f} "
                f"aug_dims=({len(augmented['qec'])}, {len(augmented['qint'])}, "
                f"{len(augmented['do'])}, {len(augmented['qw'])})"
            )

            previous_row = dict(row)
            previous_actions = actions
            step += 1
            last_activity = time.monotonic()
            if args.max_steps and step >= args.max_steps:
                print(f"[GAME2] max steps reached ({args.max_steps}).")
                break

    except KeyboardInterrupt:
        print("\n[GAME2] interrupted by user.")
    finally:
        release_coordinator_lock()


if __name__ == "__main__":
    main()
