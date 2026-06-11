"""
Multi-agent SAC coordinator for the Game2 bridge.

Trainable modes:
  - sac-local:  four independent actors, local observations only.
  - sac-game2:  four independent actors, local obs + 28-feature Game2 context.
  - sac-ctde:   four decentralized actors (local obs only) + twin G2ANet
                central critic trained on joint (obs, actions).  CTDE as in
                Nam et al. (2023).
  - sac-ctde-rnn: same decentralized actors + recurrent soft-attention
                central critic trained on sequence chunks.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Mapping

import numpy as np
import pandas as pd
import torch

AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(AGENTS_DIR, ".."))
CORE_DIR = os.path.join(ROOT, "core")
sys.path.insert(0, CORE_DIR)

from game2_abstraction import (  # noqa: E402
    ACTION_DEFAULTS,
    ACTION_DELTA_BOUNDS,
    AGENT_ORDER,
    apply_action_deltas,
    clip_action_deltas,
)
from game2_sac import AGENT_LOCAL_DIMS, IndependentSACAgent, SACConfig  # noqa: E402
from game2_ctde_sac import CTDEMultiAgentSAC, CTDESACConfig  # noqa: E402
from game2_recurrent_ctde_sac import (  # noqa: E402
    RecurrentCTDEMultiAgentSAC,
    RecurrentCTDESACConfig,
)
from reward import compute_EC, compute_joint_reward  # noqa: E402

from ctrl_game2_coordinator import (  # noqa: E402
    COMMS_DIR,
    FLAG_ACTION,
    FLAG_STATE,
    LOG_DIR,
    acquire_coordinator_lock,
    archive_existing_file_for_fresh_run,
    build_augmented_states,
    build_local_states,
    clear_stale_flags,
    game2_context_dim,
    read_state_row,
    release_coordinator_lock,
    write_actions,
    _float,
)


LOG_FILE = os.path.join(LOG_DIR, "game2_sac_training_log.csv")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
ACTION_SPACE_VERSION = "delta_v1"


def local_state_dict(local_states) -> dict[str, np.ndarray]:
    return {
        "qec": local_states.qec,
        "qint": local_states.qint,
        "do": local_states.do,
        "qw": local_states.qw,
    }


def build_states(
    mode: str,
    row: Mapping[str, object],
    previous_row: Mapping[str, object] | None,
    previous_actions: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    if mode in ("sac-local", "sac-ctde", "sac-ctde-rnn"):
        return local_state_dict(build_local_states(row, previous_actions))
    if mode == "sac-game2":
        return build_augmented_states(row, previous_row, previous_actions)
    raise ValueError(f"Unknown SAC Game2 mode: {mode}")


def make_agents(mode: str, batch_size: int, warmup_steps: int) -> dict[str, IndependentSACAgent]:
    agents = {}
    for name in AGENT_ORDER:
        low, high = ACTION_DELTA_BOUNDS[name]
        state_dim = AGENT_LOCAL_DIMS[name]
        if mode == "sac-game2":
            state_dim += game2_context_dim()
        agents[name] = IndependentSACAgent(
            name,
            SACConfig(
                state_dim=state_dim,
                action_low=low,
                action_high=high,
                batch_size=batch_size,
                warmup_steps=warmup_steps,
            ),
        )
    return agents


def make_ctde_agent(batch_size: int, warmup_steps: int) -> CTDEMultiAgentSAC:
    return CTDEMultiAgentSAC(
        CTDESACConfig(batch_size=batch_size, warmup_steps=warmup_steps)
    )


def make_recurrent_ctde_agent(
    batch_size: int,
    warmup_steps: int,
    seq_len: int,
    burn_in: int,
) -> RecurrentCTDEMultiAgentSAC:
    return RecurrentCTDEMultiAgentSAC(
        RecurrentCTDESACConfig(
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            seq_len=seq_len,
            burn_in=burn_in,
        )
    )


def checkpoint_path(prefix: str, mode: str, run_id: str, suffix: str) -> str:
    safe_prefix = prefix.strip() or "game2_multiagent_sac"
    return os.path.join(CKPT_DIR, f"{safe_prefix}_{mode}_{run_id}_{suffix}.pt")


def save_checkpoint(
    agents: Mapping[str, IndependentSACAgent] | None,
    mode: str,
    run_id: str,
    step: int,
    prefix: str,
    suffix: str,
    ctde_agent: CTDEMultiAgentSAC | None = None,
) -> str:
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = checkpoint_path(prefix, mode, run_id, suffix)
    if ctde_agent is not None:
        payload = {
            "schema_version": 1,
            "action_space": ACTION_SPACE_VERSION,
            "mode": mode,
            "run_id": run_id,
            "step": step,
            "ctde_agent": ctde_agent.state_dict(),
        }
    else:
        payload = {
            "schema_version": 1,
            "action_space": ACTION_SPACE_VERSION,
            "mode": mode,
            "run_id": run_id,
            "step": step,
            "agents": {name: agent.state_dict() for name, agent in agents.items()},
        }
    torch.save(payload, path)
    return path


def load_checkpoint(
    agents: Mapping[str, IndependentSACAgent] | None,
    path: str,
    ctde_agent: CTDEMultiAgentSAC | None = None,
) -> tuple[str, int]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("action_space") != ACTION_SPACE_VERSION:
        raise ValueError(
            "Checkpoint uses the legacy absolute-action policy space. "
            "Start a fresh run after the delta-action update."
        )
    if ctde_agent is not None:
        ctde_agent.load_state_dict(data["ctde_agent"])
    elif agents is not None:
        for name, agent_state in data["agents"].items():
            if name in agents:
                agents[name].load_state_dict(agent_state)
    return str(data.get("run_id", "unknown")), int(data.get("step", 0))


def _optional_float(row: Mapping[str, object], key: str) -> float:
    return _float(row, key, np.nan)


def compute_bsm2_instant_metrics(
    row: Mapping[str, object],
    actions: Mapping[str, float],
) -> dict[str, float]:
    """
    Best-effort instantaneous BSM2 metrics for logging.

    These do not replace the official final evaluation in perf_plant_bsm2.m,
    but they keep the long run auditable while it is training.
    """
    i_xb = 0.08
    i_xp = 0.06
    f_p = 0.08

    eff_q = _optional_float(row, "Eff_Q")
    eff_tss = _optional_float(row, "Eff_TSS")
    eff_si = _optional_float(row, "Eff_SI")
    eff_ss = _optional_float(row, "Eff_SS")
    eff_xi = _optional_float(row, "Eff_XI")
    eff_xs = _optional_float(row, "Eff_XS")
    eff_xbh = _optional_float(row, "Eff_XBH")
    eff_xba = _optional_float(row, "Eff_XBA")
    eff_xp = _optional_float(row, "Eff_XP")
    eff_sno = _optional_float(row, "Eff_SNO")
    eff_snh = _optional_float(row, "Eff_SNH")
    eff_snd = _optional_float(row, "Eff_SND")
    eff_xnd = _optional_float(row, "Eff_XND")

    if all(
        np.isfinite(v)
        for v in (
            eff_q, eff_tss, eff_si, eff_ss, eff_xi, eff_xs, eff_xbh, eff_xba,
            eff_xp, eff_sno, eff_snh, eff_snd, eff_xnd,
        )
    ):
        cod = eff_si + eff_ss + eff_xi + eff_xs + eff_xbh + eff_xba + eff_xp
        snkj = eff_snh + eff_snd + eff_xnd + i_xb * (eff_xbh + eff_xba) + i_xp * (eff_xi + eff_xp)
        bod5 = 0.25 * (eff_ss + eff_xs + (1.0 - f_p) * (eff_xbh + eff_xba))
        eqi = eff_q * (2.0 * eff_tss + cod + 30.0 * snkj + 10.0 * eff_sno + 2.0 * bod5) / 1000.0
        ntot = eff_sno + snkj
    else:
        eqi = np.nan
        ntot = np.nan
        cod = np.nan
        bod5 = np.nan

    # BSM2 aeration energy relation for the AS reactors. Missing KLa values
    # are treated as NaN so the official MATLAB evaluator remains authoritative.
    kla1 = _optional_float(row, "KLa1")
    kla2 = _optional_float(row, "KLa2")
    kla3 = _optional_float(row, "KLa3")
    kla4 = _optional_float(row, "KLa4")
    kla5 = _optional_float(row, "KLa5")
    vols = (1500.0, 1500.0, 3000.0, 3000.0, 3000.0)
    if all(np.isfinite(v) for v in (kla1, kla2, kla3, kla4, kla5)):
        ae = 8.0 * sum(k * v for k, v in zip((kla1, kla2, kla3, kla4, kla5), vols)) / (1.8 * 1000.0)
    else:
        ae = np.nan

    qint = float(actions["qint"])
    qw = float(actions["qw"])
    qr = _float(row, "Qr", 20648.0)
    pe = 0.004 * qint + 0.008 * qr + 0.05 * qw
    ec = compute_EC(float(actions["qec"]))

    return {
        "EQI_bsm2_inst": eqi,
        "AE_bsm2_inst": ae,
        "PE_bsm2_inst": pe,
        "EC_bsm2_inst": ec,
        "Ntot_eff_inst": ntot,
        "COD_eff_inst": cod,
        "BOD5_eff_inst": bod5,
        "SNH_eff_inst": eff_snh,
        "TSS_eff_inst": eff_tss,
    }


def compute_rewards(
    row: Mapping[str, object],
    actions: Mapping[str, float],
    previous_actions: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Single minimalist joint reward shared by all four agents."""
    if "Eff_TSS" not in row:
        raise KeyError("state.csv is missing Eff_TSS; refusing to train without effluent TSS for logging.")
    eff_tss = _float(row, "Eff_TSS", np.nan)
    if not np.isfinite(eff_tss):
        raise ValueError("state.csv Eff_TSS is not finite.")

    bsm2_metrics = compute_bsm2_instant_metrics(row, actions)
    required = (
        "EQI_bsm2_inst", "AE_bsm2_inst", "PE_bsm2_inst",
        "EC_bsm2_inst", "SNH_eff_inst",
    )
    missing = [k for k in required if not np.isfinite(bsm2_metrics.get(k, np.nan))]
    if missing:
        raise ValueError(
            f"state.csv is missing finite BSM2-aligned reward inputs {missing}; "
            "refusing to train with stale or partial bridge data."
        )

    r, info = compute_joint_reward(bsm2_metrics, actions, previous_actions=previous_actions)

    rewards = {name: r for name in AGENT_ORDER}
    metrics = {
        "reward_total": r,
        "reward_unclipped": info["reward_unclipped"],
        "reward_floor": info["reward_floor"],
        "J_joint":        info["J"],
        "J_manual_joint": info["J_manual"],
        "ratio_joint":    info["ratio"],
        "EQI_joint": info["EQI"],
        "AE_joint":  info["AE"],
        "PE_joint":  info["PE"],
        "EC_joint":  info["EC"],
        "J_qec":  info["J"],
        "J_qint": info["J"],
        "J_do":   info["J"],
        "J_qw":   info["J"],
        "EQI_qw": info["EQI"],
    }
    metrics.update(bsm2_metrics)
    return rewards, metrics


def append_log(
    row: Mapping[str, object],
    actions: Mapping[str, float],
    proposed_actions: Mapping[str, float] | None,
    policy_deltas: Mapping[str, float] | None,
    proposed_policy_deltas: Mapping[str, float] | None,
    reward_actions: Mapping[str, float] | None,
    rewards: Mapping[str, float],
    metrics: Mapping[str, float],
    losses: Mapping[str, Mapping[str, float] | None],
    step: int,
    mode: str,
    run_id: str,
    attention_stats: Mapping[str, float] | None = None,
) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "step": step,
        "mode": mode,
        "time": _float(row, "time", np.nan),
        "Qec": actions["qec"],
        "Qint": actions["qint"],
        "SO4ref": actions["do"],
        "Qw": actions["qw"],
        "action_space": ACTION_SPACE_VERSION,
        "Qec_proposed": np.nan if proposed_actions is None else proposed_actions["qec"],
        "Qint_proposed": np.nan if proposed_actions is None else proposed_actions["qint"],
        "SO4ref_proposed": np.nan if proposed_actions is None else proposed_actions["do"],
        "Qw_proposed": np.nan if proposed_actions is None else proposed_actions["qw"],
        "Qec_delta": np.nan if policy_deltas is None else policy_deltas["qec"],
        "Qint_delta": np.nan if policy_deltas is None else policy_deltas["qint"],
        "SO4ref_delta": np.nan if policy_deltas is None else policy_deltas["do"],
        "Qw_delta": np.nan if policy_deltas is None else policy_deltas["qw"],
        "Qec_delta_proposed": np.nan if proposed_policy_deltas is None else proposed_policy_deltas["qec"],
        "Qint_delta_proposed": np.nan if proposed_policy_deltas is None else proposed_policy_deltas["qint"],
        "SO4ref_delta_proposed": np.nan if proposed_policy_deltas is None else proposed_policy_deltas["do"],
        "Qw_delta_proposed": np.nan if proposed_policy_deltas is None else proposed_policy_deltas["qw"],
        "Qec_delta_clip": np.nan if proposed_policy_deltas is None or policy_deltas is None else proposed_policy_deltas["qec"] - policy_deltas["qec"],
        "Qint_delta_clip": np.nan if proposed_policy_deltas is None or policy_deltas is None else proposed_policy_deltas["qint"] - policy_deltas["qint"],
        "SO4ref_delta_clip": np.nan if proposed_policy_deltas is None or policy_deltas is None else proposed_policy_deltas["do"] - policy_deltas["do"],
        "Qw_delta_clip": np.nan if proposed_policy_deltas is None or policy_deltas is None else proposed_policy_deltas["qw"] - policy_deltas["qw"],
        "Qec_clip_delta": np.nan if proposed_actions is None else proposed_actions["qec"] - actions["qec"],
        "Qint_clip_delta": np.nan if proposed_actions is None else proposed_actions["qint"] - actions["qint"],
        "SO4ref_clip_delta": np.nan if proposed_actions is None else proposed_actions["do"] - actions["do"],
        "Qw_clip_delta": np.nan if proposed_actions is None else proposed_actions["qw"] - actions["qw"],
        "Qec_applied_for_reward": np.nan if reward_actions is None else reward_actions["qec"],
        "Qint_applied_for_reward": np.nan if reward_actions is None else reward_actions["qint"],
        "SO4ref_applied_for_reward": np.nan if reward_actions is None else reward_actions["do"],
        "Qw_applied_for_reward": np.nan if reward_actions is None else reward_actions["qw"],
        "reward_uses_previous_action": int(reward_actions is not None),
        "SO_4_state": _float(row, "SO_4", np.nan),
        "SO_5_state": _float(row, "SO_5", np.nan),
        "KLa1": _float(row, "KLa1", np.nan),
        "KLa2": _float(row, "KLa2", np.nan),
        "KLa3": _float(row, "KLa3", np.nan),
        "KLa4": _float(row, "KLa4", np.nan),
        "KLa5": _float(row, "KLa5", np.nan),
        "Qr": _float(row, "Qr", np.nan),
        "SRT_3d_state": _float(row, "SRT_3d", np.nan),
        "Eff_Q": _float(row, "Eff_Q", np.nan),
        "Eff_SNO": _float(row, "Eff_SNO", np.nan),
        "Eff_SNH": _float(row, "Eff_SNH", np.nan),
        "Eff_SND": _float(row, "Eff_SND", np.nan),
        "Eff_XND": _float(row, "Eff_XND", np.nan),
        "Eff_TSS": _float(row, "Eff_TSS", np.nan),
        "reward_qec": rewards.get("qec", np.nan),
        "reward_qint": rewards.get("qint", np.nan),
        "reward_do": rewards.get("do", np.nan),
        "reward_qw": rewards.get("qw", np.nan),
        **metrics,
    }
    for agent_name in AGENT_ORDER:
        record[f"{agent_name}_loss_actor"] = np.nan
        record[f"{agent_name}_loss_critic"] = np.nan
        record[f"{agent_name}_alpha"] = np.nan

    for agent_name, loss in losses.items():
        if loss:
            record[f"{agent_name}_loss_actor"] = loss["loss_actor"]
            record[f"{agent_name}_loss_critic"] = loss["loss_critic"]
            record[f"{agent_name}_alpha"] = loss["alpha"]

    # Actor attention stats (always present so the CSV header is stable
    # across modes; NaN when there is no recurrent train_step result).
    stats = attention_stats or {}
    for key in ATTENTION_STAT_KEYS:
        record[key] = stats.get(key, np.nan)

    out = pd.DataFrame([record])
    if os.path.exists(LOG_FILE):
        existing_cols = list(pd.read_csv(LOG_FILE, nrows=0).columns)
        if existing_cols != list(record.keys()):
            raise RuntimeError(
                "Game2 log schema drift: existing log header does not match the "
                "current record schema. Archive or delete the log before "
                "appending."
            )
    out.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/smoke-test Game2 multi-agent SAC.")
    parser.add_argument(
        "--mode",
        choices=("sac-local", "sac-game2", "sac-ctde", "sac-ctde-rnn"),
        default="sac-local",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--idle-timeout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--burn-in", type=int, default=8)
    parser.add_argument("--save-freq", type=int, default=250)
    parser.add_argument("--checkpoint-prefix", default="game2_multiagent_sac")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--append-log", action="store_true")
    parser.add_argument("--keep-stale-flags", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--eval-only", action="store_true",
                        help="Eval-only mode: requires --resume-checkpoint, forces deterministic, "
                             "skips all add_transition/train_step/save_checkpoint, "
                             "writes to game2_sac_eval_log.csv.")
    return parser.parse_args()


_SELF_TEST_ROW = {
    "time": 1.0,
    "Qec": 2.0, "Qint": 61944.0, "SO4ref": 2.0, "Qw": 300.0,
    "SNO_1": 5.5, "SNO_2": 3.8, "SNO_3": 7.1,
    "SNH_2": 7.0, "SNH_4": 1.3, "SNH_5": 0.55,
    "SO_4": 2.0, "TSS_3": 4700.0, "TSS_5": 3260.0, "Eff_TSS": 12.0, "Flow": 20648.0, "CODTN": 2.1,
    "KLa1": 0.0, "KLa2": 0.0, "KLa3": 120.0, "KLa4": 120.0, "KLa5": 60.0, "Qr": 20648.0,
    "Eff_Q": 20648.0, "Eff_SI": 30.0, "Eff_SS": 1.2, "Eff_XI": 12.0, "Eff_XS": 2.0,
    "Eff_XBH": 1.5, "Eff_XBA": 0.2, "Eff_XP": 3.0, "Eff_SNO": 7.1,
    "Eff_SNH": 0.55, "Eff_SND": 0.3, "Eff_XND": 0.2,
    "TSS_5_1d": 3260.0, "SND_5_1d": 0.53, "SRT_3d": 17.44,
    "SNH_5_1d": 0.55, "EQI_1d": 1671.58, "Qw_mean_1d": 300.0,
}


def self_test() -> None:
    rng = np.random.default_rng(42)
    row = _SELF_TEST_ROW

    safe_rewards, safe_metrics = compute_rewards(row, ACTION_DEFAULTS)
    # Unit test: compute_joint_reward must implement
    #   J = 200*EQI + 40*AE + 3*PE + EC; r = max(-5*J/J_MANUAL + 5, -5).
    # Tested directly so the assertion does not depend on the bsm2 bridge.
    from reward import compute_joint_reward, J_MANUAL, REWARD_FLOOR
    r_base, info_base = compute_joint_reward(
        {"EQI_bsm2_inst": 5577.0, "AE_bsm2_inst": 4225.4,
         "PE_bsm2_inst": 445.5,  "EC_bsm2_inst": 800.0},
        ACTION_DEFAULTS,
    )
    expected_base = -5.0 * (info_base["J"] / J_MANUAL) + 5.0
    assert abs(r_base - expected_base) < 1e-9, (
        "reward must match -5*J/J_MANUAL + 5 at baseline."
    )
    r_floor, _ = compute_joint_reward(
        {"EQI_bsm2_inst": 5577.0 * 100, "AE_bsm2_inst": 4225.4 * 100,
         "PE_bsm2_inst": 445.5 * 100,  "EC_bsm2_inst": 800.0 * 100},
        ACTION_DEFAULTS,
    )
    assert abs(r_floor - REWARD_FLOOR) < 1e-9, (
        "reward must hit floor -5 at 100x baseline cost."
    )
    r_zero, _ = compute_joint_reward(
        {"EQI_bsm2_inst": 0.0, "AE_bsm2_inst": 0.0,
         "PE_bsm2_inst": 0.0, "EC_bsm2_inst": 0.0},
        ACTION_DEFAULTS,
    )
    assert abs(r_zero - 5.0) < 1e-9, (
        "reward must equal 5 at J=0."
    )
    # --- independent modes ---
    for mode in ("sac-local", "sac-game2"):
        agents = make_agents(mode, batch_size=4, warmup_steps=0)
        states = build_states(mode, row, None, ACTION_DEFAULTS)
        deltas = clip_action_deltas(
            {name: agent.select_action(states[name], rng) for name, agent in agents.items()}
        )
        actions = apply_action_deltas(deltas, previous_actions=ACTION_DEFAULTS)
        rewards, metrics = compute_rewards(row, actions)
        assert set(rewards) == set(AGENT_ORDER), f"{mode}: missing per-agent rewards"
        assert np.isfinite(metrics["reward_total"]), f"{mode}: reward_total not finite"
        # exercise one training step (needs batch; just checks no crash before buffer full)
        for agent in agents.values():
            agent.add_transition(
                states[agent.name], deltas[agent.name], rewards[agent.name],
                states[agent.name], False,
            )
            agent.train_step()
        print(f"[GAME2-SAC] self-test passed: {mode}")

    # --- sac-ctde ---
    ctde = make_ctde_agent(batch_size=4, warmup_steps=0)
    states = build_states("sac-ctde", row, None, ACTION_DEFAULTS)
    deltas = clip_action_deltas(ctde.select_actions(states, rng))
    actions = apply_action_deltas(deltas, previous_actions=ACTION_DEFAULTS)
    rewards, metrics = compute_rewards(row, actions)
    assert np.isfinite(metrics["reward_total"]), "sac-ctde: reward_total not finite"
    ctde.add_transition(states, deltas, metrics["reward_total"], states, False)
    # Fill buffer to batch_size so train_step actually runs
    for _ in range(ctde.config.batch_size):
        ctde.add_transition(states, deltas, metrics["reward_total"], states, False)
    result = ctde.train_step()
    assert result is not None, "sac-ctde: train_step returned None after buffer fill"
    assert "critic_loss" in result, "sac-ctde: missing critic_loss in train_step result"
    assert np.isfinite(result["critic_loss"]), "sac-ctde: critic_loss not finite"
    print("[GAME2-SAC] self-test passed: sac-ctde")

    # --- sac-ctde-rnn ---
    rnn = make_recurrent_ctde_agent(batch_size=2, warmup_steps=0, seq_len=4, burn_in=1)
    states = build_states("sac-ctde-rnn", row, None, ACTION_DEFAULTS)
    deltas = clip_action_deltas(rnn.select_actions(states, rng))
    actions = apply_action_deltas(deltas, previous_actions=ACTION_DEFAULTS)
    rewards, metrics = compute_rewards(row, actions)
    for _ in range(8):
        rnn.add_transition(states, deltas, metrics["reward_total"], states, False)
    result = rnn.train_step()
    assert result is not None, "sac-ctde-rnn: train_step returned None after buffer fill"
    assert "critic_loss" in result, "sac-ctde-rnn: missing critic_loss in train_step result"
    assert np.isfinite(result["critic_loss"]), "sac-ctde-rnn: critic_loss not finite"
    print("[GAME2-SAC] self-test passed: sac-ctde-rnn")


def _ctde_losses_to_per_agent(
    result: Mapping[str, float] | None,
) -> dict[str, dict[str, float] | None]:
    if result is None:
        return {name: None for name in AGENT_ORDER}
    shared_critic = result.get("critic_loss", np.nan)
    losses = {}
    for name in AGENT_ORDER:
        losses[name] = {
            "loss_actor": result.get(f"{name}_loss_actor", np.nan),
            "loss_critic": shared_critic,
            "alpha": result.get(f"{name}_alpha", np.nan),
        }
    return losses


# ACTOR attention summary scalars surfaced by the recurrent CTDE train_step
# (see RecurrentCTDEMultiAgentSAC._actor_attention_stats). Logged so the thesis
# success criteria 1-2 (hard/soft attention active) are directly verifiable in
# the CSV. The hard gate is logged on the ACTOR side (ActorHardAttention), which
# is the faithful G2ANet placement; the central critic is soft-only. NaN for
# non-recurrent modes and during warmup (no train_step result).
ATTENTION_STAT_KEYS = (
    "actor_hard_open_frac",
    "actor_hard_std",
    "actor_soft_entropy",
    "actor_soft_max",
)


def _extract_attention_stats(
    result: Mapping[str, float] | None,
) -> dict[str, float]:
    if result is None:
        return {k: np.nan for k in ATTENTION_STAT_KEYS}
    return {k: result.get(k, np.nan) for k in ATTENTION_STAT_KEYS}


def main() -> None:
    args = parse_args()
    if args.eval_only:
        if not args.resume_checkpoint:
            raise SystemExit("--eval-only requires --resume-checkpoint <path>.")
        args.deterministic = True
        global LOG_FILE
        LOG_FILE = os.path.join(LOG_DIR, "game2_sac_eval_log.csv")
        print(f"[GAME2-SAC] eval-only mode: log -> {LOG_FILE}")
    if args.self_test:
        self_test()
        return

    use_ctde = args.mode in ("sac-ctde", "sac-ctde-rnn")

    os.makedirs(COMMS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    acquire_coordinator_lock(args.mode, run_id)
    if not args.append_log:
        archive_existing_file_for_fresh_run(LOG_FILE)
    if not args.keep_stale_flags:
        clear_stale_flags()

    rng = np.random.default_rng(args.seed)

    if args.mode == "sac-ctde":
        ctde_agent: CTDEMultiAgentSAC | RecurrentCTDEMultiAgentSAC | None = make_ctde_agent(
            args.batch_size,
            args.warmup_steps,
        )
        agents: dict[str, IndependentSACAgent] | None = None
    elif args.mode == "sac-ctde-rnn":
        ctde_agent = make_recurrent_ctde_agent(
            args.batch_size,
            args.warmup_steps,
            args.seq_len,
            args.burn_in,
        )
        agents: dict[str, IndependentSACAgent] | None = None
    else:
        ctde_agent = None
        agents = make_agents(args.mode, args.batch_size, args.warmup_steps)

    if args.resume_checkpoint:
        resumed_run_id, resumed_step = load_checkpoint(
            agents, args.resume_checkpoint, ctde_agent=ctde_agent
        )
        print(
            f"[GAME2-SAC] resumed checkpoint run_id={resumed_run_id} "
            f"step={resumed_step}"
        )
    print(f"[GAME2-SAC] started mode={args.mode} run_id={run_id}")

    previous_row: dict[str, object] | None = None
    previous_states: dict[str, np.ndarray] | None = None
    previous_actions: dict[str, float] | None = dict(ACTION_DEFAULTS)
    previous_policy_deltas: dict[str, float] | None = None
    actions_before_previous: dict[str, float] | None = None
    step = 0
    last_activity = time.monotonic()

    FLAG_ERROR = os.path.join(COMMS_DIR, "flag_error.run")
    try:
      while True:
        while not os.path.exists(FLAG_STATE):
            if args.idle_timeout and step > 0 and time.monotonic() - last_activity >= args.idle_timeout:
                if not args.eval_only:
                    path = save_checkpoint(
                        agents, args.mode, run_id, step, args.checkpoint_prefix, "final",
                        ctde_agent=ctde_agent,
                    )
                    print(f"[GAME2-SAC] final checkpoint saved: {path}")
                print(f"[GAME2-SAC] idle timeout reached ({args.idle_timeout:.1f}s).")
                return
            time.sleep(0.05)

        row = read_state_row()
        states = build_states(args.mode, row, previous_row, previous_actions)

        rewards = {name: np.nan for name in AGENT_ORDER}
        metrics = {
            "reward_total": np.nan,
            "reward_unclipped": np.nan,
            "reward_floor": np.nan,
            "J_joint":        np.nan,
            "J_manual_joint": np.nan,
            "ratio_joint":    np.nan,
            "EQI_joint": np.nan,
            "AE_joint":  np.nan,
            "PE_joint":  np.nan,
            "EC_joint":  np.nan,
            "J_qec":  np.nan,
            "J_qint": np.nan,
            "J_do":   np.nan,
            "J_qw":   np.nan,
            "EQI_qw": np.nan,
            "EQI_bsm2_inst": np.nan,
            "AE_bsm2_inst":  np.nan,
            "PE_bsm2_inst":  np.nan,
            "EC_bsm2_inst":  np.nan,
            "Ntot_eff_inst": np.nan,
            "COD_eff_inst":  np.nan,
            "BOD5_eff_inst": np.nan,
            "SNH_eff_inst":  np.nan,
            "TSS_eff_inst":  np.nan,
        }
        losses = {name: None for name in AGENT_ORDER}
        attention_stats: dict[str, float] | None = None

        if (
            previous_states is not None
            and previous_actions is not None
            and previous_policy_deltas is not None
        ):
            rewards, metrics = compute_rewards(row, previous_actions, previous_actions=actions_before_previous)

            if not args.eval_only:
                if use_ctde:
                    ctde_agent.add_transition(
                        previous_states,
                        previous_policy_deltas,
                        metrics["reward_total"],
                        states,
                        done=False,
                    )
                    for _ in range(ctde_agent.config.updates_per_step):
                        train_result = ctde_agent.train_step()
                        losses = _ctde_losses_to_per_agent(train_result)
                        attention_stats = _extract_attention_stats(train_result)
                else:
                    for name, agent in agents.items():
                        agent.add_transition(
                            previous_states[name],
                            previous_policy_deltas[name],
                            rewards[name],
                            states[name],
                            done=False,
                        )
                        for _ in range(2):
                            losses[name] = agent.train_step()

        if use_ctde:
            proposed_policy_deltas = ctde_agent.select_actions(
                states,
                rng,
                deterministic=args.deterministic,
            )
        else:
            proposed_policy_deltas = {
                name: agent.select_action(states[name], rng, deterministic=args.deterministic)
                for name, agent in agents.items()
            }
        policy_deltas = clip_action_deltas(proposed_policy_deltas)
        proposed_actions = apply_action_deltas(proposed_policy_deltas, previous_actions=previous_actions)
        actions = apply_action_deltas(policy_deltas, previous_actions=previous_actions)

        if not write_actions(actions):
            print("[GAME2-SAC] action write failed; no flag_action created.", file=sys.stderr)
            continue

        if os.path.exists(FLAG_STATE):
            os.remove(FLAG_STATE)
        open(FLAG_ACTION, "w").close()

        reward_actions_for_log = previous_actions if previous_states is not None else None
        append_log(
            row,
            actions,
            proposed_actions,
            policy_deltas,
            proposed_policy_deltas,
            reward_actions_for_log,
            rewards,
            metrics,
            losses,
            step,
            args.mode,
            run_id,
            attention_stats=attention_stats,
        )
        print(
            f"[GAME2-SAC] step={step:05d} t={_float(row, 'time', np.nan):.4f} "
            f"Qec={actions['qec']:.3f} Qint={actions['qint']:.1f} "
            f"SO4ref={actions['do']:.3f} Qw={actions['qw']:.3f} "
            f"reward_total={metrics['reward_total']:.3f}"
        )

        previous_row = dict(row)
        previous_states = states
        actions_before_previous = previous_actions
        previous_actions = actions
        previous_policy_deltas = policy_deltas
        step += 1
        last_activity = time.monotonic()
        if not args.eval_only and args.save_freq and step > 0 and step % args.save_freq == 0:
            path = save_checkpoint(
                agents, args.mode, run_id, step, args.checkpoint_prefix, f"step{step:06d}",
                ctde_agent=ctde_agent,
            )
            print(f"[GAME2-SAC] checkpoint saved: {path}")
        if args.max_steps and step >= args.max_steps:
            if not args.eval_only:
                path = save_checkpoint(
                    agents, args.mode, run_id, step, args.checkpoint_prefix, "final",
                    ctde_agent=ctde_agent,
                )
                print(f"[GAME2-SAC] final checkpoint saved: {path}")
            print(f"[GAME2-SAC] max steps reached ({args.max_steps}).")
            break

    except KeyboardInterrupt:
        print("\n[GAME2-SAC] interrupted by user.")
    except Exception as exc:
        import traceback
        err_msg = traceback.format_exc()
        print(f"[GAME2-SAC] FATAL ERROR at step={step}: {exc}", file=sys.stderr)
        print(err_msg, file=sys.stderr)
        err_log = os.path.join(LOG_DIR, "game2_sac_error.log")
        with open(err_log, "a") as f:
            f.write(f"\n{'='*60}\nstep={step} time={time.strftime('%Y-%m-%d %H:%M:%S')}\n{err_msg}\n")
        try:
            open(FLAG_ERROR, "w").close()
        except OSError:
            pass
    finally:
        if not args.eval_only:
            try:
                path = save_checkpoint(
                    agents, args.mode, run_id, step, args.checkpoint_prefix, "crash_recovery",
                    ctde_agent=ctde_agent,
                )
                print(f"[GAME2-SAC] recovery checkpoint saved: {path}")
            except Exception:
                print("[GAME2-SAC] WARNING: could not save recovery checkpoint.", file=sys.stderr)
        release_coordinator_lock()


if __name__ == "__main__":
    main()
