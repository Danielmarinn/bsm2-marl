"""
core/reward.py - joint and diagnostic rewards for the BSM2 controllers.

Final multi-agent reward:

    J(t)     = 200 * EQI(t) + 40 * AE(t) + 3 * PE(t) + EC(t)
    ratio(t) = J(t) / J_MANUAL
    r(t)     = max(-5 * ratio(t) + 5, REWARD_FLOOR)

The manual baseline is computed on the official 245-609 day evaluation
window from the same logged components. The single-agent helpers are kept for
the diagnostic controllers.

References:
    Alex, J. et al. (2008). Benchmark Simulation Model No. 2 (BSM2).
    Gernaey, K. V. et al. (eds.) (2014). Benchmarking of Control
        Strategies for Wastewater Treatment Plants. IWA STR No. 23.
    Nam, S. M. et al. (2023). Soft actor-critic deep reinforcement
        learning for cost-effective biological wastewater treatment.
        Water Research 235, 119917.
"""

import numpy as np


# =====================================================
# CONSTANTS
# =====================================================
# Single-agent legacy constants kept for compute_reward.
AE_FIXED = 4_000.0
CODEC = 400_000.0
QINT_FIXED = 61_944.0
QEC_FIXED = 2.0
FLOW_FIXED = 20_648.0
SO4REF_FIXED = 2.0
QW_FIXED = 300.0
SRT_REF = 17.44

W_EQI = 200.0
W_AE = 40.0
W_PE = 3.0
W_EC = 1.0
W_SRT = 40.0  # single-agent qw only

# =====================================================
# J_MANUAL is the BSM2 default-controller cost on the official 245..609 d
# evaluation window, computed as J = 200*EQI + 40*AE + 3*PE + EC from the
# 2026-05-26 baseline run (post bridge-fix; EXP-002/EXP-004 set_param path).
# Source: prog_RL/results/bsm2_manual_baseline_official_summary.csv
#   EQI = 5576.665, AE = 4225.433, PE = 445.453, EC = 800.000
# Previous value 1293523.0 (from run_bsm2_manual_baseline.m:213) was 0.54%
# higher than the actually-measured baseline; corrected so ratio = 1.0
# corresponds exactly to the manual baseline.
# =====================================================
J_MANUAL = 1286486.67

# Joint-reward floor for the final multi-agent run.
# A -5.0 floor preserves gradient separation between merely bad and
# catastrophic ratios. The earlier -1.0 diagnostic floor saturated too early
# for the long Game2 runs.
REWARD_FLOOR = -5.0

# Single-agent diagnostic floor (kept for compute_reward and the
# ctrl_sac_{qec,qint,do,qw}.py diagnostics). It is intentionally separate from
# the final multi-agent floor.
REWARD_FLOOR_NAM = -1.0


# =====================================================
# COMPONENT HELPERS (single-agent compatibility)
# =====================================================
def compute_EQI_proxy(SNO, SNH, Flow):
    return Flow * (30.0 * SNH + 10.0 * SNO) / 1000.0


def compute_PE(Qint):
    return 0.05 * Qint


def compute_PE_qw(Qw):
    return 0.05 * Qw


def compute_EC(Qec):
    return Qec * CODEC / 1_000.0


def compute_AE_from_so4(so4):
    so4 = max(float(so4), 0.0)
    return AE_FIXED * (so4 / SO4REF_FIXED)


def compute_SRT_penalty(SRT_3d):
    if not np.isfinite(SRT_3d) or SRT_3d <= 0:
        return AE_FIXED
    return AE_FIXED * abs(SRT_3d - SRT_REF) / SRT_REF


def compute_J(SNO, SNH, Flow, Qint, Qec):
    EQI = compute_EQI_proxy(SNO, SNH, Flow)
    AE = AE_FIXED
    PE = compute_PE(Qint)
    EC = compute_EC(Qec)
    J = W_EQI * EQI + W_AE * AE + W_PE * PE + W_EC * EC
    return J, {"EQI": EQI, "AE": AE, "PE": PE, "EC": EC}


# =====================================================
# SINGLE-AGENT REWARD (unchanged in semantics; used by diagnostics)
# =====================================================
def compute_reward(state, action, agent='qint', qint_other=None, qec_other=None):
    if agent == 'do':
        SO4 = float(state['SO_4'])
        SNH_4 = float(state['SNH_4'])
        SNH_5 = float(state['SNH_5'])
        SNO_3 = float(state['SNO_3'])

        SNH = max(SNH_4, SNH_5)
        Flow = FLOW_FIXED
        qint = QINT_FIXED if qint_other is None else float(qint_other)
        qec = QEC_FIXED if qec_other is None else float(qec_other)

        EQI = compute_EQI_proxy(SNO_3, SNH, Flow)
        AE = compute_AE_from_so4(SO4)
        PE = compute_PE(qint)
        EC = compute_EC(qec)
        J = W_EQI * EQI + W_AE * AE + W_PE * PE + W_EC * EC

        r = -5.0 * (J / J_MANUAL) + 5.0
        r = max(r, REWARD_FLOOR_NAM)

        return float(r), {
            "EQI": EQI, "AE": AE, "PE": PE, "EC": EC,
            "J": J, "J_manual": J_MANUAL, "ratio": J / J_MANUAL,
            "Qint_used": qint, "Qec_used": qec, "SO4_used": SO4,
            "SNH_used": SNH, "SNO_used": SNO_3,
        }

    if agent == 'qw':
        EQI_1d = float(state['EQI_1d'])
        Qw_mean = float(state.get('Qw_mean_1d', action))
        SRT_3d = float(state.get('SRT_3d', np.nan))

        qint = QINT_FIXED if qint_other is None else float(qint_other)
        qec = QEC_FIXED if qec_other is None else float(qec_other)

        EQI = EQI_1d
        AE = AE_FIXED
        PE = compute_PE(qint) + compute_PE_qw(Qw_mean)
        EC = compute_EC(qec)
        SRT_pen = compute_SRT_penalty(SRT_3d)
        J = W_EQI * EQI + W_AE * AE + W_PE * PE + W_EC * EC + W_SRT * SRT_pen

        r = -5.0 * (J / J_MANUAL) + 5.0
        r = max(r, REWARD_FLOOR_NAM)

        return float(r), {
            "EQI": EQI, "AE": AE, "PE": PE, "EC": EC,
            "SRT_penalty": SRT_pen,
            "J": J, "J_manual": J_MANUAL, "ratio": J / J_MANUAL,
            "Qint_used": qint, "Qec_used": qec, "Qw_used": Qw_mean,
            "SRT_used": SRT_3d,
        }

    SNO = float(state[0])
    SNH = float(state[1])
    Flow = float(state[3])

    if agent == 'qint':
        qint = float(action)
        qec = QEC_FIXED if qec_other is None else float(qec_other)
    elif agent == 'qec':
        qec = float(action)
        qint = QINT_FIXED if qint_other is None else float(qint_other)
    else:
        raise ValueError(f"Unknown agent: {agent!r}")

    J, comps = compute_J(SNO, SNH, Flow, qint, qec)
    r = -5.0 * (J / J_MANUAL) + 5.0
    r = max(r, REWARD_FLOOR_NAM)

    return float(r), {**comps, "J": J, "J_manual": J_MANUAL,
                      "ratio": J / J_MANUAL,
                      "Qint_used": qint, "Qec_used": qec}


# =====================================================
# MULTI-AGENT JOINT REWARD
# =====================================================
def compute_joint_reward(row_data, actions, previous_actions=None):
    """
    Final frozen joint reward.

        J     = 200 * EQI + 40 * AE + 3 * PE + EC
        ratio = J / J_MANUAL
        r     = max(-5 * ratio + 5, REWARD_FLOOR)

    EQI/AE/PE/EC are consumed verbatim from the BSM2-aligned bridge
    metrics in row_data; the weights match the BSM2 OCI form and the
    MATLAB baseline at prog_RL/matlab/run_bsm2_manual_baseline.m:218.

    previous_actions is accepted for call-site compatibility and unused;
    smoothness is enforced upstream by the delta-action rate limits in
    game2_abstraction.apply_action_deltas.
    """
    del previous_actions

    qec = float(actions['qec'])
    qint = float(actions['qint'])
    so4_ref = float(actions['do'])
    qw = float(actions['qw'])

    REQUIRED = ("EQI_bsm2_inst", "AE_bsm2_inst", "PE_bsm2_inst",
                "EC_bsm2_inst")
    for key in REQUIRED:
        v = row_data.get(key, np.nan)
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"compute_joint_reward: {key} is not numeric.")
        if not np.isfinite(v):
            raise ValueError(f"compute_joint_reward: {key} is not finite.")

    EQI = float(row_data["EQI_bsm2_inst"])
    AE = float(row_data["AE_bsm2_inst"])
    PE = float(row_data["PE_bsm2_inst"])
    EC = float(row_data["EC_bsm2_inst"])

    J = W_EQI * EQI + W_AE * AE + W_PE * PE + W_EC * EC
    ratio = J / J_MANUAL

    r_unclipped = -5.0 * ratio + 5.0
    r = max(r_unclipped, REWARD_FLOOR)

    return float(r), {
        "EQI": EQI,
        "AE": AE,
        "PE": PE,
        "EC": EC,
        "J": J,
        "J_manual": J_MANUAL,
        "ratio": ratio,
        "reward_unclipped": r_unclipped,
        "reward_floor": REWARD_FLOOR,
        "qec": qec,
        "qint": qint,
        "so4_ref": so4_ref,
        "qw": qw,
    }
