"""
agents/ctrl_proportional.py — Proportional baseline controller
===============================================================
Simple SNH-based controller for Qint.
Used as baseline for comparison with the SAC agent.

BEFORE RUNNING:
  1. python agents/ctrl_proportional.py
  2. MATLAB: run matlab/RL_main_simple.m
"""

import os
import time
import numpy as np
import pandas as pd

# =====================================================
# PATHS
# =====================================================
_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_AGENTS_DIR, '..'))

COMMS_DIR   = os.path.join(_ROOT, 'comms')
LOG_DIR     = os.path.join(_ROOT, 'logs')
os.makedirs(COMMS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

STATE_FILE  = os.path.join(COMMS_DIR, 'state.csv')
ACTION_FILE = os.path.join(COMMS_DIR, 'action.csv')
FLAG_STATE  = os.path.join(COMMS_DIR, 'flag_state.run')
FLAG_ACTION = os.path.join(COMMS_DIR, 'flag_action.run')
LOG_FILE    = os.path.join(LOG_DIR,   'baseline_log.csv')

QINT_MIN = 5000.0
QINT_MAX = 61944.0

# BSM2 constants (Nam et al. 2023)
AE_FIXED  = 4000.0    # kWh/d — fixed aeration energy
EC_FIXED  = 0.8       # kg COD/d — fixed external carbon (Qec=2 m³/d * 400gCOD/L / 1e6)

log_records = []

print("\n[PROPORTIONAL] Baseline controller started")
print(f"  Comms dir: {COMMS_DIR}\n")

# =====================================================
# READ STATE
# =====================================================

def read_state():
    while True:
        try:
            df  = pd.read_csv(STATE_FILE)
            row = df.iloc[-1]
            state = np.array([
                float(row.get('SNO_2',  row.get('SNO_anox', 3.8))),
                float(row.get('SNH_in', 5.0)),
                float(row.get('Temp',   13.0)),
                float(row.get('Flow',   20648.0)),
            ], dtype=np.float32)
            return state
        except Exception as e:
            print(f"[PROPORTIONAL] error reading state.csv: {e}")
            time.sleep(0.05)

# =====================================================
# PROPORTIONAL CONTROLLER
# Qint = clip(30000 + 500*(SNH - 2), QINT_MIN, QINT_MAX)
# with exponential smoothing alpha=0.4
# =====================================================

def controller(state, prev_action=None):
    SNH    = state[1]
    action = 30000.0 + 500.0 * (SNH - 2.0)
    if prev_action is not None:
        action = prev_action + 0.4 * (action - prev_action)
    return float(np.clip(action, QINT_MIN, QINT_MAX))

# =====================================================
# WRITE ACTION — atomic
# =====================================================

def write_action(action):
    tmp = ACTION_FILE + '.tmp'
    pd.DataFrame({'Qec': [float(action)]}).to_csv(tmp, index=False)
    os.replace(tmp, ACTION_FILE)

# =====================================================
# COMPUTE J (Nam et al. 2023)
# J = 200*EQI + 40*AE + 3*PE + EC
# EC is FIXED at 0.8 kg COD/d (Qec=2 m³/d, not Qint)
# PE = 0.05 * Qint (pumping energy for internal recirculation)
# =====================================================

def compute_J(state, action):
    SNO  = state[0]
    SNH  = state[1]
    Flow = state[3]
    Qint = action

    EQI = Flow * (30.0 * SNH + 10.0 * SNO) / 1000.0
    AE  = AE_FIXED
    PE  = 0.05 * Qint
    EC  = EC_FIXED          # fixed — Qec is not controlled here
    J   = 200*EQI + 40*AE + 3*PE + 1*EC

    return J, EQI, AE, PE, EC

# =====================================================
# SAVE LOG — atomic write
# =====================================================

def save_log():
    if log_records:
        tmp = LOG_FILE + '.tmp'
        pd.DataFrame(log_records).to_csv(tmp, index=False)
        os.replace(tmp, LOG_FILE)
        print(f"[PROPORTIONAL] Log saved ({len(log_records)} steps)")

# =====================================================
# LOG STEP
# =====================================================

def log_step(step, state, action, prev_action):
    SNO, SNH = state[0], state[1]
    dQ = 0.0 if prev_action is None else abs(action - prev_action)
    J, EQI, AE, PE, EC = compute_J(state, action)

    alerts = []
    if SNH > 4.0: alerts.append("⚠ SNH high")
    if SNO < 0.5: alerts.append("⚠ SNO low")

    print(f"\n[PROPORTIONAL] step={step:05d}")
    print(f"  SNO={SNO:.4f}  SNH={SNH:.4f}  Qint={action:.1f}  dQ={dQ:.1f}")
    print(f"  J={J:.2f}  EQI={EQI:.2f}  PE={PE:.2f}  EC={EC:.2f}")
    if alerts:
        print("  " + "  ".join(alerts))

    log_records.append({
        "step": step, "SNO": SNO, "SNH": SNH,
        "Temp": state[2], "Flow": state[3],
        "Qint": action, "J": J,
        "EQI_proxy": EQI, "AE": AE, "PE": PE, "EC": EC,
    })

# =====================================================
# MAIN LOOP
# =====================================================

def main():
    step        = 0
    prev_action = None

    try:
        while True:

            while not os.path.exists(FLAG_STATE):
                time.sleep(0.05)

            state  = read_state()
            action = controller(state, prev_action)

            log_step(step, state, action, prev_action)
            write_action(action)

            if os.path.exists(FLAG_STATE):
                os.remove(FLAG_STATE)
            open(FLAG_ACTION, 'w').close()

            if step % 100 == 0:
                save_log()

            prev_action = action
            step += 1

    except KeyboardInterrupt:
        print("\n[PROPORTIONAL] Interrupted.")
    finally:
        save_log()
        print(f"[PROPORTIONAL] Finished at step {step}.")
        if log_records:
            df = pd.DataFrame(log_records)
            print(f"\n  J mean    : {df['J'].mean():.2f}")
            print(f"  SNH mean  : {df['SNH'].mean():.4f}")
            print(f"  Qint mean : {df['Qint'].mean():.1f}")

if __name__ == '__main__':
    main()