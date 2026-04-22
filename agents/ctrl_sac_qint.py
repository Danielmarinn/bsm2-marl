"""
agents/ctrl_sac_qint.py — CTRL-2: Qint control via SAC
==========================================================
SAC agent for internal recirculation control (Qint).
Observations: [SNO_2, SNO_1, SNO_3, COD/TN]
Action:        Qint ∈ [5000, 61944] m³/d

BEFORE RUNNING:
  1. pip install torch numpy pandas
  2. python agents/ctrl_sac_qint.py
  3. (MATLAB) run matlab/RL_main_simple.m
"""

import os
import sys

# --- Path fix: add core/ before any local imports ---
_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR   = os.path.abspath(os.path.join(_AGENTS_DIR, '..', 'core'))
sys.path.insert(0, _CORE_DIR)

import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from replay_buffer import ReplayBuffer
from sac_networks  import Actor, Critic
from reward        import compute_reward

# =====================================================
# PATHS
# =====================================================
_ROOT        = os.path.abspath(os.path.join(_AGENTS_DIR, '..'))
COMMS_DIR    = os.path.join(_ROOT, 'comms')
CKPT_DIR     = os.path.join(_ROOT, 'checkpoints')
LOG_DIR      = os.path.join(_ROOT, 'logs')

os.makedirs(COMMS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,  exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

STATE_FILE   = os.path.join(COMMS_DIR, 'state.csv')
ACTION_FILE  = os.path.join(COMMS_DIR, 'action.csv')
FLAG_STATE   = os.path.join(COMMS_DIR, 'flag_state.run')
FLAG_ACTION  = os.path.join(COMMS_DIR, 'flag_action.run')
FLAG_EPISODE = os.path.join(COMMS_DIR, 'flag_episode.run')
EPISODE_FILE = os.path.join(COMMS_DIR, 'episode_info.csv')

CKPT_FILE = os.path.join(CKPT_DIR, 'ctrl2_qint_sac.pt')
LOG_FILE  = os.path.join(LOG_DIR,  'ctrl2_qint_training.csv')

# =====================================================
# DIMENSIONS AND LIMITS
# =====================================================
STATE_DIM  = 4
ACTION_DIM = 1

QINT_MIN = 5_000.0
QINT_MAX = 61_944.0

# =====================================================
# SAC HYPERPARAMETERS
# =====================================================
HIDDEN       = (256, 256)
LR_ACTOR     = 3e-4
LR_CRITIC    = 3e-4
LR_ALPHA     = 3e-4
GAMMA        = 0.99
TAU          = 0.005
BATCH_SIZE   = 256
BUFFER_SIZE  = 50_000
WARMUP_STEPS = 1_000
TRAIN_FREQ   = 1
SAVE_FREQ    = 500
LOG_FREQ     = 100

TARGET_ENTROPY = -float(ACTION_DIM)

# =====================================================
# STATE NORMALISATION
# [SNO_2, SNO_1, SNO_3, COD/TN]
# Values computed from observed training data (days 245-609):
#   SNO_2: mean~3.8, std~1.8
#   SNO_1: mean~5.5, std~1.9
#   SNO_3: mean~7.1, std~1.7
#   CODTN: mean~2.1, std~0.3 (nearly constant)
# =====================================================
STATE_MEAN = np.array([3.8, 5.5, 7.1, 2.1], dtype=np.float32)
STATE_STD  = np.array([1.8, 1.9, 1.7, 0.5], dtype=np.float32)

def normalize_state(state):
    return (state - STATE_MEAN) / (STATE_STD + 1e-8)

# =====================================================
# MATLAB COMMUNICATION
# =====================================================

def read_state():
    """Read state.csv written by MATLAB.
    Expected columns: SNO_2, SNO_1, SNO_3, CODTN, SNH_in, Flow
    """
    while True:
        try:
            df  = pd.read_csv(STATE_FILE)
            row = df.iloc[-1]

            state = np.array([
                float(row.get('SNO_2',  row.get('SNO_anox', 3.8))),
                float(row.get('SNO_1',  5.5)),
                float(row.get('SNO_3',  7.1)),
                float(row.get('CODTN',  2.1)),
            ], dtype=np.float32)

            read_state.SNH  = float(row.get('SNH_in', row.get('SNH_2', 5.0)))
            read_state.Flow = float(row.get('Flow', 20648.0))

            return state

        except Exception as e:
            print(f'[CTRL2] error reading state.csv: {e}')
            time.sleep(0.05)

read_state.SNH  = 5.0
read_state.Flow = 20648.0


def write_action(qint_value):
    """Atomic write to prevent partial reads by MATLAB.

    Both column names are written for backward compatibility:
    - Qint: canonical public-facing name
    - Qec: legacy name kept for older MATLAB readers
    """
    tmp = ACTION_FILE + '.tmp'
    value = float(qint_value)
    pd.DataFrame({'Qint': [value], 'Qec': [value]}).to_csv(tmp, index=False)
    os.replace(tmp, ACTION_FILE)

# =====================================================
# SAC AGENT
# =====================================================

class SACAgent:

    def __init__(self):
        self.actor      = Actor(STATE_DIM, ACTION_DIM, HIDDEN, QINT_MIN, QINT_MAX)
        self.critic     = Critic(STATE_DIM, ACTION_DIM, HIDDEN)
        self.critic_tgt = Critic(STATE_DIM, ACTION_DIM, HIDDEN)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        self.critic_tgt.requires_grad_(False)

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)

        self.buffer      = ReplayBuffer(STATE_DIM, ACTION_DIM, BUFFER_SIZE)
        self.total_steps = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_action(self, state, deterministic=False):
        s = torch.FloatTensor(normalize_state(state)).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                a = self.actor.deterministic(s)
            else:
                a, _ = self.actor.sample(s)
        return float(a.squeeze())

    def random_action(self):
        return float(np.random.uniform(QINT_MIN, QINT_MAX))

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return None

        s, a, r, s_, d = self.buffer.sample(BATCH_SIZE)

        s  = torch.FloatTensor(s)
        a  = torch.FloatTensor(a)
        r  = torch.FloatTensor(r)
        s_ = torch.FloatTensor(s_)
        d  = torch.FloatTensor(d)

        mean_t = torch.FloatTensor(STATE_MEAN)
        std_t  = torch.FloatTensor(STATE_STD + 1e-8)
        s_n    = (s  - mean_t) / std_t
        s_n_   = (s_ - mean_t) / std_t

        mid    = (QINT_MAX + QINT_MIN) / 2.0
        scale  = (QINT_MAX - QINT_MIN) / 2.0
        a_n    = (a - mid) / scale

        with torch.no_grad():
            a_next, lp_next = self.actor.sample(s_n_)
            a_next_n = (a_next - mid) / scale
            q1t, q2t = self.critic_tgt(s_n_, a_next_n)
            q_tgt    = torch.min(q1t, q2t) - self.alpha * lp_next
            q_target = r + GAMMA * (1 - d) * q_tgt

        q1, q2      = self.critic(s_n, a_n)
        loss_critic = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.opt_critic.zero_grad()
        loss_critic.backward()
        self.opt_critic.step()

        a_new, lp  = self.actor.sample(s_n)
        a_new_n    = (a_new - mid) / scale
        q1n, q2n   = self.critic(s_n, a_new_n)
        loss_actor = (self.alpha * lp - torch.min(q1n, q2n)).mean()
        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        loss_alpha = -(self.log_alpha * (lp + TARGET_ENTROPY).detach()).mean()
        self.opt_alpha.zero_grad()
        loss_alpha.backward()
        self.opt_alpha.step()

        for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)

        return {
            'loss_critic': loss_critic.item(),
            'loss_actor':  loss_actor.item(),
            'alpha':       self.alpha,
        }

    def save(self, path=None):
        path = path or CKPT_FILE
        torch.save({
            'actor':       self.actor.state_dict(),
            'critic':      self.critic.state_dict(),
            'critic_tgt':  self.critic_tgt.state_dict(),
            'log_alpha':   self.log_alpha.detach(),
            'total_steps': self.total_steps,
        }, path)
        print(f'[CTRL2] Checkpoint saved → {path}')

    def load(self, path=None):
        path = path or CKPT_FILE
        ck = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(ck['actor'])
        self.critic.load_state_dict(ck['critic'])
        self.critic_tgt.load_state_dict(ck['critic_tgt'])
        self.log_alpha   = torch.tensor(ck['log_alpha'].item(), requires_grad=True)
        self.opt_alpha   = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)
        self.total_steps = ck.get('total_steps', 0)
        print(f'[CTRL2] Checkpoint loaded — step {self.total_steps}')

# =====================================================
# LOG — atomic write prevents corruption if Excel is open
# =====================================================
log_records = []

def save_log():
    if log_records:
        tmp = LOG_FILE + '.tmp'
        pd.DataFrame(log_records).to_csv(tmp, index=False)
        os.replace(tmp, LOG_FILE)

# =====================================================
# MAIN LOOP
# =====================================================

def main():
    print('\n[CTRL2 — Qint SAC] Started')
    print(f'  Qint ∈ [{QINT_MIN:.0f}, {QINT_MAX:.0f}] m³/d')
    print(f'  Observations: [SNO_2, SNO_1, SNO_3, COD/TN]')
    print(f'  Warmup: {WARMUP_STEPS} steps')
    print(f'  Comms dir: {COMMS_DIR}\n')

    agent   = SACAgent()
    episode = 0
    step    = 0

    if os.path.exists(CKPT_FILE):
        agent.load()
        step = agent.total_steps

    prev_state  = None
    prev_action = None

    try:
        while True:

            # --- Detect new episode ---
            if os.path.exists(FLAG_EPISODE):
                episode += 1
                try:
                    ep = pd.read_csv(EPISODE_FILE).iloc[0].to_dict()
                    print(f'\n{"="*50}')
                    print(f'[CTRL2] EPISODE {episode}  '
                          f'(days {ep.get("start_day","?")}–{ep.get("stop_day","?")})')
                    print(f'{"="*50}\n')
                except Exception:
                    pass
                os.remove(FLAG_EPISODE)
                prev_state  = None
                prev_action = None

            # --- Wait for MATLAB ---
            while not os.path.exists(FLAG_STATE):
                time.sleep(0.05)

            # --- Read state ---
            state = read_state()
            SNH   = read_state.SNH
            Flow  = read_state.Flow

            # --- Select action ---
            if step < WARMUP_STEPS:
                action = agent.random_action()
                mode   = 'random'
            else:
                action = agent.select_action(state)
                mode   = 'SAC'

            action = float(np.clip(action, QINT_MIN, QINT_MAX))

            # --- Reward and buffer ---
            reward    = 0.0
            breakdown = {}
            if prev_state is not None:
                reward_state = np.array([state[0], SNH, 0.0, Flow])
                reward, breakdown = compute_reward(reward_state, prev_action)
                agent.buffer.add(prev_state, [prev_action], reward, state, 0.0)

            # --- Train ---
            losses = None
            if step >= WARMUP_STEPS and step % TRAIN_FREQ == 0:
                losses = agent.train_step()

            # --- Print ---
            print(f'\n[CTRL2] ep={episode:03d} step={step:05d} ({mode})')
            print(f'  SNO_2={state[0]:.3f}  SNO_1={state[1]:.3f}  '
                  f'SNO_3={state[2]:.3f}  COD/TN={state[3]:.2f}')
            print(f'  SNH={SNH:.3f}  Qint={action:.0f}  reward={reward:.4f}')
            if losses:
                print(f'  Lc={losses["loss_critic"]:.4f}  '
                      f'La={losses["loss_actor"]:.4f}  '
                      f'α={losses["alpha"]:.4f}')
            if SNH > 4.0:
                print('  ⚠ SNH high (> 4 g N/m³)')

            # --- Log ---
            log_records.append({
                'episode': episode, 'step': step, 'mode': mode,
                'SNO_2': state[0], 'SNO_1': state[1],
                'SNO_3': state[2], 'CODTN': state[3],
                'SNH': SNH, 'Flow': Flow,
                'Qint': action, 'reward': reward,
                'buffer': len(agent.buffer),
                **breakdown, **(losses or {}),
            })

            if step % LOG_FREQ  == 0: save_log()
            if step % SAVE_FREQ == 0 and step > 0:
                agent.total_steps = step
                agent.save()

            # --- Reply to MATLAB ---
            write_action(action)
            if os.path.exists(FLAG_STATE):
                os.remove(FLAG_STATE)
            open(FLAG_ACTION, 'w').close()

            prev_state  = state.copy()
            prev_action = action
            step       += 1

    except KeyboardInterrupt:
        print('\n[CTRL2] Interrupted.')
    finally:
        save_log()
        agent.total_steps = step
        agent.save()
        print(f'[CTRL2] Finished at step {step}.')

if __name__ == '__main__':
    main()
