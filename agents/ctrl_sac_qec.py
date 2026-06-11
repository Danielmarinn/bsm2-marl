"""
agents/ctrl_sac_qec.py — CTRL-1: Qec control via SAC.
==========================================================
SAC agent for the external carbon flow (Qec = QEC1, dosed in the
anoxic reactor 1).
    Observations : [SNO_2, SNO_1, SNO_3, COD/TN]
    Action       : Qec in [0, 5] m3/d   (BSM2 Table 18)

Based on Nam et al. (2023), Journal of Water Process Engineering.
"""

import os
import sys

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

CKPT_FILE = os.path.join(CKPT_DIR, 'ctrl1_qec_sac_optionB.pt')
BEST_CKPT_FILE = os.path.join(CKPT_DIR, 'ctrl1_qec_sac_optionB_best.pt')
BEST_REWARD_CKPT_FILE = os.path.join(
    CKPT_DIR, 'ctrl1_qec_sac_optionB_best_reward.pt')
LOG_FILE  = os.path.join(LOG_DIR,  'ctrl1_qec_training_optionB.csv')
DIAG_FILE = os.path.join(LOG_DIR,  'diag_qec_sno2_corr.csv')

# =====================================================
# DIMENSIONS AND LIMITS
# =====================================================
STATE_DIM  = 4
ACTION_DIM = 1

QEC_MIN = 0.0
QEC_MAX = 5.0     # BSM2 Table 18
QEC_DEFAULT = 2.0 # BSM2 default no inicio de cada episodio

# =====================================================
# HIPERPARAMETROS SAC
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
DIAG_FREQ    = 2_000
RESUME_FROM_CHECKPOINT = False

# Gradient clipping. Standard value in SB3/CleanRL.
GRAD_CLIP = 1.0

TARGET_ENTROPY = -float(ACTION_DIM)

# =====================================================
# STATE NORMALIZATION
# =====================================================
STATE_MEAN = np.array([3.8, 5.5, 7.1, 2.1], dtype=np.float32)
STATE_STD  = np.array([1.8, 1.9, 1.7, 0.5], dtype=np.float32)

def normalise_prev_qec(q):
    q = float(np.clip(q, QEC_MIN, QEC_MAX))
    return (q / 2.5) - 1.0

def normalize_state(state):
    return (state - STATE_MEAN) / (STATE_STD + 1e-8)

def build_agent_state(obs_state, prev_qec_applied):
    # prev_qec_applied is intentionally NOT part of the observation: CTRL-1 now
    # observes only plant state [SNO_2, SNO_1, SNO_3, COD/TN], identical to CTRL-2
    # (Qint), so the four single-agent baselines are compared on equal footing.
    return np.array([
        obs_state[0],
        obs_state[1],
        obs_state[2],
        obs_state[3],
    ], dtype=np.float32)

# =====================================================
# HELPER — atomic rename with retry
# =====================================================
def atomic_replace_with_retry(src, dst, max_retries=20, retry_delay=0.05,
                               label='atomic_replace'):
    """
    On Windows os.replace fails with PermissionError if the destination
    is open in another process (MATLAB reading action.csv or Excel
    viewing training.csv). Retry a few times; under normal conditions
    it succeeds on the first attempt.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            os.replace(src, dst)
            return True
        except PermissionError as e:
            last_err = e
            time.sleep(retry_delay)
    print(f'[CTRL1] AVISO: {label} falhou apos {max_retries} tentativas '
          f'({last_err}).')
    try:
        os.remove(src)
    except Exception:
        pass
    return False

# =====================================================
# COMUNICACAO COM MATLAB
# =====================================================

def read_state():
    while True:
        try:
            df  = pd.read_csv(STATE_FILE)
            row = df.iloc[-1]

            state = np.array([
                float(row.get('SNO_2', row.get('SNO_anox', 3.8))),
                float(row.get('SNO_1', 5.5)),
                float(row.get('SNO_3', 7.1)),
                float(row.get('CODTN', 2.1)),
            ], dtype=np.float32)

            read_state.SNH  = float(row.get('SNH_in', row.get('SNH_2', 5.0)))
            read_state.Flow = float(row.get('Flow', 20648.0))
            read_state.Temp = float(row.get('Temp', np.nan))
            read_state.sim_time = float(row.get('time', np.nan))

            return state

        except Exception as e:
            print(f'[CTRL1] erro a ler state.csv: {e}')
            time.sleep(0.05)

read_state.SNH  = 5.0
read_state.Flow = 20648.0
read_state.Temp = np.nan
read_state.sim_time = np.nan


def write_action(qec_value):
    """Atomic write robust to collisions with MATLAB on Windows."""
    tmp = ACTION_FILE + '.tmp'
    pd.DataFrame({'Qec': [float(qec_value)]}).to_csv(tmp, index=False)
    return atomic_replace_with_retry(tmp, ACTION_FILE, label='write_action')

# =====================================================
# AGENTE SAC
# =====================================================

class SACAgent:

    def __init__(self):
        self.actor      = Actor(STATE_DIM, ACTION_DIM, HIDDEN, QEC_MIN, QEC_MAX)
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
        self.state_mean_t = torch.as_tensor(STATE_MEAN, dtype=torch.float32)
        self.state_std_t  = torch.as_tensor(STATE_STD + 1e-8, dtype=torch.float32)
        self.action_mid   = (QEC_MAX + QEC_MIN) / 2.0
        self.action_scale = (QEC_MAX - QEC_MIN) / 2.0

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
        return float(np.random.uniform(QEC_MIN, QEC_MAX))

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return None

        s, a, r, s_, d = self.buffer.sample(BATCH_SIZE)

        s  = torch.as_tensor(s, dtype=torch.float32)
        a  = torch.as_tensor(a, dtype=torch.float32)
        r  = torch.as_tensor(r, dtype=torch.float32)
        s_ = torch.as_tensor(s_, dtype=torch.float32)
        d  = torch.as_tensor(d, dtype=torch.float32)

        s_n  = (s  - self.state_mean_t) / self.state_std_t
        s_n_ = (s_ - self.state_mean_t) / self.state_std_t
        a_n  = (a - self.action_mid) / self.action_scale

        # --- Critic ---
        with torch.no_grad():
            a_next, lp_next = self.actor.sample(s_n_)
            a_next_n = (a_next - self.action_mid) / self.action_scale
            q1t, q2t = self.critic_tgt(s_n_, a_next_n)
            q_tgt    = torch.min(q1t, q2t) - self.alpha * lp_next
            q_target = r + GAMMA * (1 - d) * q_tgt

        q1, q2      = self.critic(s_n, a_n)
        loss_critic = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.opt_critic.zero_grad()
        loss_critic.backward()
        critic_gnorm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), GRAD_CLIP)
        self.opt_critic.step()

        # --- Actor ---
        a_new, lp  = self.actor.sample(s_n)
        a_new_n    = (a_new - self.action_mid) / self.action_scale
        q1n, q2n   = self.critic(s_n, a_new_n)
        loss_actor = (self.alpha * lp - torch.min(q1n, q2n)).mean()
        self.opt_actor.zero_grad()
        loss_actor.backward()
        actor_gnorm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), GRAD_CLIP)
        self.opt_actor.step()

        # --- Alpha ---
        loss_alpha = -(self.log_alpha * (lp + TARGET_ENTROPY).detach()).mean()
        self.opt_alpha.zero_grad()
        loss_alpha.backward()
        self.opt_alpha.step()

        # --- Polyak ---
        for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)

        return {
            'loss_critic':  loss_critic.item(),
            'loss_actor':   loss_actor.item(),
            'alpha':        self.alpha,
            'critic_gnorm': float(critic_gnorm),
            'actor_gnorm':  float(actor_gnorm),
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
        print(f'[CTRL1] Checkpoint guardado -> {path}')

    def load(self, path=None):
        path = path or CKPT_FILE
        ck = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(ck['actor'])
        self.critic.load_state_dict(ck['critic'])
        self.critic_tgt.load_state_dict(ck['critic_tgt'])
        self.log_alpha   = torch.tensor(ck['log_alpha'].item(), requires_grad=True)
        self.opt_alpha   = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)
        self.total_steps = ck.get('total_steps', 0)
        print(f'[CTRL1] Checkpoint carregado — step {self.total_steps}')

# =====================================================
# LOG
# =====================================================
log_records = []
recent_rewards = []
_log_initialized = False

LOG_COLUMNS = [
    'schema_version',
    'episode', 'episode_start_day', 'episode_stop_day',
    'step', 'sim_time', 'mode',
    'SNO_2', 'SNO_1', 'SNO_3', 'CODTN',
    'prev_Qec',
    'SNH', 'Flow', 'Temp',
    'Qec_prev', 'Qec_new',
    'Qec_applied_for_reward', 'Qec_command_next',
    'reward_uses_previous_action', 'reward',
    'buffer',
    'EQI', 'AE', 'PE', 'EC', 'J', 'J_manual', 'ratio',
    'Qint_used', 'Qec_used',
    'loss_critic', 'loss_actor', 'alpha',
    'critic_gnorm', 'actor_gnorm',
]

def save_log():
    global log_records, _log_initialized
    if not log_records:
        return
    mode = 'a' if _log_initialized or (
        RESUME_FROM_CHECKPOINT and os.path.exists(LOG_FILE)
    ) else 'w'
    header = mode == 'w' or not os.path.exists(LOG_FILE)
    try:
        pd.DataFrame(log_records).reindex(columns=LOG_COLUMNS).to_csv(
            LOG_FILE, mode=mode, header=header, index=False)
        log_records = []
        _log_initialized = True
    except PermissionError as e:
        print(f'[CTRL1] AVISO: save_log adiado ({e}).')


def archive_existing_file_for_fresh_run(path):
    if RESUME_FROM_CHECKPOINT or not os.path.exists(path):
        return
    root, ext = os.path.splitext(path)
    archived = f'{root}_archived_{time.strftime("%Y%m%d_%H%M%S")}{ext}'
    try:
        os.replace(path, archived)
        print(f'[CTRL1] Ficheiro existente arquivado -> {archived}')
    except PermissionError as e:
        raise RuntimeError(
            f'[CTRL1] Nao consegui arquivar {path}; fechar Excel/editores '
            'antes de iniciar novo run para evitar perder dados.'
        ) from e

def save_qec_sno2_diag(agent, step):
    n = min(DIAG_FREQ, len(agent.buffer))
    if n <= 1:
        return None

    start = (agent.buffer.ptr - n) % agent.buffer.max_size
    idx = (start + np.arange(n)) % agent.buffer.max_size

    qec = agent.buffer.actions[idx, 0]
    sno2 = agent.buffer.next_states[idx, 0]
    if np.std(qec) < 1e-8 or np.std(sno2) < 1e-8:
        corr = np.nan
    else:
        corr = float(np.corrcoef(qec, sno2)[0, 1])

    row = pd.DataFrame([{
        'step': step,
        'corr_qec_sno2': corr,
        'mean_qec': float(np.mean(qec)),
        'mean_sno2': float(np.mean(sno2)),
    }])
    tmp = DIAG_FILE + '.tmp'
    if os.path.exists(DIAG_FILE):
        try:
            diag_df = pd.read_csv(DIAG_FILE)
            row = pd.concat([diag_df, row], ignore_index=True)
        except Exception as e:
            print(f'[CTRL1] AVISO: step {step} falha a ler diag existente '
                  f'({e}).', file=sys.stderr)
            return corr
    row.to_csv(tmp, index=False)
    ok = atomic_replace_with_retry(tmp, DIAG_FILE, label='save_qec_sno2_diag')
    if not ok:
        print(f'[CTRL1] AVISO: step {step} falha a guardar '
              'diag_qec_sno2_corr.csv; treino continua.', file=sys.stderr)
    return corr

# =====================================================
# MAIN LOOP
# =====================================================

def main():
    print('\n[CTRL1 — Qec SAC v2] Iniciado')
    print(f'  Qec in [{QEC_MIN:.1f}, {QEC_MAX:.1f}] m3/d')
    print('  Observacoes: [SNO_2, SNO_1, SNO_3, COD/TN]')
    print(f'  Warmup: {WARMUP_STEPS} steps')
    print(f'  Gradient clip: {GRAD_CLIP}  (v2)')
    print(f'  Comms dir: {COMMS_DIR}\n')
    archive_existing_file_for_fresh_run(LOG_FILE)
    archive_existing_file_for_fresh_run(DIAG_FILE)

    agent   = SACAgent()
    episode = 0
    step    = 0

    if RESUME_FROM_CHECKPOINT and os.path.exists(CKPT_FILE):
        agent.load()
        step = agent.total_steps

    prev_state  = None
    prev_action = None
    prev_qec_applied = QEC_DEFAULT
    best_diag_reward = -np.inf
    best_recent_reward = -np.inf
    episode_start_day = np.nan
    episode_stop_day = np.nan

    try:
        while True:

            if os.path.exists(FLAG_EPISODE):
                episode += 1
                try:
                    ep = pd.read_csv(EPISODE_FILE).iloc[0].to_dict()
                    episode_start_day = ep.get("start_day", np.nan)
                    episode_stop_day = ep.get("stop_day", np.nan)
                    print(f'\n{"="*50}')
                    print(f'[CTRL1] EPISODIO {episode}  '
                          f'(dias {ep.get("start_day","?")}–{ep.get("stop_day","?")})')
                    print(f'{"="*50}\n')
                except Exception:
                    episode_start_day = np.nan
                    episode_stop_day = np.nan
                    pass
                os.remove(FLAG_EPISODE)
                prev_state  = None
                prev_action = None
                prev_qec_applied = QEC_DEFAULT

            while not os.path.exists(FLAG_STATE):
                time.sleep(0.05)

            obs_state = read_state()
            SNH   = read_state.SNH
            Flow  = read_state.Flow
            Temp  = read_state.Temp
            sim_time = read_state.sim_time
            state = build_agent_state(obs_state, prev_qec_applied)

            if step < WARMUP_STEPS:
                action = agent.random_action()
                mode   = 'random'
            else:
                action = agent.select_action(state)
                mode   = 'SAC'

            action = float(np.clip(action, QEC_MIN, QEC_MAX))

            reward    = 0.0
            breakdown = {}
            reward_uses_previous_action = prev_state is not None
            if prev_state is not None:
                reward_state = np.array([obs_state[0], SNH, 0.0, Flow])
                reward, breakdown = compute_reward(reward_state, prev_action,
                                                   agent='qec')
                agent.buffer.add(prev_state, [prev_action], reward, state, 0.0)

            losses = None
            if step >= WARMUP_STEPS and step % TRAIN_FREQ == 0:
                losses = agent.train_step()

            print(f'\n[CTRL1] ep={episode:03d} step={step:05d} ({mode})')
            print(f'  SNO_2={obs_state[0]:.3f}  SNO_1={obs_state[1]:.3f}  '
                  f'SNO_3={obs_state[2]:.3f}  COD/TN={obs_state[3]:.2f}  '
                  f'prev_Qec={prev_qec_applied:.3f}')
            print(f'  SNH={SNH:.3f}  Qec={action:.3f}  reward={reward:.4f}')
            if losses:
                print(f'  Lc={losses["loss_critic"]:.4f}  '
                      f'La={losses["loss_actor"]:.4f}  '
                      f'alpha={losses["alpha"]:.4f}  '
                      f'|g_c|={losses["critic_gnorm"]:.2f}  '
                      f'|g_a|={losses["actor_gnorm"]:.2f}')
            if obs_state[2] < 0.5:
                print('  [!] SNO_3 baixo — possivel overdose de carbono')

            log_records.append({
                'schema_version': 2,
                'episode': episode,
                'episode_start_day': episode_start_day,
                'episode_stop_day': episode_stop_day,
                'step': step, 'sim_time': sim_time, 'mode': mode,
                'SNO_2': obs_state[0], 'SNO_1': obs_state[1],
                'SNO_3': obs_state[2], 'CODTN': obs_state[3],
                'prev_Qec': prev_qec_applied,
                'SNH': SNH, 'Flow': Flow, 'Temp': Temp,
                'Qec_prev': prev_action, 'Qec_new': action, 'reward': reward,
                'Qec_applied_for_reward': prev_action,
                'Qec_command_next': action,
                'reward_uses_previous_action': int(reward_uses_previous_action),
                'buffer': len(agent.buffer),
                **breakdown, **(losses or {}),
            })
            recent_rewards.append(reward)
            if len(recent_rewards) > DIAG_FREQ:
                del recent_rewards[:-DIAG_FREQ]

            if step % LOG_FREQ  == 0: save_log()
            if step % DIAG_FREQ == 0 and step > 0:
                corr_qec_sno2 = save_qec_sno2_diag(agent, step)
                mean_recent_reward = (
                    float(np.mean(recent_rewards))
                    if recent_rewards else np.nan
                )
                if (np.isfinite(mean_recent_reward) and
                        mean_recent_reward > best_recent_reward):
                    best_recent_reward = mean_recent_reward
                    agent.total_steps = step
                    agent.save(BEST_REWARD_CKPT_FILE)
                    print('[CTRL1] Melhor checkpoint por reward guardado '
                          f'(step={step}, '
                          f'mean_reward_2000={mean_recent_reward:.4f}) '
                          f'-> {BEST_REWARD_CKPT_FILE}')
                if (corr_qec_sno2 is not None and
                        np.isfinite(corr_qec_sno2) and
                        corr_qec_sno2 < -0.15 and
                        np.isfinite(mean_recent_reward) and
                        mean_recent_reward > best_diag_reward):
                    best_diag_reward = mean_recent_reward
                    agent.total_steps = step
                    agent.save(BEST_CKPT_FILE)
                    print('[CTRL1] Melhor checkpoint Qec guardado '
                          f'(step={step}, corr={corr_qec_sno2:.4f}, '
                          f'mean_reward_2000={mean_recent_reward:.4f}) '
                          f'-> {BEST_CKPT_FILE}')

            if step % SAVE_FREQ == 0 and step > 0:
                agent.total_steps = step
                agent.save()

            action_written = write_action(action)
            if not action_written:
                print(f'[CTRL1] AVISO: step {step} sem action.csv novo; '
                      'flag_action.run nao criado. MATLAB usara o default '
                      'se atingir timeout.', file=sys.stderr)
                continue
            prev_qec_applied = action
            if os.path.exists(FLAG_STATE):
                os.remove(FLAG_STATE)
            open(FLAG_ACTION, 'w').close()

            prev_state  = state.copy()
            prev_action = action
            step       += 1

    except KeyboardInterrupt:
        print('\n[CTRL1] Interrompido.')
    finally:
        save_log()
        agent.total_steps = step
        agent.save()
        print(f'[CTRL1] Terminado no step {step}.')


if __name__ == '__main__':
    main()
