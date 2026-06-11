"""
agents/ctrl_sac_qw.py - CTRL-4: Qw control via SAC.
======================================================
SAC agent for the sludge wastage flow Qw, operating on a slower
timescale than the other controllers.

    Observations : [TSS_5_1d, SND_5_1d, SRT_3d, SNH_5_1d, EQI_1d]
    Action       : Qw in [0, 450] m3/d

The agent decides once per day; MATLAB aggregates the 15-minute signals
into a daily observation before sending them to Python.
"""

import os
import sys

_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.abspath(os.path.join(_AGENTS_DIR, "..", "core"))
sys.path.insert(0, _CORE_DIR)

import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from replay_buffer import ReplayBuffer
from sac_networks import Actor, Critic
from reward import compute_reward


# =====================================================
# PATHS
# =====================================================
_ROOT = os.path.abspath(os.path.join(_AGENTS_DIR, ".."))
COMMS_DIR = os.path.join(_ROOT, "comms")
CKPT_DIR = os.path.join(_ROOT, "checkpoints")
LOG_DIR = os.path.join(_ROOT, "logs")

os.makedirs(COMMS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

STATE_FILE = os.path.join(COMMS_DIR, "state.csv")
ACTION_FILE = os.path.join(COMMS_DIR, "action.csv")
FLAG_STATE = os.path.join(COMMS_DIR, "flag_state.run")
FLAG_ACTION = os.path.join(COMMS_DIR, "flag_action.run")
FLAG_EPISODE = os.path.join(COMMS_DIR, "flag_episode.run")
EPISODE_FILE = os.path.join(COMMS_DIR, "episode_info.csv")

CKPT_FILE = os.path.join(CKPT_DIR, "ctrl4_qw_sac.pt")
BEST_REWARD_CKPT_FILE = os.path.join(CKPT_DIR, "ctrl4_qw_sac_best_reward.pt")
LOG_FILE = os.path.join(LOG_DIR, "ctrl4_qw_training.csv")


# =====================================================
# DIMENSIONS AND LIMITS
# =====================================================
STATE_DIM = 5
ACTION_DIM = 1

ACTION_MIN = 0.0
ACTION_MAX = 450.0
ACTION_DEFAULT = 300.0
DECISION_WINDOW_DAYS = 1.0


# =====================================================
# HIPERPARAMETROS SAC
# =====================================================
HIDDEN = (256, 256)
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 64
BUFFER_SIZE = 10_000
WARMUP_STEPS = 60
TRAIN_FREQ = 1
SAVE_FREQ = 25
LOG_FREQ = 1
BEST_REWARD_WINDOW = 60
BEST_REWARD_FREQ = 25
RESUME_FROM_CHECKPOINT = False
APPEND_EXISTING_LOG_ON_RESUME = False

GRAD_CLIP = 1.0

TARGET_ENTROPY = -float(ACTION_DIM)


# =====================================================
# STATE NORMALIZATION
# =====================================================
STATE_MEAN = [3762.93, 0.531962, 17.4439, 0.545891, 1671.58]   # Baseline run: TSS_5_1d, SND_5_1d, SRT_3d, SNH_5_1d, EQI_1d
STATE_STD = [254.892, 0.0396945, 2.324, 0.383135, 304.829]    # Baseline run: TSS_5_1d, SND_5_1d, SRT_3d, SNH_5_1d, EQI_1d

_STATE_MEAN_ARR = None
_STATE_STD_ARR = None


def validate_state_normalization():
    if any(value is None for value in STATE_MEAN) or any(value is None for value in STATE_STD):
        raise ValueError(
            "STATE_MEAN and STATE_STD in agents/ctrl_sac_qw.py must be filled "
            "before training CTRL-4."
        )

    mean = np.array(STATE_MEAN, dtype=np.float32)
    std = np.array(STATE_STD, dtype=np.float32)
    if np.any(std <= 0):
        raise ValueError("STATE_STD must contain only positive values for CTRL-4.")
    return mean, std


def normalize_state(state):
    if _STATE_MEAN_ARR is None or _STATE_STD_ARR is None:
        raise RuntimeError("CTRL-4 state normalisation arrays were not initialised.")
    return (state - _STATE_MEAN_ARR) / (_STATE_STD_ARR + 1e-8)


# =====================================================
# HELPER - atomic rename with retry
# =====================================================
def atomic_replace_with_retry(src, dst, max_retries=20, retry_delay=0.05, label="atomic_replace"):
    last_err = None
    for _ in range(max_retries):
        try:
            os.replace(src, dst)
            return True
        except PermissionError as exc:
            last_err = exc
            time.sleep(retry_delay)
    print(f"[CTRL4] AVISO: {label} falhou apos {max_retries} tentativas ({last_err}).")
    try:
        os.remove(src)
    except Exception:
        pass
    return False


def compute_ctrl4_reward(state_dict, action):
    return compute_reward(state_dict, action, agent="qw")


# =====================================================
# COMUNICACAO COM MATLAB
# =====================================================
def read_state():
    while True:
        try:
            df = pd.read_csv(STATE_FILE)
            row = df.iloc[-1]

            read_state.QW_MEAN_1D = float(row.get("Qw_mean_1d", ACTION_DEFAULT))
            read_state.SNO_3_1D = float(row.get("SNO_3_1d", np.nan))
            read_state.TIME_D = float(row.get("time", np.nan))

            return np.array(
                [
                    float(row["TSS_5_1d"]),
                    float(row["SND_5_1d"]),
                    float(row["SRT_3d"]),
                    float(row["SNH_5_1d"]),
                    float(row["EQI_1d"]),
                ],
                dtype=np.float32,
            )
        except Exception as exc:
            print(f"[CTRL4] erro a ler state.csv: {exc}")
            time.sleep(0.05)


read_state.QW_MEAN_1D = ACTION_DEFAULT
read_state.SNO_3_1D = np.nan
read_state.TIME_D = np.nan


def write_action(qw_value):
    """
    Atomic write robust to collisions with MATLAB on Windows.
    The column is still named Qec for historical compatibility of the
    shared action.csv file.
    """
    tmp = ACTION_FILE + ".tmp"
    pd.DataFrame({"Qec": [float(qw_value)]}).to_csv(tmp, index=False)
    return atomic_replace_with_retry(tmp, ACTION_FILE, label="write_action")


# =====================================================
# AGENTE SAC
# =====================================================
class SACAgent:
    def __init__(self):
        self.actor = Actor(STATE_DIM, ACTION_DIM, HIDDEN, ACTION_MIN, ACTION_MAX)
        self.critic = Critic(STATE_DIM, ACTION_DIM, HIDDEN)
        self.critic_tgt = Critic(STATE_DIM, ACTION_DIM, HIDDEN)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        self.critic_tgt.requires_grad_(False)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)

        self.buffer = ReplayBuffer(STATE_DIM, ACTION_DIM, BUFFER_SIZE)
        self.total_steps = 0
        self.state_mean_t = torch.as_tensor(_STATE_MEAN_ARR, dtype=torch.float32)
        self.state_std_t = torch.as_tensor(_STATE_STD_ARR + 1e-8, dtype=torch.float32)
        self.action_mid = (ACTION_MAX + ACTION_MIN) / 2.0
        self.action_scale = (ACTION_MAX - ACTION_MIN) / 2.0

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
        return float(np.random.uniform(ACTION_MIN, ACTION_MAX))

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return None

        s, a, r, s_, d = self.buffer.sample(BATCH_SIZE)

        s = torch.as_tensor(s, dtype=torch.float32)
        a = torch.as_tensor(a, dtype=torch.float32)
        r = torch.as_tensor(r, dtype=torch.float32)
        s_ = torch.as_tensor(s_, dtype=torch.float32)
        d = torch.as_tensor(d, dtype=torch.float32)

        s_n = (s - self.state_mean_t) / self.state_std_t
        s_n_ = (s_ - self.state_mean_t) / self.state_std_t
        a_n = (a - self.action_mid) / self.action_scale

        with torch.no_grad():
            a_next, lp_next = self.actor.sample(s_n_)
            a_next_n = (a_next - self.action_mid) / self.action_scale
            q1t, q2t = self.critic_tgt(s_n_, a_next_n)
            q_tgt = torch.min(q1t, q2t) - self.alpha * lp_next
            q_target = r + GAMMA * (1 - d) * q_tgt

        q1, q2 = self.critic(s_n, a_n)
        loss_critic = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.opt_critic.zero_grad()
        loss_critic.backward()
        critic_gnorm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.opt_critic.step()

        a_new, lp = self.actor.sample(s_n)
        a_new_n = (a_new - self.action_mid) / self.action_scale
        q1n, q2n = self.critic(s_n, a_new_n)
        loss_actor = (self.alpha * lp - torch.min(q1n, q2n)).mean()
        self.opt_actor.zero_grad()
        loss_actor.backward()
        actor_gnorm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.opt_actor.step()

        loss_alpha = -(self.log_alpha * (lp + TARGET_ENTROPY).detach()).mean()
        self.opt_alpha.zero_grad()
        loss_alpha.backward()
        self.opt_alpha.step()

        for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)

        return {
            "loss_critic": loss_critic.item(),
            "loss_actor": loss_actor.item(),
            "alpha": self.alpha,
            "critic_gnorm": float(critic_gnorm),
            "actor_gnorm": float(actor_gnorm),
        }

    def save(self, path=None):
        path = path or CKPT_FILE
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_tgt": self.critic_tgt.state_dict(),
                "log_alpha": self.log_alpha.detach(),
                "total_steps": self.total_steps,
            },
            path,
        )
        print(f"[CTRL4] Checkpoint guardado -> {path}")

    def load(self, path=None):
        path = path or CKPT_FILE
        ck = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.critic_tgt.load_state_dict(ck["critic_tgt"])
        self.log_alpha = torch.tensor(ck["log_alpha"].item(), requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)
        self.total_steps = ck.get("total_steps", 0)
        print(f"[CTRL4] Checkpoint carregado - step {self.total_steps}")


# =====================================================
# LOG
# =====================================================
log_records = []
recent_rewards = []
_log_initialized = False

LOG_COLUMNS = [
    "schema_version",
    "episode",
    "episode_start_day",
    "episode_stop_day",
    "step",
    "time_d",
    "mode",
    "TSS_5_1d",
    "SND_5_1d",
    "SRT_3d",
    "SNH_5_1d",
    "EQI_1d",
    "Qw_mean_1d",
    "SNO_3_1d",
    "prev_Qw",
    "Qw_prev",
    "Qw_new",
    "Qw_applied_for_reward",
    "Qw_command_next",
    "reward_uses_previous_action",
    "reward",
    "buffer",
    "EQI",
    "AE",
    "PE",
    "EC",
    "J",
    "J_manual",
    "ratio",
    "Qint_used",
    "Qec_used",
    "Qw_used",
    "SRT_used",
    "loss_critic",
    "loss_actor",
    "alpha",
    "critic_gnorm",
    "actor_gnorm",
]


def save_log():
    global log_records, _log_initialized
    if not log_records:
        return
    mode = "a" if _log_initialized or (
        RESUME_FROM_CHECKPOINT and os.path.exists(LOG_FILE)
    ) else "w"
    header = mode == "w" or not os.path.exists(LOG_FILE)
    try:
        pd.DataFrame(log_records).reindex(columns=LOG_COLUMNS).to_csv(
            LOG_FILE, mode=mode, header=header, index=False)
        log_records = []
        _log_initialized = True
    except PermissionError as exc:
        print(f"[CTRL4] AVISO: save_log adiado ({exc}).")


def archive_existing_file_for_fresh_run(path):
    if RESUME_FROM_CHECKPOINT or not os.path.exists(path):
        return
    root, ext = os.path.splitext(path)
    archived = f'{root}_archived_{time.strftime("%Y%m%d_%H%M%S")}{ext}'
    try:
        os.replace(path, archived)
        print(f"[CTRL4] Ficheiro existente arquivado -> {archived}")
    except PermissionError as exc:
        raise RuntimeError(
            f"[CTRL4] Nao consegui arquivar {path}; fechar Excel/editores "
            "antes de iniciar novo run para evitar perder dados."
        ) from exc


def load_existing_log():
    if not os.path.exists(LOG_FILE):
        return []
    try:
        df = pd.read_csv(LOG_FILE)
        return df.to_dict(orient="records")
    except Exception as exc:
        print(f"[CTRL4] AVISO: nao foi possivel carregar log existente: {exc}")
        return []


# =====================================================
# MAIN LOOP
# =====================================================
def main():
    global _STATE_MEAN_ARR, _STATE_STD_ARR

    _STATE_MEAN_ARR, _STATE_STD_ARR = validate_state_normalization()

    print("\n[CTRL4 - Qw SAC] Iniciado")
    print(f"  Qw in [{ACTION_MIN:.1f}, {ACTION_MAX:.1f}] m3/d")
    print("  Observacoes: [TSS_5_1d, SND_5_1d, SRT_3d, SNH_5_1d, EQI_1d]")
    print(f"  Janela de decisao: {DECISION_WINDOW_DAYS:.1f} dia")
    print(f"  Warmup: {WARMUP_STEPS} steps")
    print(f"  Gradient clip: {GRAD_CLIP}")
    print(f"  Comms dir: {COMMS_DIR}\n")
    archive_existing_file_for_fresh_run(LOG_FILE)

    agent = SACAgent()
    episode = 0
    step = 0

    if RESUME_FROM_CHECKPOINT and APPEND_EXISTING_LOG_ON_RESUME:
        existing_records = load_existing_log()
        if existing_records:
            log_records.extend(existing_records)
            print(f"[CTRL4] Log existente carregado - {len(existing_records)} rows")
    elif RESUME_FROM_CHECKPOINT and os.path.exists(LOG_FILE):
        print("[CTRL4] Resume activo: novo segmento de log sera iniciado.")

    if RESUME_FROM_CHECKPOINT and os.path.exists(CKPT_FILE):
        agent.load()
        step = agent.total_steps
    elif RESUME_FROM_CHECKPOINT:
        print("[CTRL4] Sem checkpoint previo - arranque fresco.")

    prev_state = None
    prev_action = None
    prev_qw_applied = ACTION_DEFAULT
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
                    print(f"\n{'=' * 50}")
                    print(
                        f"[CTRL4] EPISODIO {episode}  "
                        f"(dias {ep.get('start_day', '?')}-{ep.get('stop_day', '?')})"
                    )
                    print(f"{'=' * 50}\n")
                except Exception:
                    episode_start_day = np.nan
                    episode_stop_day = np.nan
                    pass
                os.remove(FLAG_EPISODE)
                prev_state = None
                prev_action = None
                prev_qw_applied = ACTION_DEFAULT

            while not os.path.exists(FLAG_STATE):
                time.sleep(0.05)

            obs_state = read_state()
            state = obs_state.copy()

            if step < WARMUP_STEPS:
                action = agent.random_action()
                mode = "random"
            else:
                action = agent.select_action(state)
                mode = "SAC"

            action = float(np.clip(action, ACTION_MIN, ACTION_MAX))

            reward = 0.0
            breakdown = {}
            reward_uses_previous_action = prev_state is not None
            if prev_state is not None:
                reward_state = {
                    "EQI_1d": float(obs_state[4]),
                    "Qw_mean_1d": float(read_state.QW_MEAN_1D),
                    "SRT_3d": float(obs_state[2]),
                    "SNH_5_1d": float(obs_state[3]),
                    "SNO_3_1d": float(read_state.SNO_3_1D),
                }
                reward, breakdown = compute_ctrl4_reward(reward_state, prev_action)
                agent.buffer.add(prev_state, [prev_action], reward, state, 0.0)

            losses = None
            if step >= WARMUP_STEPS and step % TRAIN_FREQ == 0:
                losses = agent.train_step()

            print(f"\n[CTRL4] ep={episode:03d} step={step:05d} ({mode})")
            print(
                f"  TSS_5_1d={obs_state[0]:.3f}  SND_5_1d={obs_state[1]:.3f}  "
                f"SRT_3d={obs_state[2]:.3f}  SNH_5_1d={obs_state[3]:.3f}  "
                f"EQI_1d={obs_state[4]:.3f}"
            )
            print(
                f"  prev_Qw={prev_qw_applied:.3f}  Qw={action:.3f}  "
                f"Qw_mean_1d={read_state.QW_MEAN_1D:.3f}  reward={reward:.4f}"
            )
            if losses:
                print(
                    f"  Lc={losses['loss_critic']:.4f}  "
                    f"La={losses['loss_actor']:.4f}  "
                    f"alpha={losses['alpha']:.4f}  "
                    f"|g_c|={losses['critic_gnorm']:.2f}  "
                    f"|g_a|={losses['actor_gnorm']:.2f}"
                )

            log_records.append(
                {
                    "schema_version": 2,
                    "episode": episode,
                    "episode_start_day": episode_start_day,
                    "episode_stop_day": episode_stop_day,
                    "step": step,
                    "mode": mode,
                    "TSS_5_1d": obs_state[0],
                    "SND_5_1d": obs_state[1],
                    "SRT_3d": obs_state[2],
                    "SNH_5_1d": obs_state[3],
                    "EQI_1d": obs_state[4],
                    "Qw_mean_1d": read_state.QW_MEAN_1D,
                    "SNO_3_1d": read_state.SNO_3_1D,
                    "time_d": read_state.TIME_D,
                    "prev_Qw": prev_qw_applied,
                    "Qw_prev": prev_action,
                    "Qw_new": action,
                    "Qw_applied_for_reward": prev_action,
                    "Qw_command_next": action,
                    "reward_uses_previous_action": int(reward_uses_previous_action),
                    "reward": reward,
                    "buffer": len(agent.buffer),
                    **breakdown,
                    **(losses or {}),
                }
            )
            recent_rewards.append(reward)
            if len(recent_rewards) > BEST_REWARD_WINDOW:
                del recent_rewards[:-BEST_REWARD_WINDOW]

            if step % LOG_FREQ == 0:
                save_log()
            if step % BEST_REWARD_FREQ == 0 and step > 0:
                mean_recent_reward = (
                    float(np.mean(recent_rewards))
                    if recent_rewards else np.nan
                )
                if (np.isfinite(mean_recent_reward) and
                        mean_recent_reward > best_recent_reward):
                    best_recent_reward = mean_recent_reward
                    agent.total_steps = step
                    agent.save(BEST_REWARD_CKPT_FILE)
                    print(
                        "[CTRL4] Melhor checkpoint por reward guardado "
                        f"(step={step}, "
                        f"mean_reward_{BEST_REWARD_WINDOW}="
                        f"{mean_recent_reward:.4f}) "
                        f"-> {BEST_REWARD_CKPT_FILE}"
                    )
            if step % SAVE_FREQ == 0 and step > 0:
                agent.total_steps = step
                agent.save()

            action_written = write_action(action)
            if not action_written:
                print(
                    f"[CTRL4] AVISO: step {step} sem action.csv novo; "
                    "flag_action.run nao criado. MATLAB usara o default "
                    "se atingir timeout.",
                    file=sys.stderr,
                )
                continue
            prev_qw_applied = action
            if os.path.exists(FLAG_STATE):
                os.remove(FLAG_STATE)
            open(FLAG_ACTION, "w").close()

            prev_state = state.copy()
            prev_action = action
            step += 1

    except KeyboardInterrupt:
        print("\n[CTRL4] Interrompido.")
    finally:
        save_log()
        agent.total_steps = step
        agent.save()
        print(f"[CTRL4] Terminado no step {step}.")


if __name__ == "__main__":
    main()
