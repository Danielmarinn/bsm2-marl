"""
core/reward.py — Função de recompensa partilhada entre todos os agentes
Baseada em Nam et al. (2023), eq. (1) e (2).

J(t) = 200*EQI + 40*AE + 3*PE + 1*EC
r(t) = -5 * (J_RL / J_manual) + 5,  clipped >= -1
"""

import numpy as np

# =====================================================
# CONSTANTES BSM2
# =====================================================
J_MANUAL  = 1_293_523.0   # baseline BSM2 open-loop (calculado analiticamente)
AE_FIXED  = 4_000.0       # kWh/d — aeração fixa (sem controlo de KLa)
CODEC     = 400_000.0     # g COD/m³ — concentração carbono externo
QEC_FIXED = 2.0           # m³/d — carbono externo default BSM2

# Pesos Nam et al. (2023)
W_EQI = 200.0
W_AE  = 40.0
W_PE  = 3.0
W_EC  = 1.0


def compute_EQI_proxy(SNO, SNH, Flow):
    """
    Proxy de EQI usando estados do reactor anóxico (R2).
    Captura os termos dominantes: BNKj=30 (amónia) e BNO=10 (nitrato).

    Args:
        SNO  : nitrato no reactor anóxico (g N/m³)
        SNH  : amónia no reactor anóxico (g N/m³)
        Flow : caudal de entrada Qin (m³/d)

    Returns:
        EQI proxy em kg poll.units/d
    """
    return Flow * (30.0 * SNH + 10.0 * SNO) / 1000.0


def compute_PE(Qint):
    """
    Componente variável da energia de bombagem associada a Qint.
    Baseada na eq. (255) do BSM2: coeficiente 0.05 kWh/(m³/d).

    Args:
        Qint : caudal de recirculação interna (m³/d)

    Returns:
        PE em kWh/d
    """
    return 0.05 * Qint


def compute_EC(Qec=QEC_FIXED):
    """
    Consumo de carbono externo — eq. (257) BSM2.
    Fixo nesta fase (controlado por CTRL-1 em fases futuras).

    Args:
        Qec : caudal de carbono externo (m³/d), default=2.0

    Returns:
        EC em kg COD/d
    """
    return Qec * CODEC / 1_000_000.0


def compute_J(SNO, SNH, Flow, Qint, Qec=QEC_FIXED):
    """
    Índice de desempenho J(t) — Nam et al. (2023), eq. (1).

    Returns:
        J     : valor escalar do índice
        breakdown : dicionário com componentes individuais
    """
    EQI = compute_EQI_proxy(SNO, SNH, Flow)
    AE  = AE_FIXED
    PE  = compute_PE(Qint)
    EC  = compute_EC(Qec)

    J = W_EQI * EQI + W_AE * AE + W_PE * PE + W_EC * EC

    return J, {"EQI": EQI, "AE": AE, "PE": PE, "EC": EC}


def compute_reward(state, action):
    """
    Recompensa — Nam et al. (2023), eq. (2):
        r(t) = -5 * (J_RL / J_manual) + 5
        r(t) >= -1  (clipping inferior)

    Args:
        state  : array [SNO_2, SNH_2, Temp, Flow]
        action : Qint (m³/d)

    Returns:
        r         : recompensa escalar
        breakdown : dicionário com J e componentes
    """
    SNO  = float(state[0])
    SNH  = float(state[1])
    Flow = float(state[3])
    Qint = float(action)

    J, components = compute_J(SNO, SNH, Flow, Qint)

    r = -5.0 * (J / J_MANUAL) + 5.0
    r = max(r, -1.0)

    return float(r), {**components, "J": J, "J_manual": J_MANUAL,
                      "ratio": J / J_MANUAL}
