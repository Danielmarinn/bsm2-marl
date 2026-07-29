# Methodology Notes

This project started from a practical question: can a multi-agent SAC controller coordinate the main BSM2 manipulated variables without hiding the plant logic behind a black-box simulator?

The answer is implemented as a file-based MATLAB/Python co-simulation. MATLAB advances the BSM2 plant and exports a state row. Python reads that state, chooses bounded actions, writes them back, and logs the reward and diagnostic metrics.

## Reward

The final multi-agent controller uses one shared plant objective:

```text
J = 200 * EQI + 40 * AE + 3 * PE + EC
r = max(-5 * J / J_manual + 5, -5)
```

The manual baseline value is computed on the same official 245-609 day evaluation window. This keeps the learning signal tied to the same tradeoff used for the thesis comparison.

## Attention

The implementation is G2ANet-inspired, not a literal copy of every design choice in the reference paper.

Actor-side communication has two stages:

1. a straight-through binary hard gate decides which peers are active;
2. a soft attention layer weights the active peer embeddings.

The hard gate is a binary logistic-noise / straight-through gate over connect-or-ignore decisions. It is not categorical Gumbel-softmax. The critic uses soft inter-agent attention only, because a hard gate on value regression did not add a clear sparsity incentive.

## CTDE

Execution is decentralized: each actor uses its own local observation and recurrent state. Training is centralized: the critic receives the joint sequence and estimates a single shared `Q_tot`.

Per-agent credit is handled by the actor update. For agent `i`, the sampled action for agent `i` is inserted into the joint replay action row, while the other agents' actions remain at replay values. The critic is frozen during this actor step, so gradients flow through the selected agent's action path.

## Limitations

- No tested configuration improves the manual baseline while respecting the official effluent limits. The wide original action ranges make a biologically unsafe underaeration operating point reachable, and the reward floor weakens the recovery signal once the plant has collapsed. Restricting the action ranges removes the collapse and restores ammonia compliance, but the resulting controller has a higher official OCI than the manual baseline (about 12.5% higher).
- Ammonium violations remain the main water-quality failure mode.
- Full reproduction requires MATLAB, Simulink, and the official BSM2 distribution, which is not included here.
- Runtime checkpoints and large logs are excluded from Git; this repository keeps the implementation and the supporting evidence, not the full experiment archive.
