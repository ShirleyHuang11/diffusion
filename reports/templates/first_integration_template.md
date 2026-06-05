# Pre-registered report: first integration experiment

**Status: TEMPLATE — committed before the experiment runs.**

## Protocol (fixed before execution)

- Task: Overcooked-AI Forced Coordination, sparse delivery-only reward, horizon 400
- Arms: REAP-shaped MAPPO; vanilla MAPPO; MAPPO+RND — identical trunk, wrappers, and budgets
- Seeds: {0, 1, 2} (fixed)
- Budget: 10,000,000 environment steps per run (exact stop), identical for every arm
  - **AMENDMENT (Round 12, disclosed):** the plan records all numeric budgets as
    configurable defaults, and the executed protocol family used 5,000,000 steps
    per arm (identical across every arm, exact stop, protocol-validated in the
    generated report). The 10M default above was never executed; this note
    exists so the template and the executed protocol records cannot diverge
    silently. The executed budget is authoritative in
    `reports/first_integration_forced.json` (`protocol.expected_env_steps`).
- Primary metrics: final extrinsic success rate; extrinsic episode return (rolling window)
- Claims may reference the extrinsic channel only; shaped/intrinsic values are diagnostics
- REAP validity gate: the held-out calibration check (predicted propensity vs. realized
  success). A calibration failure escalates the automated ladder and is reported here
  as a first-class outcome.
- Outcome policy: win, loss, and null results are all reported with the same prominence.

## Results (filled after runs complete; no edits to the protocol section)

| arm | seed | final success rate | final return mean | env steps |
|-----|------|--------------------|-------------------|-----------|
| (pending) | | | | |

- Mean ± CI per arm: (pending)
- Calibration evidence (per refresh: ECE/Brier, ladder actions): (pending)
- Refresh/snapshot log summary: (pending)
- Wall-clock and memory: (pending)

## Conclusion (extrinsic metrics only)

(pending)
