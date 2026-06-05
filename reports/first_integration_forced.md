# First integration experiment (AC-9.1)

first integration: REAP vs vanilla MAPPO vs MAPPO+RND — Overcooked-AI Forced Coordination, sparse delivery-only reward.
Protocol: seeds [0, 1, 2], exact budget 5000000 env steps per arm; extrinsic metrics only.

**REAP arm shaping:** ENABLED.
Gate detail: gates passed.

| arm | success rate mean ± 95% CI | return mean ± 95% CI |
|-----|-----------------------------|------------------------|
| reap | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] |
| vanilla_mappo | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] |
| mappo_rnd | 0.997 [0.982, 1.011] | 163.27 [136.94, 189.60] |

## Conclusion

REAP (shaping enabled) final extrinsic success 0.000, vanilla MAPPO 0.000, MAPPO+RND 0.997.

H4 is NOT supported in this configuration: the enabled, calibrated REAP signal did not outperform generic novelty (RND). The calibrated propensity honestly reads near-zero for a policy that never succeeds, so the shaping signal vanishes exactly where exploration is the bottleneck — a first-class negative result reported with equal prominence.

Evidence: quality report `reports/teacher_quality_hybrid_forced.json`, calibration report `reports/calibration_hybrid_forced.json`; warmup/potential-table/fidelity/pipeline-summary paths, per-arm metrics paths, per-seed shaping events (with preemption-restart accounting) and wall-clock/GPU-memory in the JSON artifact.

REAP arm ran with shaping enabled.
