# First integration experiment (AC-9.1)

first integration: REAP vs vanilla MAPPO vs MAPPO+RND — Overcooked-AI Forced Coordination, sparse delivery-only reward.
Protocol: seeds [0, 1, 2], exact budget 5000000 env steps per arm; extrinsic metrics only.

**REAP arm shaping:** DISABLED by the scope gate.
Gate detail: warmup ladder collected 0 successful episodes (< required 25) within 120000 env steps; teacher training must not proceed on this scope.

| arm | success rate mean [range] | return mean [range] |
|-----|---------------------------|----------------------|
| reap | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] |
| vanilla_mappo | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] |
| mappo_rnd | 0.997 [0.990, 1.000] | 163.27 [153.60, 174.60] |

the REAP arm ran with shaping DISABLED by the scope-specific teacher-quality gate; its result is therefore expected to match vanilla MAPPO up to seed noise.
