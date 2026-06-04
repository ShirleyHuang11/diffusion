# PBRS invariance sanity check

PBRS invariance sanity check (not a statistical claim).

Potential: overcooked_progress (reap/envs/overcooked_env.py::progress_potential): 0.05*min(ingredients_loaded,6) + 0.30*min(soups_ready,2) + 0.15*min(held_dishes,2) + 0.50*min(held_soups,2); beta=5.0, gamma equal to the MAPPO discount; potential is zero at every episode end.
Protocol: seeds [0, 1, 2], exact budget 5000000 env steps; arms `invariance_shaped_rnd_cramped` (shaped) vs. `probe_mappo_rnd_cramped` (unshaped).

| metric | shaped (mean [min, max]) | unshaped (mean [min, max]) | comparable |
|--------|--------------------------|----------------------------|------------|
| success_rate_final | 0.997 [0.990, 1.000] | 1.000 [1.000, 1.000] | True |
| episode_return_mean_final | 193.733 [189.600, 196.600] | 193.867 [183.800, 202.400] | True |

**Invariance holds: True**
