# Pre-registered report: baseline-completion experiments (H2 / H4)

**Status: TEMPLATE — committed before the experiments run.**

## Protocol (fixed before execution)

- Tasks: Overcooked-AI Forced Coordination AND Counter Circuit, sparse reward, horizon 400
- H2 arms: REAP-shaped MAPPO vs. COMA vs. QMIX (in-repo, shared harness)
- H4 arms: REAP-shaped MAPPO vs. MAPPO+RND vs. MAPPO+count-based, equal tuning budget
- COMA and QMIX must pass MPE Spread sanity validation against published ranges
  BEFORE their results may be used in any claim (validation reports linked here)
- Seeds: {0, 1, 2}; identical fixed step budget per arm (exact stop)
- Invariance check: on a layout the unshaped baseline solves, shaped and unshaped
  MAPPO must converge to comparable final extrinsic returns (setting recorded here)
- Claims reference the extrinsic channel only
- Outcome policy: win, loss, and null results reported with equal prominence;
  H4 failure means the construction is not justified (draft Section 5.2) and is
  reported as such.

## Results (filled after runs; protocol section immutable)

(pending)

## Conclusion (extrinsic metrics only)

(pending)
