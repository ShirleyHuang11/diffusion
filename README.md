# REAP: Reachability-Estimated Adaptive Potentials

Research codebase for **REAP** — a training-time framework that densifies
credit assignment in sparse-reward cooperative MARL. A diffusion trajectory
model estimates per-state *propensity* (policy-conditioned success
probability) and *feasibility* (policy-independent reachability); the
feasibility-gated propensity becomes a potential-based shaping reward that
preserves the optimal policy while accelerating learning.

See `REAP_proposal.md` for the research design and `plan.md` for the
implementation plan.

## Setup

Python 3.10+, single GPU (CPU works for tests and smoke runs):

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# smoke-mode infrastructure exercise (random policy, sparse Cramped Room)
python -m reap.train --config configs/smoke_random_cramped.yaml

# resume an interrupted run from its latest checkpoint
python -m reap.train --config configs/smoke_random_cramped.yaml --resume

# tests
pytest
```

## Layout

```
reap/
  config.py      # YAML config loading + strict validation
  seeding.py     # deterministic seeding (python / numpy / torch)
  metrics.py     # JSONL+CSV metrics with strict reward-channel separation
  checkpoint.py  # integrity-verified checkpoint save/load
  train.py       # config-driven entrypoint
  envs/          # cooperative multi-agent env interface + Overcooked wrapper
configs/         # smoke-mode and paper-mode run configs
tests/           # pytest suite
```

## Principles

- **Sparse means sparse**: the extrinsic reward channel carries only the
  task (delivery) reward; dense/shaped/intrinsic signals live in separate,
  clearly-labeled channels and never enter result claims.
- **Reproducibility**: pinned dependencies, deterministic seeding, same-seed
  runs produce identical metric streams, integrity-checked checkpoints.
- **Training-time scaffolding only**: generative teacher and shaping
  machinery never exist at deployment/evaluation time.
