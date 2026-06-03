# REAP Implementation Plan: Reachability-Estimated Adaptive Potentials for Sparse-Reward Cooperative MARL

## Goal Description

Build, in this currently empty repository, a single-GPU research codebase that implements the REAP framework from the design draft (appended at the bottom of this file): a diffusion trajectory model estimates per-state **propensity** (policy-conditioned success probability, via forward inpainting with classifier-free guidance on a policy embedding) and **feasibility** (policy-independent reachability, via backward goal-bridges with likelihood weighting and transition-consistency filtering); the feasibility-gated propensity defines a potential `Φ` injected as a potential-based shaping reward `r′ = r + β(γΦ(s′) − Φ(s))` into MAPPO training on sparse-reward cooperative MARL tasks.

The work is staged in independently verifiable milestones culminating in the draft's own "minimal first milestone" (draft Section 5.5): shaped MAPPO on hard Overcooked-AI layouts compared against vanilla MAPPO, intrinsic-motivation baselines (H4), and critic-based credit-assignment baselines (H2), with the calibration check as the single validity gate. Stretch milestones cover the remaining baselines (H3), MPE environments, the ablation suite, and interpretability heatmaps (H1).

**Honest-science constraint (load-bearing for this plan):** experiment *outcomes* (REAP beating baselines) are hypotheses under test, never acceptance criteria. Acceptance criteria verify that infrastructure is correct and experiments are executed and reported honestly. Wins and losses are both valid, documentable results.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

All numeric thresholds below are **configurable defaults** recorded in config files. Per the convergence deliberation, validity thresholds are gates for *enabling* downstream stages (e.g., turning shaping on), not hard success requirements of the project — a documented threshold violation is a first-class negative result, not a silent failure.

- AC-1: Reproducible research infrastructure exists: a config-driven training entrypoint, deterministic seeding, checkpoint/resume, structured metrics logging (JSON/CSV; extrinsic return and success rate always logged in fields separate from any shaped/intrinsic reward), a pytest suite, pinned dependencies, and a smoke-mode vs. paper-mode config split.
  - Positive Tests (expected to PASS):
    - A fresh environment install from pinned dependencies succeeds and `pytest` passes.
    - Two reduced-config runs with the same seed produce identical metric streams.
    - A run interrupted and resumed from checkpoint continues with consistent step counts and metrics schema.
    - Smoke-mode end-to-end run completes under the configured wall-clock cap.
  - Negative Tests (expected to FAIL/be rejected):
    - Launching with an invalid or incomplete config is rejected with a validation error (no silent defaults for required fields).
    - Resuming from a corrupted checkpoint fails loudly with a diagnostic, never silently restarts.
- AC-2: Overcooked-AI sparse-reward environment harness: `overcooked_ai` consumed as a pinned external dependency, with an in-repo wrapper providing terminal/delivery-only sparse reward, an explicit success predicate defining the goal set `G`, joint-state feature encoding, agent-local observations, and action mapping for the five draft layouts (Cramped Room, Asymmetric Advantages, Coordination Ring, Forced Coordination, Counter Circuit).
  - Positive Tests:
    - Unit tests pass for observation/state shapes, action mapping, reward sparsification, termination handling, and the success predicate on each of the five layouts.
    - A random-policy rollout yields zero extrinsic reward except at delivery events.
  - Negative Tests:
    - Native dense/shaped Overcooked reward leaking into the logged extrinsic return fails a dedicated test.
    - The success predicate firing on a non-success terminal (timeout without delivery) fails a test.
- AC-3: MARL baseline trunk: an in-repo MAPPO implementation (CTDE: centralized critic on joint state, decentralized actors on local observations) plus RND and count-based intrinsic-bonus variants on the same trunk, with a measured benchmark gate and a hardness profile.
  - Positive Tests:
    - MAPPO reaches the configured benchmark gate on sparse Cramped Room (default: success rate ≥ 0.8) within the configured fixed budget and seed set (defaults: 5M env steps, seeds {0,1,2}); the gate is a measured benchmark with exact budget/seeds, not an open-ended training expectation.
    - A hardness-profile report artifact exists for Forced Coordination and Counter Circuit under the same protocol (low/no success expected and documented — this motivates REAP).
    - RND and count-based bonuses each have unit tests and run end-to-end on the trunk.
  - Negative Tests:
    - A test asserting that intrinsic/shaped reward terms never contaminate the logged extrinsic return fails if contamination occurs.
    - Same-seed nondeterminism in the trunk fails the reproducibility test.
- AC-4: PBRS shaping harness with exact semantics: `F_t = β · (γ · Φ(s_{t+1}, k_{t+1}) − Φ(s_t, k_t))` where `k` is the remaining-horizon index, `Φ(terminal, ·) = 0` enforced at both success and timeout termination, and `γ` identical to the MAPPO discount. The potential may take remaining time as input (dynamic PBRS).
  - Positive Tests:
    - Unit tests against hand-computed values on a tiny deterministic chain-MDP test fixture (test-only, not a research environment) cover mid-episode, success-terminal, and timeout-truncation transitions.
    - Invariance sanity check: with a hand-crafted potential on Cramped Room under fixed budget and seeds, shaped and unshaped MAPPO reach overlapping final extrinsic returns (a sanity check, not a statistical claim).
  - Negative Tests:
    - Any code path yielding `Φ(terminal) ≠ 0` fails a test.
    - A shaping discount mismatched with the MAPPO discount is rejected by config validation.
- AC-5: Diffusion trajectory teacher `D`: trained once on a warmup buffer over joint-state trajectories with inpainting-style conditioning (pin `τ[0]`; pin `τ[0]` and `τ[H]`), gated by a warmup-buffer report and a generation-quality report.
  - Positive Tests:
    - Warmup-buffer report artifact contains episode count, success count, state-coverage summary, and success-state examples; success count meets the configured minimum (default: ≥ 25 successful episodes) within the configured collection cap (default: 5M env steps), using the fallback ladder (vanilla MAPPO warmup → MAPPO+RND warmup) if needed.
    - Generation-quality report computes invalid-state rate after projection to the valid state manifold (default gate: ≤ 10%), endpoint success rate, and bridge transition-consistency rate via simulator validation (default gate: ≥ 80%).
  - Negative Tests:
    - Training `D` on a zero-success buffer halts with a diagnostic buffer report; it never silently proceeds.
    - If a validity gate is violated, shaping is auto-disabled for the affected scope and the violation is logged as a first-class result; a fault-injection test verifies this.
- AC-6: Propensity and feasibility estimation: forward policy-conditioned sampling (classifier-free guidance on a behavioral policy embedding — policy action distributions on a fixed probe-state set) yields `propensity(s) ∈ [0,1]`; backward goal-bridges with likelihood weighting and simulator-based transition-consistency filtering yield `feasibility(s)`; direct-query mode on a state subsample works before any distillation (draft Section 5.5 shortcut); distilled predictors `p̂`/`f̂` pass a distillation-fidelity check against direct queries.
  - Positive Tests:
    - Propensity and feasibility unit tests verify output ranges, sample-count plumbing, and gate-only use of feasibility.
    - Direct-query mode produces the feasibility-gated potential on a configured state subsample without requiring `p̂`/`f̂`.
    - Distillation-fidelity report shows `p̂`/`f̂` agreement with direct queries above a configured threshold on held-out states.
  - Negative Tests:
    - Feasibility entering the reward as a magnitude (rather than a gate) fails a test.
    - Propensity values outside [0,1] fail validation.
- AC-7: Calibration gate (the draft's single load-bearing safeguard): held-out calibration check (data disjoint from `D`'s training data) comparing binned predicted propensity vs. realized success rate, with ECE/Brier computed at each refresh (default gate: ECE ≤ 0.15), and an automated response ladder: isotonic recalibration → shrink `β` → set `β = 0` and alert.
  - Positive Tests:
    - Calibration report artifact (binned reliability data + ECE/Brier + configured gate) is produced at each propensity refresh.
    - A fault-injection test with a deliberately miscalibrated `p̂` triggers the ladder in order and ends at `β = 0` with an alert if recalibration fails.
  - Negative Tests:
    - A calibration holdout overlapping `D`'s training data fails a dataset-disjointness test.
    - Miscalibration beyond the gate without the ladder firing fails the test.
- AC-8: Integrated REAP training loop per the draft algorithm (Phase 0: train `D` once; Phase 1: feasibility once, frozen `f̂`; Phase 2: RL loop with propensity refresh): `p̂` refreshed every `K` PPO updates (default `K = 50`), with snapshot consistency — within any transition, `Φ(s)` and `Φ(s′)` come from the same `p̂` snapshot, pinned per rollout batch and never swapped mid-batch; refresh events logged with timestamps and calibration metrics; deployment boundary enforced.
  - Positive Tests:
    - A smoke run of the full integrated loop completes under the wall-clock cap with no NaNs and bounded potential values, and exercises at least one propensity refresh.
    - Deployment-boundary test: the evaluation entrypoint constructs the policy from a checkpoint in a context where teacher/predictor classes are replaced by raising stubs, completes evaluation, and produces actions identical to a reference forward pass; an import-graph check asserts the evaluation module does not import diffusion/shaping modules.
    - Wall-clock and GPU-memory logging make the single-GPU constraint visible in run artifacts.
  - Negative Tests:
    - A mid-batch `p̂` snapshot swap fails the snapshot-consistency test.
    - The evaluation module importing diffusion or shaping modules fails the import-graph check.
- AC-9: Minimal-milestone experiments executed and honestly reported (report-only outcomes; pre-registered report templates so failed hypotheses are cleanly documented).
  - AC-9.1: First integration experiment — REAP-shaped MAPPO vs. vanilla MAPPO vs. MAPPO+RND on Forced Coordination; seeds fixed to {0, 1, 2}; fixed environment-step budget (default: 10M env steps per run); mean ± CI on extrinsic success rate and return.
    - Positive: report artifact exists with the required schema, fixed seeds/budget recorded, and conclusions referencing only extrinsic metrics.
    - Negative: any claim based on shaped return fails the report-audit check; missing seeds or budget mismatches fail.
  - AC-9.2: Baseline-completion experiment (H2/H4) — adds in-repo COMA and QMIX (each validated on MPE Spread against published-result sanity ranges before use in claims), count-based intrinsic baseline, and Counter Circuit; evaluates H2 (vs. COMA/QMIX) and H4 (vs. RND/count-based at equal tuning budget); includes the empirical PBRS invariance check on a layout the unshaped baseline solves.
    - Positive: H2/H4 report artifacts with the same protocol discipline; COMA/QMIX sanity-validation reports exist; invariance-check report shows shaped and unshaped converging to comparable extrinsic returns on the solvable layout.
    - Negative: using COMA/QMIX in claims without passing MPE Spread sanity validation fails the gate; cross-implementation comparisons with mismatched wrappers/observations fail the harness-consistency check.
- AC-10 (stretch, compute-contingent): H3 diffusion-augmentation baseline (MADiff/CODA-style: diffusion without the shaping signal), hand-crafted-potential PBRS baseline, MPE Spread and Tag as research environments, the draft's ablation suite (potential form: propensity vs. feasibility−propensity vs. reach-probability; feasibility gating on/off; policy guidance on/off; horizon `H`; goal-set specification: known terminal vs. learned classifier vs. return threshold; calibration method; frozen vs. periodically-refreshed `D`; frozen-`p̂`-for-entire-run vs. refreshed `p̂`; per-agent `β`), H1 interpretability heatmaps, and optional SMAC/SMACv2.
  - Positive: each executed ablation/baseline produces a report artifact under the same protocol discipline as AC-9.
  - Negative: ablation claims without per-arm config records fail the report audit.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation covers the full draft scope: all five Overcooked-AI layouts plus MPE (Spread, Tag) as research environments; the complete baseline set (vanilla MAPPO, COMA, QMIX, MADiff/CODA-style diffusion augmentation, hand-crafted-potential PBRS, RND, count-based); distilled predictors `p̂`/`f̂` replacing direct queries in the RL loop; the full ablation suite from draft Section 5.4 (including frozen vs. periodically-refreshed `D` for coverage drift, and per-agent `β`); H1 interpretability heatmaps; optional SMAC/SMACv2 — all with full test coverage, pre-registered report templates, and 3+ seed protocols.

### Lower Bound (Minimum Acceptable Scope)

The implementation includes the infrastructure (AC-1), the Overcooked sparse harness (AC-2), the MAPPO trunk with RND (AC-3), the unit-tested PBRS shaping harness with the hand-crafted-potential sanity check (AC-4), the diffusion teacher with warmup-buffer and generation-quality gates (AC-5), direct-query propensity/feasibility with the three calibration guards (AC-6), the calibration gate with the automated ladder (AC-7), the integrated REAP loop with deployment boundary (AC-8), and the first integration experiment on Forced Coordination with seeds {0,1,2} honestly reported (AC-9.1) — i.e., the draft's Section-5.5 shortcut path (direct diffusion queries on a state subsample, before distillation), with COMA/H2 completion (AC-9.2) following as the next increment.

### Allowed Choices

- Can use: PyTorch (fixed); `overcooked_ai` as a pinned pip dependency (fixed); any trajectory-diffusion backbone supporting inpainting-style conditioning (temporal U-Net or transformer denoiser; DDPM/DDIM samplers, or an equivalent conditional trajectory generative model, provided pin-`τ[0]`/pin-`τ[0]`-and-`τ[H]` conditioning is preserved); simulator-based exact transition validation in place of a learned inverse-dynamics model for Overcooked (learned IDM reserved for environments without cheap validity checks); a learned success classifier or return threshold for `G` in environments lacking explicit terminal success (Overcooked uses the explicit delivery predicate); remaining-time input to `Φ` (dynamic PBRS); wandb/tensorboard as optional logging sinks alongside the mandatory JSON/CSV artifacts.
- Cannot use: external MARL frameworks as the execution harness for headline comparisons (all compared algorithms run in-repo on the shared harness — reference implementations such as EPyMARL may be consulted read-only and used for sanity ranges); experiment outcomes as acceptance criteria; shaped or intrinsic return in any claim (extrinsic metrics only); native dense Overcooked reward in primary claims; symbolic-state environments (e.g., Hanabi) for estimator claims (out of scope per draft Section 3).
- Fixed per the draft specification: the potential form `Φ(s) = propensity(s) · 1[feasibility(s) ≥ τ_gate]` as the default REAP configuration (alternative potential forms appear only in the ablation suite); PBRS as the injection mechanism (never replacing the objective); feasibility as a gate only, never a reward magnitude; `f̂` computed once and frozen; `p̂` refreshed every `K` updates against the frozen teacher `D`; teacher and predictors absent at deployment.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

One possible repository layout:

```
reap/
  envs/        # overcooked sparse wrapper, success predicates, (later) MPE; chain-MDP test fixture lives in tests/
  algos/       # mappo trunk (CTDE), rnd, count bonus, (later) coma, qmix
  shaping/     # potential interface, PBRS wrapper with snapshot pinning, hand-crafted potentials
  diffusion/   # trajectory dataset, denoiser, inpainting samplers (forward / bridge), validity projection
  signals/     # propensity (CFG on policy embedding), feasibility (likelihood weighting + consistency filter), distillation
  calibration/ # reliability binning, ECE/Brier, isotonic recalibration, response ladder
  eval/        # deployment-boundary evaluation entrypoint (must not import diffusion/shaping)
configs/       # smoke-mode and paper-mode configs per experiment
scripts/       # warmup collection, teacher training, experiment launchers, report generation
tests/
reports/       # pre-registered templates + generated artifacts
```

Suggested flow mirrors the draft's algorithm: warmup buffer via MAPPO(+RND) → train `D` once (inpainting objective) → feasibility pass once over anchor states (backward bridges, filter, likelihood-weight) → frozen `f̂` → RL loop where every `K` updates forward samples from frozen `D` (CFG on the behavioral policy embedding) refresh propensity → `p̂` → shape transitions with the snapshot-pinned potential → MAPPO update on `r′` → deploy `π_θ` alone. For the first integration experiment, skip distillation and query `D` directly on a state subsample (draft Section 5.5).

A behavioral policy embedding can be built by evaluating the current actors on a fixed probe-state set sampled once from the warmup buffer and concatenating (possibly projected) action distributions — cheap, policy-faithful, and refreshable without touching `D`.

### Relevant References

- `overcooked_ai` (Carroll et al. 2019) — environments, layouts, joint-state featurization, delivery events
- MAPPO (Yu et al. 2021/2022) — reference hyperparameters and CTDE conventions for Overcooked/MPE
- Diffuser (Janner et al. 2022), Decision Diffuser (Ajay et al. 2022) — trajectory diffusion + inpainting/goal conditioning
- Recall Traces (Goyal et al. 2018), ROMI (2021), BIFRL (2022) — backward generation from goal states
- MADiff (2024), CODA (2026) — diffusion in MARL; H3 baseline shape
- Ng et al. 1999 — PBRS invariance; Devlin & Kudenko 2012 — dynamic (time-varying) PBRS
- EPyMARL — read-only reference for COMA/QMIX correctness and published sanity ranges
- Burda et al. 2018 (RND); count-based exploration baselines

## Dependencies and Sequence

### Milestones

1. M1 — Research infrastructure: project scaffold, config system, seeding, checkpointing, metrics schema (extrinsic strictly separated), pytest, smoke/paper config split. (AC-1)
2. M2 — Environment and baseline trunk: Overcooked sparse wrapper + success predicates; in-repo MAPPO; Cramped Room benchmark gate; Forced Coordination/Counter Circuit hardness profile; RND + count-based bonuses. (AC-2, AC-3)
3. M3 — PBRS shaping harness: exact shaping semantics, chain-MDP fixture tests, hand-crafted-potential invariance sanity check on Cramped Room. (AC-4) — depends on M1; the invariance check depends on M2.
4. M4 — Diffusion teacher: warmup buffer + report + gates; trajectory diffusion with inpainting; generation-quality report. (AC-5) — depends on M2.
5. M5 — REAP signals and calibration: policy embedding + forward propensity; backward bridges + feasibility with the three guards; direct-query mode; calibration module + ladder; distillation `p̂`/`f̂` + fidelity check. (AC-6, AC-7) — depends on M4.
6. M6a — First integration: integrated REAP loop (snapshot pinning, refresh logging, deployment boundary) + Forced Coordination experiment vs. vanilla MAPPO and MAPPO+RND, seeds {0,1,2}, fixed budget, report. (AC-8, AC-9.1) — depends on M3, M5 (direct-query mode sufficient; distillation not required for M6a).
7. M6b — Baseline completion (H2/H4): in-repo COMA + QMIX with MPE Spread sanity validation; count-based comparison; Counter Circuit; invariance check on a solvable layout; H2/H4 reports. (AC-9.2) — depends on M6a; distilled `p̂`/`f̂` recommended for paper-mode budgets.
8. M7 — Stretch: H3 diffusion-augmentation baseline, hand-PBRS comparison, MPE research environments, ablation suite, H1 heatmaps, optional SMAC. (AC-10) — depends on M6b.

Dependency summary: M1 → M2 → {M3 (sanity part), M4} → M5 → M6a → M6b → M7. M3's unit-test portion depends only on M1.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Project scaffold: package layout, pinned deps, config system with validation, seeding, checkpoint/resume, metrics schema (extrinsic-separated), pytest harness, smoke/paper config split | AC-1 | coding | - |
| task2 | Overcooked sparse wrapper: five layouts, terminal-only reward, success predicate `G`, joint-state encoding, local observations, action mapping, unit tests | AC-2 | coding | task1 |
| task3 | In-repo MAPPO trunk (CTDE) + Cramped Room benchmark gate run (fixed budget/seeds) | AC-3 | coding | task2 |
| task4 | RND and count-based intrinsic bonuses on the MAPPO trunk, with unit tests | AC-3 | coding | task3 |
| task5 | Hardness-profile runs and report for Forced Coordination + Counter Circuit | AC-3 | coding | task3 |
| task6 | PBRS shaping module with exact semantics, chain-MDP test fixture, terminal/timeout unit tests | AC-4 | coding | task1 |
| task7 | Hand-crafted-potential invariance sanity check on Cramped Room | AC-4 | coding | task3, task6 |
| task8 | Warmup buffer collection + buffer report + success-count gate + fallback ladder | AC-5 | coding | task3, task4 |
| task9 | Trajectory diffusion teacher `D` with inpainting conditioning + validity projection + generation-quality report | AC-5 | coding | task8 |
| task10 | Behavioral policy embedding + forward propensity sampling (CFG), direct-query mode | AC-6 | coding | task9 |
| task11 | Backward goal-bridges + feasibility with likelihood weighting and simulator consistency filter, gate-only use | AC-6 | coding | task9 |
| task12 | Calibration module: held-out disjoint check, reliability bins, ECE/Brier, automated ladder, fault-injection tests | AC-7 | coding | task10 |
| task13 | Distillation of `p̂`/`f̂` + distillation-fidelity check | AC-6 | coding | task10, task11 |
| task14 | Integrated REAP loop: snapshot-pinned shaping, `K`-periodic refresh with logging, deployment-boundary evaluation entrypoint + import-graph test | AC-8 | coding | task6, task10, task11, task12 |
| task15 | First integration experiment on Forced Coordination (REAP vs. MAPPO vs. MAPPO+RND, seeds {0,1,2}, fixed budget) + pre-registered report | AC-9.1 | coding | task14 |
| task16 | External audit of the first-integration report and calibration evidence (extrinsic-only claims, protocol fidelity) | AC-9.1 | analyze | task15 |
| task17 | In-repo COMA and QMIX on the shared harness + minimal MPE Spread harness + published-sanity-range validation reports | AC-9.2 | coding | task3 |
| task18 | Baseline-completion experiments: H2/H4 on Forced Coordination + Counter Circuit, count-based arm, invariance check on solvable layout, reports | AC-9.2 | coding | task13, task15, task17 |
| task19 | External audit of H2/H4 conclusions and harness-consistency (same wrappers/observations across compared algorithms) | AC-9.2 | analyze | task18 |
| task20 | Stretch: H3 diffusion-augmentation baseline, hand-PBRS arm, MPE research runs, ablation suite, H1 heatmaps, optional SMAC | AC-10 | coding | task18 |

## Claude-Codex Deliberation

Deliberation ran as: Codex first-pass analysis of the raw draft → Claude candidate plan v1 → two convergence rounds with a second Codex reviewer. Round 2 returned no disagreements, no required changes, and no unresolved items.

### Codex First-Pass Findings (incorporated)

- The draft's minimal milestone is too large as a single step for an empty repository → staged milestones M1–M6a with the draft's Section-5.5 scope completed across M6a + M6b.
- Diffusion over structured Overcooked states risks invalid generations → validity projection, invalid-state-rate gate, simulator-based transition-consistency gate (AC-5).
- Warmup buffers on hard layouts may contain zero successes → success-count gate, collection cap, fallback ladder, halt-with-diagnostic (AC-5).
- PBRS guarantee is fragile under learned/refreshed/gated potentials → exact shaping semantics with `Φ(terminal)=0` (AC-4), snapshot pinning (AC-8), claim-scope note below.
- Raw-policy-weight embeddings are unrealistic → behavioral probe-state embedding (AC-6).
- Finite-horizon tasks likely need time in the potential → remaining-horizon index `k` permitted (dynamic PBRS, AC-4).
- Milestone-1 acceptance must be pipeline correctness + calibration signal, not "beats all baselines" → the honest-science constraint in the Goal Description.

### Agreements

- Staging order infrastructure → env/MAPPO → PBRS harness → diffusion → REAP is the correct dependency order.
- Experiment wins are hypotheses, never acceptance criteria; extrinsic metrics strictly separated from shaped/intrinsic.
- Overcooked Cramped Room serves for pipeline validation (no custom research gridworld); a tiny chain MDP exists only as a unit-test fixture.
- Behavioral probe-state policy embeddings; simulator-based transition validation in place of a learned inverse-dynamics model for Overcooked.
- Warmup gates, generation-validity gates, calibration ECE gate, and the automated `β`-disable ladder as specified in AC-5/AC-7.
- Centralized joint state for training-time shaping is CTDE-consistent; actors stay decentralized; teacher and predictors are absent at deployment (enforced by AC-8 tests).

### Resolved Disagreements

- **Frozen vs. refreshed propensity during training**: Codex initially required a frozen potential per run (non-stationarity risk to the PBRS guarantee), with refresh only as an ablation. Claude pushed back: `K`-periodic propensity refresh is a stated core design element of the draft ("on-policy where it matters") and must not be silently overridden. Resolution (accepted by Codex in Round 2): the draft's refresh design stays the default, with safeguards — per-rollout-batch `p̂` snapshot pinning so `Φ(s)` and `Φ(s′)` never mix snapshots, refresh/calibration event logging, the automated calibration ladder, a frozen-`p̂` arm in the ablation suite, and a conservative (long) default refresh period for first-integration runs.
- **COMA in the first integrated experiment**: Codex flagged the draft's minimal milestone as too large to be the first end-to-end run. Resolution: split into M6a (REAP vs. MAPPO vs. RND on Forced Coordination) and M6b (COMA/QMIX, Counter Circuit, H2/H4 completion). The draft's Section-5.5 scope is fully preserved across the two milestones, only sequenced.
- **First hard layout**: Forced Coordination first (most handoff-bottlenecked, canonical coordination challenge); Counter Circuit added in M6b.
- **COMA/QMIX implementation source**: in-repo on the shared harness (same wrappers, observations, logging, eval protocol — Codex's own cross-framework-comparability concern was the deciding argument), with correctness validated against published MPE Spread sanity ranges before use in H2 claims. External frameworks remain read-only references.
- **Invariance check strength**: "statistically indistinguishable" weakened to an explicit sanity check with fixed budget/seeds (deep-RL variance makes a statistical-equivalence claim from few seeds unsupportable); the load-bearing PBRS correctness evidence is the unit-tested shaping math plus the empirical invariance check of draft Section 5.2.

### Claim-Scope Note (adopted from Codex Round 2)

Refreshed-`p̂` REAP is presented as **dynamic/on-policy shaping** (Devlin & Kudenko 2012), not as relying on the classic stationary-PBRS invariance guarantee, unless the dynamic-PBRS assumptions are met and tested. Paper text and report templates must reflect this scoping.

### Convergence Status

- Final Status: `converged`
- Rounds executed: Codex first-pass + 2 convergence rounds (Round 2: no DISAGREE, no REQUIRED_CHANGES, no UNRESOLVED)

## Pending User Decisions

None. All nine open questions from the Codex first-pass (publishable fidelity vs. proof-of-concept; vendoring vs. in-repo; compute budgeting; first environment; COMA/QMIX timing; centralized-state shaping; time-dependent potential; warmup success minimum; calibration-failure response) were substantively resolved during the convergence loop with explicit Codex agreement, as recorded in the Deliberation section above. All numeric values introduced by this plan (warmup minimum 25 successes, 5M-step warmup cap, ≤10% invalid-state rate, ≥80% bridge consistency, ECE ≤ 0.15, `K = 50`, Cramped Room gate 0.8 within 5M steps, 10M-step experiment budget, seeds {0,1,2}) are configurable defaults chosen by the deliberation, not user-stated hard requirements; the user may adjust any of them in config without amending this plan.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Additional Notes
- All numeric gates and budgets live in config files with the defaults stated in this plan; changing them is a config edit, not a plan change.
- Every hypothesis test (H2, H3, H4, invariance) uses a pre-registered report template committed before the experiment runs, so negative results are documented with the same rigor as positive ones.
- Wall-clock and GPU-memory usage are logged in all training runs to keep the single-GPU constraint observable.
- The evaluation/deployment code path must remain structurally incapable of touching the diffusion teacher, predictors, or shaped rewards (raising-stub test + import-graph check).

--- Original Design Draft Start ---

# REAP: Reachability-Estimated Adaptive Potentials for Sparse-Reward Cooperative MARL

**Author:** Shirley Huang
**Affiliation:** Harvard Business School (Technology & Operations Management)
**Status:** Draft v0.1
**Date:** June 3, 2026

---

## Abstract

Long-horizon reinforcement learning collapses a single terminal reward onto a long sequence of decisions, leaving almost no learning signal at the intermediate states where the decisive choices are made. Cooperative multi-agent RL (MARL) with sparse terminal reward is the sharpest instance of this problem: the effective horizon is multiplied by the number of agents, and critic-based credit assignment (COMA, value decomposition) is trained on near-zero reward at exactly the rare, success-adjacent joint states that matter.

We propose **REAP** (Reachability-Estimated Adaptive Potentials), a training-time framework that manufactures a dense, per-timestep learning signal from a diffusion trajectory model and injects it as a **potential-based shaping reward** — densifying credit assignment without moving the optimal policy. At each state REAP estimates two quantities: **propensity**, how likely the *current* joint policy is to reach success from here (a policy-conditioned, on-policy success-value), and **feasibility**, whether a realizable success path exists at all (a policy-independent reachability gate). The feasibility-gated propensity defines a potential `Φ`; the shaping term `β(γΦ(s′) − Φ(s))` rewards progress toward reachable success and is silent in unrecoverable regions.

REAP is, in effect, **a reward-shaping analogue of on-policy distillation** — a frozen generative teacher grades the student's own on-policy trajectories with a dense per-step signal — but, unlike on-policy distillation, the signal enters as a potential and therefore carries a *provable optimality guarantee* (Ng et al. 1999) that on-policy distillation lacks. The diffusion teacher and its distilled predictors are discarded at deployment; the shipped agent is an ordinary MARL policy that converged faster because it learned against a denser reward. We claim **usefulness** (faster convergence and higher final return on sparse-reward cooperative tasks), not identification of any latent quantity, and we evaluate it as a learning signal. The central structural claim: **the advantage grows with horizon, agent count, and reward sparsity** — precisely the regime where every other credit-assignment method degrades.

---

## 1. Motivation

### 1.1 The problem is bigger than MARL

The defining bottleneck of modern long-horizon RL is credit assignment under sparse terminal reward. It is the same wall that agentic LLM RL (RLVR) is hitting today: one verifiable reward at the end of a long trajectory, and effectively no gradient information about which of the many intermediate decisions earned it. Methods that work when reward is dense fail when the signal arrives once, late, and far from the actions that caused it.

Cooperative MARL with sparse terminal reward is the cleanest, cheapest laboratory for this problem. The horizon is effectively multiplied by the number of agents (credit must be split across *joint* actions), the reward is a single delivery/win at episode end, and the success-adjacent states are rare. A method that densifies credit assignment here — without biasing the solution — is a candidate mechanism for the general long-horizon problem.

### 1.2 Why critic-based credit assignment is weakest exactly where it matters

COMA and value-decomposition critics are trained on observed reward. Under sparse terminal reward, the rare states that sit one or two coordinated joint actions away from success carry almost no reward signal during training, so the critic's estimate there is high-variance and slow to form — yet these are the states whose value most determines whether the agents ever succeed. The critic is least informed exactly where information is most decisive.

### 1.3 What REAP adds

A generative trajectory model can *imagine* what the observed reward never reveals: from this state, conditioned on how the agents currently behave, how often does a rollout reach success (propensity), and does any realizable success path exist at all (feasibility). This is a dense, model-derived estimate available at every timestep, including the rare success-adjacent states the critic starves on. We convert it into a reward bonus that is dense where the critic is sparse — and, by construction, leaves the optimum untouched.

---

## 2. Why reward shaping is the right target (not measurement)

REAP's signal is **confounded by design**, and that is acceptable because of how it is used. An earlier framing of this idea asserted that the propensity/feasibility gap *equals* miscoordination. That is false: the gap is a sum of miscoordination, individual-skill suboptimality, the price of decentralization, equilibrium-mode mismatch, exploration entropy, and selection bias. Identifying any one component requires separating all of them and a ground-truth coordination label we do not have.

We avoid the entire problem by never claiming the signal *measures* anything. The license is **potential-based reward shaping (PBRS)**: adding `F(s, s′) = γΦ(s′) − Φ(s)` to the reward leaves the optimal policy unchanged for *any* potential `Φ` (Ng et al. 1999). A confounded potential changes the optimization *path*, not the fixed point. The signal need not be a pure measurement; it needs to be a useful path-shaper that does not move the solution. Under this lens the confounders are features, not bugs — e.g., shaping toward feasible-but-unrealized success pushes a weak agent to improve, which is additional signal, not contamination.

**Exactly one property is load-bearing: calibration.** An over-optimistic feasibility estimate can shape toward states that are not reachable in expectation. This is the success-conditioned selection effect: conditioning a backward bridge on *reaching* the goal selects favorable environment noise and overstates what is achievable. A miscalibrated potential still preserves the optimum in theory but wastes the optimization path in practice and can stall learning. Calibration (Section 4.4 and Section 6) is therefore the single safeguard the method actually owes.

---

## 3. Positioning

REAP is a credit-assignment-and-exploration contribution. Two framings carry it; a third is intuition.

**(Lead 1) A reward-shaping analogue of on-policy distillation.** On-policy distillation grades a student's *own* sampled trajectories with a dense per-step teacher signal, fixing the train/test mismatch of off-policy distillation. REAP maps onto this almost exactly — the MAPPO student samples on-policy rollouts; the frozen diffusion teacher `D` scores each visited state by reachability-to-success, conditioned on the current policy. The differentiator is decisive: on-policy distillation changes the objective to match the teacher (no invariance guarantee), whereas REAP injects the distilled signal as a *potential* and provably preserves the optimum. **REAP offers the density of on-policy distillation with an optimality guarantee on-policy distillation lacks.** ("On-policy" here refers to the data distribution the teacher is evaluated on, not the teacher's weights — which is why a frozen teacher is consistent with the framing.)

**(Lead 2) Model-based value shaping.** `propensity(s) = P(reach G | s, π_θ)` is exactly an on-policy success-probability value function. REAP uses a policy-conditioned generative model as a model-based estimator of that value, distills it into a cheap predictor, and injects it as a PBRS potential. This places the work at *model-based value estimation × reward shaping*, which is the reviewer-safe, MARL-native description of the contribution.

**(Intuition) Generative self-imitation.** Rather than waiting for real successful episodes to reinforce (self-imitation learning), REAP's teacher *imagines* feasible successful continuations and shapes toward them.

**Adjacent work, and why REAP is distinct.**
- *Generic intrinsic motivation (RND, count-based).* The live objection is "novelty-seeking with extra steps." Beating these is mandatory (H4): reachability structure buys *directed* pressure toward task completion that undirected novelty does not.
- *State-marginal-matching imitation (GAIfO, SMODICE).* These minimize divergence to a fixed expert distribution as the **objective**. REAP is a per-state potential used as **shaping**, not as the objective, and is derived per-goal from a model rather than from expert data.
- *COMA / value decomposition.* Baselines to beat on sparse-reward coordination, not a paradigm we replace. Expected advantage: density where the critic has near-zero signal.
- *Mechanism prior art (reused, not claimed).* Goal-conditioning via inpainting (Diffuser, Janner 2022; Decision Diffuser, Ajay 2022); backward generation from goal states (Recall Traces, Goyal 2018; ROMI 2021; BIFRL 2022); diffusion in MARL (MADiff 2024; CODA 2026). REAP reuses these as subroutines; the contribution is the **feasibility-gated propensity potential as a shaping signal**, not the generation machinery.

**Scope.** Diffusion state modeling earns its cost only for continuous or perceptual state. Primary claims are restricted to continuous-feature environments; symbolic-state settings (e.g., Hanabi) are out of scope for the estimator.

---

## 4. Method: the REAP framework

**Setting.** Dec-POMDP, `N` agents, joint state `s`, joint policy `π_θ`, horizon `H`, cooperative sparse terminal reward `r`. Let `G` be a success set (Overcooked: terminal states with a delivered soup; generally a return threshold or a learned success classifier).

REAP has one generative teacher and two distilled predictors. The teacher is trained once; the expensive computation is done once; only two tiny MLPs ever touch the RL loop. **Everything is training-time scaffolding — none of it exists at deployment.** This is what makes REAP single-GPU viable.

### 4.1 The two scalars

Train one diffusion trajectory model `D` over joint-state trajectories `τ = (s_0, …, s_H)` and query it by inpainting in two modes:

- **Forward (policy-conditioned):** pin `τ[0] = s_t`, condition on the current policy via classifier-free guidance on a policy embedding, sample continuations.
  `propensity(s_t)` = fraction of forward samples whose endpoint lands in `G`. *(On-policy success-value of π_θ.)*
- **Backward (goal-conditioned):** pin `τ[0] = s_t` and `τ[H] = g` for `g ∈ G`, denoise the bridge.
  `feasibility(s_t)` = validity-filtered, likelihood-weighted bridge quality. *(Policy-independent reachability.)*

### 4.2 The potential and the shaping term

```
Φ(s_t) = propensity(s_t) · 1[ feasibility(s_t) ≥ τ_gate ]
r′_t   = r_t + β · ( γ · Φ(s_{t+1}) − Φ(s_t) )
```

Propensity is the value being climbed; feasibility enters **only as a gate**, not as reward magnitude. Shaping is applied where feasibility is high (success is plausibly reachable, so climbing propensity is meaningful) and suppressed where feasibility is near zero (hard or lost states, where there is nothing to climb toward). This keeps the dense signal from rewarding motion in genuinely unrecoverable regions — without any claim about *why* those regions are unrecoverable.

### 4.3 Why the design is compute-frugal (and on-policy where it matters)

The two halves have different costs and different refresh needs, and REAP exploits this:

- **Feasibility is policy-independent → computed once, frozen.** It is a property of environment dynamics and goal set, not of π. The expensive, calibration-risky backward sampling runs a single time, offline, and is distilled into a frozen predictor `f̂`.
- **Propensity is the only policy-tracking quantity → refreshed cheaply.** It is re-estimated by querying the *same frozen* `D` with the updated policy embedding (no retraining of `D`) and distilled into `p̂`, refreshed every `K` updates.

This is on-policy in the sense that matters: the signal is computed on states the current policy visits and conditioned on the current policy. A frozen teacher `D` is consistent with the on-policy-distillation contract (the teacher is fixed; the evaluation distribution is on-policy). The only residual risk of a frozen `D` is **coverage drift** as π explores states `D` never saw; mitigated by training `D` on a diverse warmup/offline buffer, with periodic `D`-refresh available as an ablation (Section 6) for budgets that can afford it.

### 4.4 Calibration

Backward, success-conditioned sampling overstates feasibility (selection effect). Three guards, applied once at distillation time:
1. **Likelihood-weight** backward samples rather than counting raw success-conditioned draws.
2. **Inverse-dynamics consistency filter:** reject bridges whose state transitions are not realizable under a learned inverse-dynamics model.
3. **Sign-level use only:** feasibility enters as a gate (a path plausibly exists), never as a reward magnitude.
A held-out **calibration check** (predicted propensity vs. realized success rate) validates `p̂` at each refresh; if miscalibrated, isotonic-recalibrate or shrink `β`.

### 4.5 Algorithm

```
REAP — training-time only. D, f̂, p̂ are discarded at deployment.

Inputs: env; MAPPO policy π_θ; success set G; shaping weight β;
        gate threshold τ_gate; refresh period K; sample counts.

Phase 0 — Train the teacher (ONCE)
  Collect/Load buffer B of joint-state trajectories (MAPPO warmup or offline data).
  Train diffusion trajectory model D on B via the inpainting objective.

Phase 1 — Feasibility: policy-INDEPENDENT, computed ONCE (the expensive step)
  for sampled anchor states s_i in B:
     draw M backward goal-bridges: pin τ[0]=s_i, τ[H]=g∈G; denoise
     filter bridges by inverse-dynamics consistency; likelihood-weight
     feasibility(s_i) ← calibrated bridge quality
  distill frozen predictor f̂ on {(s_i, feasibility(s_i))}

Phase 2 — RL loop with REFRESHED propensity (cheap, every K updates)
  for each PPO update t = 1, 2, ...:
     if t mod K == 1:                                  # refresh propensity
        for sampled on-policy states s_j:
           draw M forward rollouts from FROZEN D, CFG on emb(π_θ)
           propensity(s_j) ← fraction landing in G
        distill/refresh p̂ on {(s_j, propensity(s_j))}
        calibration check: predicted propensity vs realized success (held-out)
           → if off: isotonic-recalibrate p̂ or shrink β
     collect on-policy rollouts with π_θ
     for each transition (s, a, r, s′):
        Φ(s)  ← p̂(s)  · 1[ f̂(s)  ≥ τ_gate ]
        Φ(s′) ← p̂(s′) · 1[ f̂(s′) ≥ τ_gate ]
        r′ ← r + β · (γ · Φ(s′) − Φ(s))               # PBRS: optimum preserved
     MAPPO update on shaped rewards r′

  return π_θ        # deployed alone — no D, f̂, p̂ at test time
```

Recurring in-loop cost is two MLP forward passes (`f̂` frozen, `p̂` refreshed on a schedule). One diffusion model trained once; backward sampling done once.

---

## 5. Evaluation

### 5.1 Tasks (compute-scaled)

**Primary (single-GPU):**
- **Overcooked-AI** (Carroll et al. 2019): Cramped Room, Asymmetric Advantages (easy); Coordination Ring (medium); Forced Coordination, Counter Circuit (hard, handoff-bottlenecked). Sparse delivery reward, clear goal set, continuous feature encodings, interpretable spatial maps, existing MAPPO baseline.
- **MPE** (Lowe et al. 2017): Cooperative Navigation (Spread) and Predator-Prey (Tag). Continuous state, cheap for ablations; Tag instantiates catch-needs-two structure.

**Optional stretch (compute-contingent):**
- **SMAC / SMACv2** (Samvelyan et al. 2019): home turf of COMA and value decomposition; included only if compute allows, to strengthen the competitive story. Not load-bearing.

### 5.2 Hypotheses

- **H2 — Beats critic-based credit assignment (load-bearing).** REAP improves convergence speed and final return over COMA and QMIX on the hard, sparse settings, **with the margin widening as reward sparsity, horizon, and agent count increase.**
- **H3 — It is the signal, not the diffusion (load-bearing).** Diffusion augmentation *without* the shaping signal (MADiff/CODA-style) does not match REAP. Isolates the contribution from "diffusion helps MARL."
- **H4 — Beats generic novelty (mandatory).** REAP beats RND and count-based intrinsic reward at equal tuning budget. If H4 fails, the construction is not justified.
- **H1 — Interpretability (optional, not load-bearing).** High-signal states coincide with human-identifiable coordination bottlenecks on Overcooked layouts. A heatmap, not a validity claim.
- **Invariance check.** On a task the unshaped baseline can solve, shaped and unshaped policies converge to the same returns — confirming REAP accelerates rather than distorts (the PBRS guarantee, verified empirically).

### 5.3 Baselines

Vanilla MAPPO; COMA; QMIX; MADiff or CODA (diffusion without the signal); PBRS with a hand-crafted potential; RND and count-based intrinsic motivation.

### 5.4 Ablations

Potential form (propensity vs. feasibility−propensity vs. reach-probability); feasibility gating on/off; forward conditioning with/without policy guidance; horizon `H`; goal-set specification (known terminal vs. learned classifier vs. return threshold); calibration method; **frozen vs. periodically-refreshed `D` (coverage drift)**; per-agent `β` split by marginal effect on propensity (a shaping convenience, single line — not a coordination claim).

### 5.5 Minimal first milestone (ship this first)

On Overcooked **Forced Coordination** and **Counter Circuit** with the existing MAPPO baseline: train `D` on the replay buffer, compute the feasibility-gated propensity potential (direct diffusion queries on a state subsample — the Phase-1/Phase-2 shortcut — before bothering to distill), run shaped MAPPO, and test **H2 vs. COMA** and **H4 vs. RND**. The calibration check (predicted vs. realized success) is the single validity gate. This is the smallest experiment that can support or kill the usefulness claim; it requires no decomposition, no coordination labels, no per-agent attribution.

---

## 6. Risks and mitigations

- **Calibration (the only load-bearing risk).** Success-conditioned backward sampling overstates feasibility. Mitigations: likelihood-weighting, inverse-dynamics consistency filter, sign-level (gate-only) use. Validate with the held-out calibration check.
- **Coverage drift of frozen `D`.** As π improves it visits states `D` never saw. Mitigate by training `D` on a diverse buffer; periodic `D`-refresh available as an ablation if budget allows.
- **Cost.** Per-state backward sampling is expensive — which is why it is done **once**, offline, and distilled. Use a state subsample for the milestone.
- **Goal-set specification.** Requires a defined `G`; in open-ended tasks use a learned success classifier or return threshold; ablate sensitivity.
- **Modality.** Claims restricted to continuous / perceptual state.

---

## 7. Contributions

1. **REAP**, a training-time framework that converts a policy-conditioned generative reachability estimate into a potential-based shaping reward, densifying credit assignment in sparse-reward cooperative MARL while provably preserving the optimal policy.
2. A **frugal, on-policy-where-it-matters algorithm**: feasibility computed once (policy-independent, frozen), propensity refreshed cheaply against a frozen diffusion teacher — single-GPU viable.
3. A positioning that unifies **on-policy distillation** (density on the student's own trajectories) with **model-based value shaping** (PBRS invariance), yielding "on-policy-distillation density with an optimality guarantee it lacks."
4. A **structural claim with empirical test**: the advantage of dense, unbiased shaping grows with horizon, agent count, and reward sparsity — the regime where critic-based and novelty-based methods degrade.

--- Original Design Draft End ---
