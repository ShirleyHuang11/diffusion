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
