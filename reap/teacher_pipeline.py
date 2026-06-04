"""End-to-end teacher pipeline on Overcooked: warmup -> teacher -> signals.

Produces the durable artifacts the framework owes before any shaped training:
warmup-buffer report (success-gated ladder), generation-quality report,
held-out calibration report, and distillation-fidelity reports.

Delivery events are not statically readable from the lossless grid encoding,
so the modeled joint state is augmented with the (scaled) cumulative-delivery
count; goal-set membership of a sampled window then reduces to comparing the
endpoint's count feature against the pinned start's.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from reap.algos.mappo import MappoNets
from reap.calibration import CalibrationLadder, ensure_disjoint
from reap.checkpoint import latest_checkpoint, load_checkpoint
from reap.data.buffer import TrajectoryBuffer
from reap.data.warmup import collect_warmup
from reap.diffusion import GaussianDiffusion, TrajectoryDenoiser, TrajectoryWindowDataset
from reap.diffusion.ddpm import train_teacher
from reap.diffusion.quality import StateValidator, generation_quality_report
from reap.envs.base import CoopEnv, StepResult
from reap.envs.overcooked_env import OvercookedSparseEnv
from reap.signals import (
    BehavioralPolicyEmbedding,
    TransitionChecker,
    collect_probe_observations,
    estimate_feasibility,
    estimate_propensity,
)
from reap.signals.distill import DistilledPredictor, distillation_fidelity_report

DELIVERY_SCALE = 0.1  # cumulative deliveries are stored as count * scale


class DeliveryAugmentedOvercooked(CoopEnv):
    """Appends the scaled cumulative-delivery count to the joint state."""

    def __init__(self, base: OvercookedSparseEnv):
        self.base = base
        self.num_agents = base.num_agents
        self.num_actions = base.num_actions
        self.horizon = base.horizon
        self.local_obs_dim = base.local_obs_dim
        self.joint_state_dim = base.joint_state_dim + 1

    def _augment(self, joint: np.ndarray) -> np.ndarray:
        count = self.base._deliveries * DELIVERY_SCALE
        return np.concatenate([joint, np.array([count], dtype=np.float32)])

    def reset(self, seed: int | None = None):
        local, joint = self.base.reset(seed)
        return local, self._augment(joint)

    def step(self, actions) -> StepResult:
        result = self.base.step(actions)
        return StepResult(
            local_obs=result.local_obs,
            joint_state=self._augment(result.joint_state),
            extrinsic_reward=result.extrinsic_reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
        )

    def is_success(self) -> bool:
        return self.base.is_success()

    @property
    def steps_elapsed(self) -> int:
        return self.base.steps_elapsed

    def get_state(self) -> dict:
        return self.base.get_state()

    def set_state(self, snapshot: dict):
        local, joint = self.base.set_state(snapshot)
        return local, self._augment(joint)


def window_success(endpoint: np.ndarray, start: np.ndarray) -> bool:
    """A window succeeds when at least one delivery happens inside it."""
    return bool(endpoint[-1] - start[-1] >= 0.5 * DELIVERY_SCALE)


class OvercookedExactValidator(StateValidator):
    """Exact validity for augmented lossless-grid states.

    ``project`` rounds the grid part to integers (clipped to training-data
    channel ranges) and the delivery feature to the nearest count multiple.
    ``is_valid`` is exact by construction: the grid must decode to a native
    OvercookedState whose re-encoding reproduces the grid (urgency layer
    excluded — it only reflects the unobserved timestep).
    """

    method = "simulator-decode roundtrip (decode_lossless -> re-encode identity)"

    def __init__(self, training_states: np.ndarray, grid_shape, mdp):
        self.grid_shape = tuple(grid_shape)
        self.grid_size = int(np.prod(grid_shape))
        self.mdp = mdp
        flat = np.asarray(training_states, dtype=np.float64)
        self.low = flat.min(axis=0)
        self.high = flat.max(axis=0)

    def project(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float64)
        projected = states.copy()
        projected[:, : self.grid_size] = np.round(projected[:, : self.grid_size])
        projected[:, self.grid_size :] = (
            np.round(projected[:, self.grid_size :] / DELIVERY_SCALE) * DELIVERY_SCALE
        )
        return np.clip(projected, self.low, self.high)

    def is_valid(self, states: np.ndarray) -> np.ndarray:
        from reap.envs.overcooked_decode import roundtrip_valid

        states = np.asarray(states, dtype=np.float64)
        ok = np.empty(len(states), dtype=bool)
        for i, flat in enumerate(states):
            grid = flat[: self.grid_size].reshape(*self.grid_shape)
            count_ok = (
                abs(flat[-1] / DELIVERY_SCALE - round(flat[-1] / DELIVERY_SCALE)) < 1e-6
                and flat[-1] >= -1e-6
            )
            ok[i] = count_ok and roundtrip_valid(grid, self.mdp)
        return ok


class OvercookedSimulatorChecker(TransitionChecker):
    """Exact transition validation through the native simulator.

    Both states are decoded back to OvercookedState; the transition is
    realizable iff some joint action maps the first to the second under
    ``mdp.get_state_transition`` (urgency layer excluded) AND the augmented
    delivery feature changes exactly when that action delivers.
    """

    method = "simulator search over all joint actions on decoded states"

    def __init__(self, mdp, grid_shape):
        import itertools

        from overcooked_ai_py.mdp.actions import Action

        self.mdp = mdp
        self.grid_shape = tuple(grid_shape)
        self.grid_size = int(np.prod(grid_shape))
        self.joint_actions = list(itertools.product(Action.ALL_ACTIONS, repeat=2))
        self._decode_cache: dict[bytes, object] = {}

    def _decode(self, flat: np.ndarray):
        """Decode ONLY fully valid states: nonnegative delivery count on the
        scale grid, integer-valued grid, and exact decode->re-encode
        roundtrip. Anything else is unusable for transition replay."""
        from reap.envs.overcooked_decode import decode_if_roundtrip

        flat = np.asarray(flat, dtype=np.float64)
        key = flat.tobytes()
        if key not in self._decode_cache:
            native = None
            count = flat[-1]
            count_ok = (
                count >= -1e-9
                and abs(count / DELIVERY_SCALE - round(count / DELIVERY_SCALE)) < 1e-6
            )
            grid = flat[: self.grid_size]
            if count_ok and np.all(grid == np.round(grid)):
                native = decode_if_roundtrip(
                    grid.astype(int).reshape(*self.grid_shape), self.mdp
                )
            self._decode_cache[key] = native
        return self._decode_cache[key]

    def realizable(self, state: np.ndarray, next_state: np.ndarray) -> bool:
        from reap.envs.overcooked_decode import COMPARE_LAYERS, encode_state

        state = np.asarray(state, dtype=np.float64)
        next_state = np.asarray(next_state, dtype=np.float64)
        native = self._decode(state)
        if native is None or self._decode(next_state) is None:
            return False
        target = (
            next_state[: self.grid_size]
            .reshape(*self.grid_shape)
            .astype(int)[..., COMPARE_LAYERS]
        )
        delta_count = next_state[-1] - state[-1]
        for joint_action in self.joint_actions:
            successor, infos = self.mdp.get_state_transition(native, joint_action)
            sparse = sum(infos["sparse_reward_by_agent"])
            expected_delta = DELIVERY_SCALE * (sparse / 20.0) if sparse > 0 else 0.0
            if abs(delta_count - expected_delta) > 1e-6:
                continue
            if np.array_equal(encode_state(successor, self.mdp)[..., COMPARE_LAYERS], target):
                return True
        return False


def load_policy_nets(run_dir: str | Path, env: CoopEnv, hidden: int = 128) -> MappoNets:
    """Rebuild actor nets from a training checkpoint (policy only)."""
    ckpt = latest_checkpoint(Path(run_dir) / "checkpoints")
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint under {run_dir}")
    state = load_checkpoint(ckpt)["trainer"]
    nets = MappoNets(
        env.local_obs_dim, env.joint_state_dim - 1, env.num_agents, env.num_actions, hidden
    )
    nets.load_state_dict(state["nets"])
    return nets


def nets_policy(nets: MappoNets):
    def policy(local_obs, joint_state):
        with torch.no_grad():
            actions, _ = nets.act(local_obs)
        return actions.tolist()

    return policy


def extract_goal_states(buffer: TrajectoryBuffer) -> np.ndarray:
    """States immediately after a delivery (count feature increased)."""
    goals = []
    for episode in buffer.episodes:
        counts = episode[:, -1]
        increased = np.where(np.diff(counts) > 0.5 * DELIVERY_SCALE)[0] + 1
        goals.extend(episode[t] for t in increased)
    return np.stack(goals) if goals else np.empty((0, buffer.state_dim), dtype=np.float32)


def anchor_outcomes(
    episodes: list[np.ndarray], window: int, per_episode: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample anchor states and their realized within-window delivery outcomes."""
    anchors, outcomes = [], []
    for episode in episodes:
        horizon = len(episode)
        if horizon <= window:
            continue
        for t in rng.integers(0, horizon - window, size=per_episode):
            anchors.append(episode[t])
            outcomes.append(
                float(episode[t + window - 1][-1] - episode[t][-1] >= 0.5 * DELIVERY_SCALE)
            )
    return np.stack(anchors), np.array(outcomes, dtype=np.float64)


def run_pipeline(
    layout: str = "cramped_room",
    horizon: int = 400,
    vanilla_run: str = "runs/gate_mappo_cramped/seed0",
    rnd_run: str = "runs/probe_mappo_rnd_cramped/seed0",
    out_dir: str | Path = "runs/teacher_cramped",
    reports_dir: str | Path = "reports",
    min_successes: int = 25,
    max_warmup_steps: int = 120_000,
    window: int = 32,
    teacher_steps: int = 4000,
    d_model: int = 128,
    num_layers: int = 3,
    nhead: int = 4,
    n_anchors: int | None = None,  # None = use ALL unique training anchors
    samples_per_state: int = 64,
    feasibility_samples: int = 8,
    distill_hidden: int = 256,
    distill_epochs: int = 600,
    device: str = "auto",
    seed: int = 0,
) -> dict:
    import time as _time

    start_time = _time.monotonic()
    out_dir = Path(out_dir)
    reports_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None

    base = OvercookedSparseEnv(layout=layout, horizon=horizon, encoding="lossless")
    env = DeliveryAugmentedOvercooked(base)

    # ladder: trained vanilla policy (measured 0.0 success) -> trained RND policy
    vanilla_nets = load_policy_nets(vanilla_run, env)
    rnd_nets = load_policy_nets(rnd_run, env)
    buffer, warmup_report = collect_warmup(
        env,
        ladder=[("mappo_vanilla", nets_policy(vanilla_nets)),
                ("mappo_rnd", nets_policy(rnd_nets))],
        min_successes=min_successes,
        max_env_steps=max_warmup_steps,
        report_path=reports_dir / "warmup_buffer_cramped.json",
    )
    buffer.save(out_dir / "warmup_buffer.npz")

    # episode-level holdout split, disjoint from teacher training data
    indices = rng.permutation(len(buffer.episodes))
    holdout_count = max(2, len(indices) // 5)
    holdout_idx = set(indices[:holdout_count].tolist())
    train_episodes = [buffer.episodes[i] for i in range(len(buffer.episodes)) if i not in holdout_idx]
    holdout_episodes = [buffer.episodes[i] for i in sorted(holdout_idx)]
    ensure_disjoint(
        {f"ep{i}" for i in sorted(holdout_idx)},
        {f"ep{i}" for i in range(len(buffer.episodes)) if i not in holdout_idx},
    )

    train_buffer = TrajectoryBuffer(buffer.state_dim)
    for i, episode in enumerate(buffer.episodes):
        if i not in holdout_idx:
            train_buffer.add_episode(
                episode, buffer.returns[i], buffer.successes[i], buffer.sources[i]
            )

    dataset = TrajectoryWindowDataset(train_buffer, window=window, stride=4)

    # behavioral policy embedding on a fixed probe set; teacher is conditional
    probes = collect_probe_observations(env, nets_policy(rnd_nets), n_probes=16, rng=rng)
    embedding = BehavioralPolicyEmbedding(probes)
    emb_vanilla = embedding.embed(vanilla_nets)
    emb_rnd = embedding.embed(rnd_nets)
    cond_dim = emb_rnd.size

    window_sources = []
    for ep, source in zip(train_buffer.episodes, train_buffer.sources):
        n_windows = max(0, (len(ep) - window) // 4 + 1)
        window_sources.extend([source] * n_windows)
    window_conds = np.stack(
        [emb_rnd if s == "mappo_rnd" else emb_vanilla for s in window_sources]
    ).astype(np.float32)

    model = TrajectoryDenoiser(
        state_dim=env.joint_state_dim, window=window, cond_dim=cond_dim,
        d_model=d_model, nhead=nhead, num_layers=num_layers,
    ).to(device)
    diffusion = GaussianDiffusion(num_steps=100)

    cond_pool = torch.as_tensor(window_conds)

    def cond_fn(batch):  # match the batch by sampling embeddings of the same mix
        idx = torch.randint(0, len(cond_pool), (batch.shape[0],))
        return cond_pool[idx]

    history = train_teacher(
        model, diffusion, dataset, steps=teacher_steps, batch_size=64, lr=3e-4, cond_fn=cond_fn, seed=seed
    )
    torch.save(
        {"model": model.state_dict(), "mean": dataset.mean, "std": dataset.std,
         "window": window, "cond_dim": cond_dim},
        out_dir / "teacher.pt",
    )

    validator = OvercookedExactValidator(
        np.concatenate(train_buffer.episodes, axis=0),
        grid_shape=base.lossless_shape,
        mdp=base._mdp,
    )
    checker = OvercookedSimulatorChecker(base._mdp, base.lossless_shape)
    goal_states = extract_goal_states(train_buffer)

    provenance = {
        "layout": layout,
        "native_layout_name": base.native_layout_name,
        "horizon": horizon,
        "seed": seed,
        "window": window,
        "teacher_steps": teacher_steps,
        "delivery_feature_scale": DELIVERY_SCALE,
        "validator_method": OvercookedExactValidator.method,
        "checker_method": OvercookedSimulatorChecker.method,
        "checker_scope": "bridge windows (quality report) and feasibility filtering",
        "ladder_checkpoints": {"mappo_vanilla": str(vanilla_run), "mappo_rnd": str(rnd_run)},
        "episodes_train": len(train_episodes),
        "episodes_holdout": len(holdout_episodes),
        "d_model": d_model,
        "num_layers": num_layers,
        "nhead": nhead,
        "samples_per_state": samples_per_state,
        "feasibility_samples": feasibility_samples,
        "distill_hidden": distill_hidden,
        "distill_epochs": distill_epochs,
        "device": device,
        "gpu_name": gpu_name,
    }

    # stamp provenance into the warmup artifact as soon as it is known
    warmup_path = reports_dir / "warmup_buffer_cramped.json"
    warmup_artifact = json.loads(warmup_path.read_text())
    warmup_artifact["provenance"] = provenance
    warmup_path.write_text(json.dumps(warmup_artifact, indent=2, sort_keys=True))

    # anchors for measurement: training-side for quality, holdout for calibration;
    # deduplicate by state key so the signal table covers unique states
    from reap.signals.potential import state_key

    raw_train_anchors, _ = anchor_outcomes(train_episodes, window, 4, rng)
    raw_anchor_count = len(raw_train_anchors)
    seen: dict[bytes, int] = {}
    for i, anchor in enumerate(raw_train_anchors):
        seen.setdefault(state_key(anchor), i)
    # deterministic order: sort unique anchors by their state key bytes
    unique_rows = sorted(seen.items(), key=lambda kv: kv[0])
    unique_anchors = raw_train_anchors[[i for _, i in unique_rows]]
    keep_n = len(unique_anchors) if n_anchors is None else min(n_anchors, len(unique_anchors))
    train_anchors = unique_anchors[:keep_n]
    anchor_counts = {
        "raw_anchor_rows": int(raw_anchor_count),
        "unique_anchor_states": int(len(unique_anchors)),
        "anchors_used": int(keep_n),
    }
    generator = torch.Generator().manual_seed(seed)

    # generation-quality measurement uses a fixed deterministic anchor subset
    quality_anchors = train_anchors[: min(48, len(train_anchors))]
    forward_windows = diffusion.sample(
        model, n=len(quality_anchors) * 4,
        pin={0: torch.as_tensor(dataset.normalize(quality_anchors), dtype=torch.float32)
             .repeat_interleave(4, dim=0)},
        cond=torch.as_tensor(emb_rnd).expand(len(quality_anchors) * 4, -1),
        guidance_scale=2.0, generator=generator,
    )
    bridge_idx = rng.choice(len(goal_states), size=len(quality_anchors) * 4)
    bridge_windows = diffusion.sample(
        model, n=len(quality_anchors) * 4,
        pin={0: torch.as_tensor(dataset.normalize(quality_anchors), dtype=torch.float32)
             .repeat_interleave(4, dim=0),
             window - 1: torch.as_tensor(
                 dataset.normalize(goal_states[bridge_idx]), dtype=torch.float32)},
        generator=generator,
    )
    bridges_denorm = dataset.denormalize(bridge_windows.numpy())
    projected_bridges = validator.project(
        bridges_denorm.reshape(-1, env.joint_state_dim)
    ).reshape(bridges_denorm.shape)
    consistency = np.mean([
        all(checker.realizable(w[t], w[t + 1]) for t in range(len(w) - 1))
        for w in projected_bridges
    ])
    quality_report = generation_quality_report(
        dataset.denormalize(forward_windows.numpy()),
        validator=validator,
        success_fn=window_success,
        bridge_consistency_rate=float(consistency),
    )
    quality_report["provenance"] = {
        **provenance,
        "teacher_loss_first20": float(np.mean(history[:20])),
        "teacher_loss_last20": float(np.mean(history[-20:])),
        "forward_sample_windows": int(len(forward_windows)),
        "bridge_sample_windows": int(len(projected_bridges)),
    }
    (reports_dir / "teacher_quality_cramped.json").write_text(
        json.dumps(quality_report, indent=2, sort_keys=True)
    )

    # direct-query propensity on holdout anchors + held-out calibration check
    holdout_anchors, holdout_realized = anchor_outcomes(holdout_episodes, window, 3, rng)
    keep = rng.choice(
        len(holdout_anchors), size=min(48, len(holdout_anchors)), replace=False
    )
    holdout_anchors, holdout_realized = holdout_anchors[keep], holdout_realized[keep]
    propensity = estimate_propensity(
        diffusion, model, dataset, holdout_anchors,
        policy_embedding=emb_rnd, success_fn=window_success, validator=validator,
        samples_per_state=samples_per_state, guidance_scale=2.0, generator=generator,
    )
    ladder = CalibrationLadder(beta=1.0)
    cal_event = ladder.check(propensity, holdout_realized)
    cal_report = {
        **{k: v for k, v in cal_event.items() if k != "bins"},
        "bins": cal_event["bins"],
        "holdout_anchors": int(len(holdout_anchors)),
        "holdout_disjointness": "episode-level split; holdout episodes never enter "
                                "teacher training windows (ensure_disjoint on episode ids)",
        "provenance": provenance,
    }
    (reports_dir / "calibration_cramped.json").write_text(
        json.dumps(cal_report, indent=2, sort_keys=True)
    )

    # feasibility/propensity direct queries over ALL kept anchors, chunked so
    # large sample counts fit device memory
    def chunked(estimator, anchors, chunk_states):
        parts = []
        for start in range(0, len(anchors), chunk_states):
            parts.append(estimator(anchors[start : start + chunk_states]))
        return np.concatenate(parts)

    feasibility = chunked(
        lambda a: estimate_feasibility(
            diffusion, model, dataset, a, goal_states,
            validator=validator, checker=checker,
            samples_per_state=feasibility_samples, generator=generator,
        ),
        train_anchors,
        chunk_states=max(1, 2048 // feasibility_samples),
    )
    train_prop = chunked(
        lambda a: estimate_propensity(
            diffusion, model, dataset, a,
            policy_embedding=emb_rnd, success_fn=window_success, validator=validator,
            samples_per_state=samples_per_state, guidance_scale=2.0, generator=generator,
        ),
        train_anchors,
        chunk_states=max(1, 2048 // samples_per_state),
    )

    # direct-query potential table (the Section-5.5 shortcut, artifact-recorded)
    from reap.signals import ReapPotential

    tau_gate = 0.5
    potential = ReapPotential(tau_gate=tau_gate)
    potential.update_tables(train_anchors, train_prop, feasibility)
    phi_values = np.array(
        [potential.value(s, horizon) for s in train_anchors], dtype=np.float64
    )
    gated_out = int(np.sum(feasibility < tau_gate))
    potential_report = {
        "tau_gate": tau_gate,
        "coverage_states": potential.coverage,
        "propensity_range": [float(train_prop.min()), float(train_prop.max())],
        "feasibility_range": [float(feasibility.min()), float(feasibility.max())],
        "states_gated_out": gated_out,
        "states_passing_gate": int(len(train_anchors) - gated_out),
        "phi_range": [float(phi_values.min()), float(phi_values.max())],
        "phi_bounded_in_unit_interval": bool(np.all((phi_values >= 0) & (phi_values <= 1))),
        "feasibility_use": "gate only — never a reward magnitude (unit-test enforced)",
        "provenance": provenance,
    }
    (reports_dir / "potential_table_cramped.json").write_text(
        json.dumps(potential_report, indent=2, sort_keys=True)
    )

    # deterministic unique-key 3:1 split: every 4th anchor (by sorted-key
    # order) is held out for the fidelity check
    holdout_mask = np.arange(len(train_anchors)) % 4 == 3
    fit_anchors, eval_anchors = train_anchors[~holdout_mask], train_anchors[holdout_mask]
    fit_prop, eval_prop = train_prop[~holdout_mask], train_prop[holdout_mask]
    fit_feas, eval_feas = feasibility[~holdout_mask], feasibility[holdout_mask]
    distill_provenance = {
        **provenance,
        **anchor_counts,
        "distill_train_states": int(len(fit_anchors)),
        "distill_holdout_states": int(len(eval_anchors)),
        "split_rule": "deterministic by sorted state key: every 4th anchor held out",
    }
    p_hat = DistilledPredictor(env.joint_state_dim, hidden=distill_hidden, seed=seed)
    p_hat.fit(fit_anchors, fit_prop, epochs=distill_epochs)
    p_report = distillation_fidelity_report(
        p_hat, eval_anchors, eval_prop,
        report_path=reports_dir / "distill_fidelity_cramped.json",
        provenance=distill_provenance,
    )
    f_hat = DistilledPredictor(env.joint_state_dim, hidden=distill_hidden, seed=seed + 1)
    f_hat.fit(fit_anchors, fit_feas, epochs=distill_epochs)
    f_report = distillation_fidelity_report(
        f_hat, eval_anchors, eval_feas,
        report_path=reports_dir / "distill_fidelity_feasibility_cramped.json",
        provenance=distill_provenance,
    )
    torch.save({"p_hat": p_hat.state_dict(), "f_hat": f_hat.state_dict()},
               out_dir / "predictors.pt")

    # schema gate: every durable pipeline artifact must carry provenance
    required_artifacts = [
        "warmup_buffer_cramped.json",
        "teacher_quality_cramped.json",
        "calibration_cramped.json",
        "potential_table_cramped.json",
        "distill_fidelity_cramped.json",
        "distill_fidelity_feasibility_cramped.json",
    ]
    for name in required_artifacts:
        artifact = json.loads((reports_dir / name).read_text())
        if "provenance" not in artifact or not artifact["provenance"]:
            raise RuntimeError(f"pipeline artifact {name} lacks provenance")

    summary = {
        "runtime_seconds": round(_time.monotonic() - start_time, 1),
        "provenance": provenance,
        "potential_table": {k: potential_report[k] for k in
                            ("tau_gate", "coverage_states", "states_gated_out",
                             "phi_bounded_in_unit_interval")},
        "warmup": {k: warmup_report[k] for k in ("episodes", "success_count", "total_env_steps")},
        "warmup_gate_met": warmup_report["gate"]["met"],
        "teacher_loss_first20": float(np.mean(history[:20])),
        "teacher_loss_last20": float(np.mean(history[-20:])),
        "quality": {k: quality_report[k] for k in
                    ("invalid_state_rate", "endpoint_success_rate",
                     "bridge_consistency_rate", "shaping_enabled")},
        "calibration": {k: cal_report[k] for k in ("raw_ece", "brier", "action", "beta_after")},
        "propensity_holdout_mean": float(np.mean(propensity)),
        "realized_holdout_mean": float(np.mean(holdout_realized)),
        "feasibility_mean": float(np.mean(feasibility)),
        "distill_p_hat": {"mae": p_report["mae"], "passed": p_report["passed"]},
        "distill_f_hat": {"mae": f_report["mae"], "passed": f_report["passed"]},
    }
    (reports_dir / "teacher_pipeline_summary_cramped.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary
