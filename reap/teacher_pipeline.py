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


class OvercookedLosslessValidator(StateValidator):
    """Structural validity for augmented lossless-grid states.

    Built from the training buffer: per-channel value ranges, the set of
    static channels (identical across all training states, e.g. counters and
    dispensers), and the player-location channels which must be one-hot.
    ``project`` rounds and clips; ``is_valid`` checks structure WITHOUT
    forcing it, so the invalid-state rate stays a meaningful measurement.
    """

    def __init__(self, training_states: np.ndarray, grid_shape: tuple[int, int, int]):
        self.grid_shape = grid_shape  # (H, W, C) of the unaugmented encoding
        self.grid_size = int(np.prod(grid_shape))
        flat = np.asarray(training_states, dtype=np.float64)
        self.low = flat.min(axis=0)
        self.high = flat.max(axis=0)
        self.static_mask = self.low == self.high  # constant across training data
        self.static_values = self.low.copy()

    def project(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float64)
        projected = np.clip(np.round(states), self.low, self.high)
        return projected

    def _grids(self, states: np.ndarray) -> np.ndarray:
        return states[:, : self.grid_size].reshape(-1, *self.grid_shape)

    def is_valid(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float64)
        ok = np.ones(len(states), dtype=bool)
        # static channels must match the layout constants
        static_diff = np.abs(states[:, self.static_mask] - self.static_values[self.static_mask])
        ok &= (static_diff < 1e-6).all(axis=1)
        # each player-location channel holds exactly one cell
        grids = self._grids(states)
        for channel in (0, 1):
            player = grids[..., channel].reshape(len(states), -1)
            ok &= np.isclose(player.sum(axis=1), 1.0) & np.isclose(player.max(axis=1), 1.0)
        return ok


class OvercookedStructuralChecker(TransitionChecker):
    """Approximate dynamics consistency on encoded states (documented).

    Checks per-player movement of at most one cell, static channels staying
    fixed, and a monotone delivery counter increasing by at most one per
    step. An exact simulator-decode check is queued as follow-up work.
    """

    def __init__(self, validator: OvercookedLosslessValidator):
        self.validator = validator

    def _positions(self, state: np.ndarray) -> list[tuple[int, int]]:
        grid = state[: self.validator.grid_size].reshape(*self.validator.grid_shape)
        positions = []
        for channel in (0, 1):
            idx = np.unravel_index(np.argmax(grid[..., channel]), grid[..., channel].shape)
            positions.append((int(idx[0]), int(idx[1])))
        return positions

    def realizable(self, state: np.ndarray, next_state: np.ndarray) -> bool:
        state = np.asarray(state, dtype=np.float64)
        next_state = np.asarray(next_state, dtype=np.float64)
        mask = self.validator.static_mask
        if not np.allclose(state[mask], next_state[mask]):
            return False
        delta_count = next_state[-1] - state[-1]
        if not -1e-6 <= delta_count <= DELIVERY_SCALE + 1e-6:
            return False
        for (r0, c0), (r1, c1) in zip(self._positions(state), self._positions(next_state)):
            if abs(r0 - r1) + abs(c0 - c1) > 1:
                return False
        return True


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
    n_anchors: int = 48,
    samples_per_state: int = 16,
    seed: int = 0,
) -> dict:
    out_dir = Path(out_dir)
    reports_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

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
        d_model=128, nhead=4, num_layers=3,
    )
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

    validator = OvercookedLosslessValidator(
        np.concatenate(train_buffer.episodes, axis=0),
        grid_shape=base.lossless_shape,
    )
    checker = OvercookedStructuralChecker(validator)
    goal_states = extract_goal_states(train_buffer)

    # anchors for measurement: training-side for quality, holdout for calibration
    train_anchors, _ = anchor_outcomes(train_episodes, window, 2, rng)
    train_anchors = train_anchors[
        rng.choice(len(train_anchors), size=min(n_anchors, len(train_anchors)), replace=False)
    ]
    generator = torch.Generator().manual_seed(seed)

    forward_windows = diffusion.sample(
        model, n=len(train_anchors) * 4,
        pin={0: torch.as_tensor(dataset.normalize(train_anchors), dtype=torch.float32)
             .repeat_interleave(4, dim=0)},
        cond=torch.as_tensor(emb_rnd).expand(len(train_anchors) * 4, -1),
        guidance_scale=2.0, generator=generator,
    )
    bridge_idx = rng.choice(len(goal_states), size=len(train_anchors) * 4)
    bridge_windows = diffusion.sample(
        model, n=len(train_anchors) * 4,
        pin={0: torch.as_tensor(dataset.normalize(train_anchors), dtype=torch.float32)
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
        report_path=reports_dir / "teacher_quality_cramped.json",
    )

    # direct-query propensity on holdout anchors + held-out calibration check
    holdout_anchors, holdout_realized = anchor_outcomes(holdout_episodes, window, 3, rng)
    keep = rng.choice(
        len(holdout_anchors), size=min(n_anchors, len(holdout_anchors)), replace=False
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
    }
    (reports_dir / "calibration_cramped.json").write_text(
        json.dumps(cal_report, indent=2, sort_keys=True)
    )

    # feasibility direct queries on training anchors, then distill both signals
    feasibility = estimate_feasibility(
        diffusion, model, dataset, train_anchors, goal_states,
        validator=validator, checker=checker,
        samples_per_state=8, generator=generator,
    )
    train_prop = estimate_propensity(
        diffusion, model, dataset, train_anchors,
        policy_embedding=emb_rnd, success_fn=window_success, validator=validator,
        samples_per_state=samples_per_state, guidance_scale=2.0, generator=generator,
    )

    split = len(train_anchors) * 3 // 4
    p_hat = DistilledPredictor(env.joint_state_dim, seed=seed)
    p_hat.fit(train_anchors[:split], train_prop[:split], epochs=300)
    p_report = distillation_fidelity_report(
        p_hat, train_anchors[split:], train_prop[split:],
        report_path=reports_dir / "distill_fidelity_cramped.json",
    )
    f_hat = DistilledPredictor(env.joint_state_dim, seed=seed + 1)
    f_hat.fit(train_anchors[:split], feasibility[:split], epochs=300)
    f_report = distillation_fidelity_report(
        f_hat, train_anchors[split:], feasibility[split:],
        report_path=reports_dir / "distill_fidelity_feasibility_cramped.json",
    )
    torch.save({"p_hat": p_hat.state_dict(), "f_hat": f_hat.state_dict()},
               out_dir / "predictors.pt")

    summary = {
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
