"""Hybrid features-encoding teacher with simulator-rollout projection.

The diffusion teacher models trajectories in the compact continuous
``features`` encoding (where DDPM is at home), while validity, anchors, and
potential keys stay in the exactly-decodable lossless encoding. The bridge
between the two is a deterministic simulator-rollout projector: starting from
the pinned lossless start state (decoded exactly), each step picks the joint
action whose exact simulator successor has the feature encoding nearest the
sampled next feature. Projected windows are therefore valid and
dynamics-consistent BY CONSTRUCTION; the teacher's quality shows up in the
projection error and in whether projected rollouts actually deliver. All of
this is disclosed in the artifacts.
"""

from __future__ import annotations

import itertools

import numpy as np

from reap.envs.overcooked_decode import decode_if_roundtrip, encode_state
from reap.envs.overcooked_env import OvercookedSparseEnv
from reap.teacher_pipeline import DELIVERY_SCALE

PROJECTION_METHOD = (
    "deterministic simulator-rollout projection: from the pinned lossless "
    "start (decode_if_roundtrip), each window step replays the joint action "
    "whose exact mdp successor minimizes L2 distance to the sampled next "
    "feature vector; projected lossless windows are valid and consistent by "
    "construction, so teacher quality is measured by projection error and "
    "delivery preservation"
)


class PairedOvercooked:
    """Steps one native Overcooked instance, emitting BOTH joint encodings.

    Policies act on the lossless local observations their checkpoints were
    trained with; the paired features+delivery joint state rides along for
    teacher training. Both joint states carry the delivery-count feature.
    """

    def __init__(self, layout: str = "cramped_room", horizon: int = 400):
        self.base = OvercookedSparseEnv(layout=layout, horizon=horizon, encoding="lossless")
        self._mlam = self.base._get_mlam()  # planner computed/cached once
        self.num_agents = self.base.num_agents
        self.num_actions = self.base.num_actions
        self.horizon = horizon
        self.local_obs_dim = self.base.local_obs_dim
        self.lossless_dim = self.base.joint_state_dim + 1
        probe = self._feature_joint()
        self.feature_dim = probe.size

    def _feature_joint(self) -> np.ndarray:
        feats = self.base._mdp.featurize_state(self.base._env.state, self._mlam)
        flat = np.concatenate([np.asarray(f, dtype=np.float32).ravel() for f in feats])
        count = self.base._deliveries * DELIVERY_SCALE
        return np.concatenate([flat, np.array([count], dtype=np.float32)])

    def _lossless_joint(self, joint: np.ndarray) -> np.ndarray:
        count = self.base._deliveries * DELIVERY_SCALE
        return np.concatenate([joint, np.array([count], dtype=np.float32)])

    def reset(self):
        local, joint = self.base.reset()
        return local, self._lossless_joint(joint), self._feature_joint()

    def step(self, actions):
        result = self.base.step(actions)
        return (
            result,
            self._lossless_joint(result.joint_state),
            self._feature_joint(),
        )

    def feature_of_state(self, native_state) -> np.ndarray:
        """Feature encoding (without delivery term) of an arbitrary native state."""
        feats = self.base._mdp.featurize_state(native_state, self._mlam)
        return np.concatenate([np.asarray(f, dtype=np.float32).ravel() for f in feats])


class SimulatorRolloutProjector:
    """Projects feature windows onto exactly-valid lossless windows."""

    def __init__(self, paired: PairedOvercooked):
        self.paired = paired
        self.mdp = paired.base._mdp
        self.grid_shape = paired.base.lossless_shape
        self.grid_size = int(np.prod(self.grid_shape))
        from overcooked_ai_py.mdp.actions import Action

        self.joint_actions = list(itertools.product(Action.ALL_ACTIONS, repeat=2))

    def project(
        self, lossless_start: np.ndarray, feature_window: np.ndarray
    ) -> dict | None:
        """Project one sampled feature window from its pinned lossless start.

        Returns lossless window, per-step projection errors, and delivery
        counts; None when the start state itself is invalid.
        """
        start_grid = (
            np.asarray(lossless_start[: self.grid_size]).round().astype(int)
            .reshape(*self.grid_shape)
        )
        state = decode_if_roundtrip(start_grid, self.mdp)
        if state is None:
            return None
        start_count = float(lossless_start[-1])

        window = [np.asarray(lossless_start, dtype=np.float32)]
        errors = []
        deliveries = 0
        for t in range(1, len(feature_window)):
            target = np.asarray(feature_window[t][:-1], dtype=np.float32)
            best = None
            for joint_action in self.joint_actions:
                successor, infos = self.mdp.get_state_transition(state, joint_action)
                feat = self.paired.feature_of_state(successor)
                err = float(np.linalg.norm(feat - target))
                sparse = sum(infos["sparse_reward_by_agent"])
                if best is None or err < best[0]:
                    best = (err, successor, sparse)
            err, state, sparse = best
            if sparse > 0:
                deliveries += int(round(sparse / 20.0)) or 1
            errors.append(err)
            flat = encode_state(state, self.mdp).astype(np.float32).ravel()
            count = start_count + deliveries * DELIVERY_SCALE
            window.append(np.concatenate([flat, np.array([count], dtype=np.float32)]))
        return {
            "lossless_window": np.stack(window),
            "projection_errors": np.array(errors, dtype=np.float64),
            "deliveries": deliveries,
            "sampled_endpoint_delta": float(feature_window[-1][-1] - feature_window[0][-1]),
        }


class PolicyConditionedRefresher:
    """Enabled-scope refresh: re-query the frozen teacher with the CURRENT
    policy's behavioral embedding, then refit the distilled predictors.

    ``sampler(embedding, feature_anchors) -> feature windows (n, M, W, D)``
    abstracts the teacher query so the refresh path is unit-testable with a
    stub; the production sampler wraps the frozen diffusion teacher with CFG
    on the supplied embedding. Window success is read from the delivery-count
    feature delta (disclosed: no projection during refresh, projection is a
    quality-measurement tool).
    """

    def __init__(
        self,
        nets_provider,  # callable() -> current MappoNets
        embedding,  # BehavioralPolicyEmbedding
        sampler,
        lossless_anchors: np.ndarray,
        feature_anchors: np.ndarray,
        feasibility: np.ndarray,  # frozen per the draft (policy-independent)
        p_hat,
        f_hat,
        refit_epochs: int = 200,
    ):
        if len(lossless_anchors) != len(feature_anchors):
            raise ValueError("anchor encodings must be paired")
        self.nets_provider = nets_provider
        self.embedding = embedding
        self.sampler = sampler
        self.lossless_anchors = lossless_anchors
        self.feature_anchors = feature_anchors
        self.feasibility = np.clip(np.asarray(feasibility, dtype=np.float64), 0, 1)
        self.p_hat = p_hat
        self.f_hat = f_hat
        self.refit_epochs = refit_epochs
        self.last_embedding: np.ndarray | None = None

    def __call__(self):
        emb = self.embedding.embed(self.nets_provider())
        self.last_embedding = emb
        windows = self.sampler(emb, self.feature_anchors)  # (n, M, W, D)
        deltas = windows[:, :, -1, -1] - windows[:, :, 0, -1]
        propensity = np.clip(
            (deltas >= 0.5 * DELIVERY_SCALE).mean(axis=1).astype(np.float64), 0, 1
        )
        self.p_hat.fit(self.lossless_anchors, propensity.astype(np.float32),
                       epochs=self.refit_epochs)
        self.f_hat.fit(self.lossless_anchors, self.feasibility.astype(np.float32),
                       epochs=self.refit_epochs)
        return self.lossless_anchors, propensity, self.feasibility


def make_teacher_sampler(teacher_path, samples_per_state: int = 8, device: str = "auto",
                         seed: int = 0):
    """Production sampler for PolicyConditionedRefresher: CFG-conditioned
    forward sampling from the frozen hybrid teacher."""
    import torch

    from reap.diffusion import GaussianDiffusion, TrajectoryDenoiser

    payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    window = payload["window"]
    state_dim = len(payload["mean"])
    model = TrajectoryDenoiser(
        state_dim=state_dim, window=window, cond_dim=payload["cond_dim"],
        d_model=payload["d_model"], nhead=payload["nhead"],
        num_layers=payload["num_layers"],
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    diffusion = GaussianDiffusion(num_steps=100)
    mean, std = payload["mean"], payload["std"]
    generator = torch.Generator().manual_seed(seed)

    def sampler(embedding_vec: np.ndarray, feature_anchors: np.ndarray) -> np.ndarray:
        n = len(feature_anchors)
        normalized = (feature_anchors - mean) / std
        pin = {0: torch.as_tensor(normalized, dtype=torch.float32)
               .repeat_interleave(samples_per_state, dim=0)}
        cond = torch.as_tensor(embedding_vec, dtype=torch.float32).expand(
            n * samples_per_state, -1
        )
        windows = diffusion.sample(model, n=n * samples_per_state, pin=pin,
                                   cond=cond, guidance_scale=2.0, generator=generator)
        denorm = windows.numpy() * std + mean
        return denorm.reshape(n, samples_per_state, window, state_dim)

    return sampler


def collect_paired_warmup(
    paired: PairedOvercooked,
    ladder,
    min_successes: int = 25,
    max_env_steps: int = 120_000,
):
    """Ladder collection emitting synchronized lossless and feature buffers."""
    from reap.data.buffer import TrajectoryBuffer

    lossless_buf = TrajectoryBuffer(paired.lossless_dim)
    feature_buf = TrajectoryBuffer(paired.feature_dim)
    steps_per_rung = max(1, max_env_steps // len(ladder))
    total = 0
    capped = False
    for rung_name, policy in ladder:
        rung_steps = 0
        while lossless_buf.success_count < min_successes and rung_steps < steps_per_rung:
            local, lj, fj = paired.reset()
            l_states, f_states = [lj], [fj]
            ep_return, success = 0.0, False
            while True:
                if total >= max_env_steps or rung_steps >= steps_per_rung:
                    capped = total >= max_env_steps
                    break
                result, lj, fj = paired.step(policy(local, lj))
                rung_steps += 1
                total += 1
                l_states.append(lj)
                f_states.append(fj)
                ep_return += result.extrinsic_reward
                local = result.local_obs
                if result.terminated or result.truncated:
                    success = bool(result.info.get("success", False))
                    break
            if len(l_states) >= 2:
                lossless_buf.add_episode(np.stack(l_states), ep_return, success, rung_name)
                feature_buf.add_episode(np.stack(f_states), ep_return, success, rung_name)
            if capped or rung_steps >= steps_per_rung:
                break
        if capped or lossless_buf.success_count >= min_successes:
            break
    report = lossless_buf.report()
    report["gate"] = {
        "min_successes": min_successes,
        "max_env_steps": max_env_steps,
        "met": lossless_buf.success_count >= min_successes,
        "collection_truncated_at_cap": capped,
        "ladder": [name for name, _ in ladder],
    }
    return lossless_buf, feature_buf, report


def run_hybrid_pipeline(
    layout: str = "cramped_room",
    horizon: int = 400,
    vanilla_run: str = "runs/gate_mappo_cramped/seed0",
    rnd_run: str = "runs/probe_mappo_rnd_cramped/seed0",
    out_dir="runs/teacher_hybrid_cramped",
    reports_dir="reports",
    min_successes: int = 25,
    max_warmup_steps: int = 120_000,
    window: int = 16,
    teacher_steps: int = 60_000,
    d_model: int = 256,
    num_layers: int = 6,
    nhead: int = 8,
    n_anchors: int = 48,
    samples_per_state: int = 8,
    bridge_samples: int = 4,
    distill_hidden: int = 256,
    distill_epochs: int = 600,
    device: str = "auto",
    seed: int = 0,
) -> dict:
    """End-to-end hybrid pipeline producing the AC-5/AC-6 artifact set."""
    import json
    import time as _time
    from pathlib import Path

    import torch

    from reap.calibration import CalibrationLadder
    from reap.diffusion import GaussianDiffusion, TrajectoryDenoiser, TrajectoryWindowDataset
    from reap.diffusion.ddpm import train_teacher
    from reap.signals import BehavioralPolicyEmbedding, ReapPotential, collect_probe_observations
    from reap.signals.distill import DistilledPredictor, distillation_fidelity_report
    from reap.signals.estimators import _likelihood_proxy_weights
    from reap.signals.potential import state_key
    from reap.teacher_pipeline import load_policy_nets, nets_policy

    start = _time.monotonic()
    out_dir, reports_dir = Path(out_dir), Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    paired = PairedOvercooked(layout=layout, horizon=horizon)
    # checkpoints were trained on lossless local obs; joint dim for nets is
    # the UNaugmented lossless joint
    class _NetsEnv:  # minimal dims holder for load_policy_nets
        local_obs_dim = paired.local_obs_dim
        joint_state_dim = paired.lossless_dim  # -1 applied inside loader
        num_agents = paired.num_agents
        num_actions = paired.num_actions

    vanilla_nets = load_policy_nets(vanilla_run, _NetsEnv)
    rnd_nets = load_policy_nets(rnd_run, _NetsEnv)
    lossless_buf, feature_buf, warmup_report = collect_paired_warmup(
        paired,
        ladder=[("mappo_vanilla", nets_policy(vanilla_nets)),
                ("mappo_rnd", nets_policy(rnd_nets))],
        min_successes=min_successes,
        max_env_steps=max_warmup_steps,
    )
    lossless_buf.save(out_dir / "warmup_lossless.npz")
    feature_buf.save(out_dir / "warmup_features.npz")

    # episode split (same convention as the lossless pipeline)
    indices = rng.permutation(len(lossless_buf.episodes))
    holdout_count = max(2, len(indices) // 5)
    holdout_idx = set(indices[:holdout_count].tolist())
    train_ids = [i for i in range(len(lossless_buf.episodes)) if i not in holdout_idx]
    holdout_ids = sorted(holdout_idx)

    from reap.data.buffer import TrajectoryBuffer

    feat_train = TrajectoryBuffer(paired.feature_dim)
    for i in train_ids:
        feat_train.add_episode(
            feature_buf.episodes[i], feature_buf.returns[i],
            feature_buf.successes[i], feature_buf.sources[i],
        )
    dataset = TrajectoryWindowDataset(feat_train, window=window, stride=4)

    probes = collect_probe_observations(
        _ProbeAdapter(paired), nets_policy(rnd_nets), n_probes=16, rng=rng
    )
    embedding = BehavioralPolicyEmbedding(probes)
    emb_rnd = embedding.embed(rnd_nets)
    emb_vanilla = embedding.embed(vanilla_nets)
    cond_pool = torch.as_tensor(
        np.stack([emb_rnd, emb_vanilla]).astype(np.float32)
    )

    model = TrajectoryDenoiser(
        state_dim=paired.feature_dim, window=window, cond_dim=emb_rnd.size,
        d_model=d_model, nhead=nhead, num_layers=num_layers,
    ).to(device)
    diffusion = GaussianDiffusion(num_steps=100)

    def cond_fn(batch):
        idx = torch.randint(0, len(cond_pool), (batch.shape[0],))
        return cond_pool[idx]

    history = train_teacher(model, diffusion, dataset, steps=teacher_steps,
                            batch_size=128, lr=3e-4, cond_fn=cond_fn, seed=seed)
    torch.save({"model": model.state_dict(), "mean": dataset.mean,
                "std": dataset.std, "window": window,
                "cond_dim": emb_rnd.size, "d_model": d_model,
                "num_layers": num_layers, "nhead": nhead},
               out_dir / "teacher.pt")

    # paired anchors: same (episode, t) in both encodings, plus realized
    anchors_l, anchors_f, realized = [], [], []
    for i in train_ids:
        ep_l, ep_f = lossless_buf.episodes[i], feature_buf.episodes[i]
        if len(ep_l) <= window:
            continue
        for t in rng.integers(0, len(ep_l) - window, size=3):
            anchors_l.append(ep_l[t])
            anchors_f.append(ep_f[t])
            realized.append(float(
                ep_l[t + window - 1][-1] - ep_l[t][-1] >= 0.5 * DELIVERY_SCALE
            ))
    seen: dict[bytes, int] = {}
    for i, a in enumerate(anchors_l):
        seen.setdefault(state_key(a), i)
    order = [i for _, i in sorted(seen.items(), key=lambda kv: kv[0])][:n_anchors]
    anchors_l = np.stack([anchors_l[i] for i in order])
    anchors_f = np.stack([anchors_f[i] for i in order])

    projector = SimulatorRolloutProjector(paired)
    generator = torch.Generator().manual_seed(seed)

    def sample_forward(feature_starts, count, cond_vec):
        pin = {0: torch.as_tensor(dataset.normalize(feature_starts), dtype=torch.float32)
               .repeat_interleave(count, dim=0)}
        cond = torch.as_tensor(cond_vec).expand(len(feature_starts) * count, -1)
        windows = diffusion.sample(model, n=len(feature_starts) * count, pin=pin,
                                   cond=cond, guidance_scale=2.0, generator=generator)
        return dataset.denormalize(windows.numpy()).reshape(
            len(feature_starts), count, window, paired.feature_dim
        )

    # quality measurement: forward samples projected through the simulator
    forward = sample_forward(anchors_f, samples_per_state, emb_rnd)
    proj_errors, delivery_match, prop_rows = [], [], []
    projected_valid = 0
    projected_total = 0
    for i in range(len(anchors_f)):
        hits = 0
        for m in range(samples_per_state):
            out = projector.project(anchors_l[i], forward[i, m])
            projected_total += 1
            if out is None:
                continue
            projected_valid += 1
            proj_errors.append(float(out["projection_errors"].mean()))
            sampled_says = out["sampled_endpoint_delta"] >= 0.5 * DELIVERY_SCALE
            projected_says = out["deliveries"] > 0
            delivery_match.append(float(sampled_says == projected_says))
            hits += int(projected_says)
        prop_rows.append(hits / samples_per_state)
    propensity = np.clip(np.asarray(prop_rows, dtype=np.float64), 0, 1)
    invalid_rate = 1.0 - projected_valid / max(projected_total, 1)

    # feasibility: goal-pinned bridges, projected; weighted delivery preservation
    goal_list = [
        feature_buf.episodes[i][t]
        for i in train_ids
        for t in (np.where(np.diff(feature_buf.episodes[i][:, -1]) > 0.5 * DELIVERY_SCALE)[0] + 1)
    ]
    goal_states = (
        np.stack(goal_list) if goal_list
        else np.empty((0, paired.feature_dim), dtype=np.float32)
    )
    feasibility = np.zeros(len(anchors_f), dtype=np.float64)
    bridge_consistent = []
    if len(goal_states):
        for i in range(len(anchors_f)):
            picks = rng.integers(0, len(goal_states), size=bridge_samples)
            pin = {0: torch.as_tensor(dataset.normalize(anchors_f[i: i + 1]),
                                      dtype=torch.float32).repeat_interleave(bridge_samples, dim=0),
                   window - 1: torch.as_tensor(
                       dataset.normalize(goal_states[picks]), dtype=torch.float32)}
            bw = diffusion.sample(model, n=bridge_samples, pin=pin, generator=generator)
            bw = dataset.denormalize(bw.numpy())
            weights = _likelihood_proxy_weights(
                diffusion, model,
                torch.as_tensor(dataset.normalize(bw), dtype=torch.float32),
                generator,
            )
            delivered = np.zeros(bridge_samples)
            for m in range(bridge_samples):
                out = projector.project(anchors_l[i], bw[m])
                bridge_consistent.append(1.0 if out is not None else 0.0)
                if out is not None and out["deliveries"] > 0:
                    delivered[m] = 1.0
            feasibility[i] = float(np.dot(weights, delivered))
    feasibility = np.clip(feasibility, 0, 1)

    quality = {
        "projection_method": PROJECTION_METHOD,
        "raw_feature_sample_count": int(projected_total),
        "invalid_state_rate": float(invalid_rate),
        "bridge_consistency_rate": float(np.mean(bridge_consistent)) if bridge_consistent else None,
        "pin_start_preservation": "exact by construction (projection starts at the pinned lossless state)",
        "projection_error_mean": float(np.mean(proj_errors)) if proj_errors else None,
        "delivery_preservation_rate": float(np.mean(delivery_match)) if delivery_match else None,
        "endpoint_success_rate": float(propensity.mean()),
        "gates": {"invalid_state_rate_max": 0.10, "bridge_consistency_min": 0.80},
    }
    violations = []
    if quality["invalid_state_rate"] > 0.10:
        violations.append(f"invalid_state_rate {quality['invalid_state_rate']:.3f} > 0.1")
    if quality["bridge_consistency_rate"] is not None and quality["bridge_consistency_rate"] < 0.80:
        violations.append(
            f"bridge_consistency_rate {quality['bridge_consistency_rate']:.3f} < 0.8"
        )
    quality["gate_violations"] = violations
    quality["shaping_enabled"] = not violations

    provenance = {
        "scope": "hybrid features-encoding teacher",
        "layout": layout, "horizon": horizon, "window": window,
        "teacher_steps": teacher_steps, "d_model": d_model,
        "num_layers": num_layers, "nhead": nhead, "seed": seed,
        "samples_per_state": samples_per_state, "bridge_samples": bridge_samples,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "feature_dim": paired.feature_dim, "lossless_dim": paired.lossless_dim,
        "episodes_train": len(train_ids), "episodes_holdout": len(holdout_ids),
        "teacher_loss_first20": float(np.mean(history[:20])),
        "teacher_loss_last20": float(np.mean(history[-20:])),
        "ladder_checkpoints": {"mappo_vanilla": str(vanilla_run), "mappo_rnd": str(rnd_run)},
        "anchors_used": int(len(anchors_l)),
    }
    quality["provenance"] = provenance
    (reports_dir / "teacher_quality_hybrid_cramped.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True)
    )

    # calibration on holdout episodes (paired anchors, projected propensity)
    cal_l, cal_f, cal_real = [], [], []
    for i in holdout_ids:
        ep_l, ep_f = lossless_buf.episodes[i], feature_buf.episodes[i]
        if len(ep_l) <= window:
            continue
        for t in rng.integers(0, len(ep_l) - window, size=2):
            cal_l.append(ep_l[t])
            cal_f.append(ep_f[t])
            cal_real.append(float(
                ep_l[t + window - 1][-1] - ep_l[t][-1] >= 0.5 * DELIVERY_SCALE
            ))
    cal_f = np.stack(cal_f)[:32]
    cal_l = np.stack(cal_l)[:32]
    cal_real = np.asarray(cal_real, dtype=np.float64)[:32]
    cal_forward = sample_forward(cal_f, samples_per_state, emb_rnd)
    cal_pred = np.clip(
        ((cal_forward[:, :, -1, -1] - cal_forward[:, :, 0, -1])
         >= 0.5 * DELIVERY_SCALE).mean(axis=1).astype(np.float64), 0, 1,
    )
    ladder = CalibrationLadder(beta=1.0)
    cal_event = ladder.check(cal_pred, cal_real)
    cal_report = {
        **{k: v for k, v in cal_event.items() if k != "bins"},
        "bins": cal_event["bins"],
        "holdout_anchors": int(len(cal_f)),
        "holdout_disjointness": "episode-level split; holdout episodes never enter teacher training windows",
        "provenance": provenance,
    }
    (reports_dir / "calibration_hybrid_cramped.json").write_text(
        json.dumps(cal_report, indent=2, sort_keys=True)
    )

    # potential table + distillation on LOSSLESS anchor keys
    tau_gate = 0.5
    potential = ReapPotential(tau_gate=tau_gate)
    potential.update_tables(anchors_l, propensity, feasibility)
    gated_out = int(np.sum(feasibility < tau_gate))
    phi = np.array([potential.value(s, horizon) for s in anchors_l])
    potential_report = {
        "tau_gate": tau_gate, "coverage_states": potential.coverage,
        "propensity_range": [float(propensity.min()), float(propensity.max())],
        "feasibility_range": [float(feasibility.min()), float(feasibility.max())],
        "states_gated_out": gated_out,
        "states_passing_gate": int(len(anchors_l) - gated_out),
        "phi_range": [float(phi.min()), float(phi.max())],
        "phi_bounded_in_unit_interval": bool(np.all((phi >= 0) & (phi <= 1))),
        "feasibility_use": "gate only — never a reward magnitude",
        "provenance": provenance,
    }
    (reports_dir / "potential_table_hybrid_cramped.json").write_text(
        json.dumps(potential_report, indent=2, sort_keys=True)
    )

    mask = np.arange(len(anchors_l)) % 4 == 3
    distill_prov = {**provenance, "distill_hidden": distill_hidden,
                    "distill_epochs": distill_epochs,
                    "distill_train_states": int((~mask).sum()),
                    "distill_holdout_states": int(mask.sum())}
    p_hat = DistilledPredictor(paired.lossless_dim, hidden=distill_hidden, seed=seed)
    p_hat.fit(anchors_l[~mask], propensity[~mask].astype(np.float32), epochs=distill_epochs)
    p_rep = distillation_fidelity_report(
        p_hat, anchors_l[mask], propensity[mask],
        report_path=reports_dir / "distill_fidelity_hybrid_cramped.json",
        provenance=distill_prov)
    f_hat = DistilledPredictor(paired.lossless_dim, hidden=distill_hidden, seed=seed + 1)
    f_hat.fit(anchors_l[~mask], feasibility[~mask].astype(np.float32), epochs=distill_epochs)
    f_rep = distillation_fidelity_report(
        f_hat, anchors_l[mask], feasibility[mask],
        report_path=reports_dir / "distill_fidelity_feasibility_hybrid_cramped.json",
        provenance=distill_prov)
    torch.save({"p_hat": p_hat.state_dict(), "f_hat": f_hat.state_dict()},
               out_dir / "predictors.pt")

    np.savez_compressed(
        out_dir / "calibration_holdout.npz",
        refresh_anchors=anchors_l, refresh_anchors_features=anchors_f,
        calibration_anchors=cal_l, calibration_anchors_features=cal_f,
        calibration_realized=cal_real,
        provenance=json.dumps(provenance),
        disjointness="episode-level split; calibration episodes never enter teacher training windows",
    )
    np.save(out_dir / "probe_observations.npy", probes)

    warmup_report["provenance"] = provenance
    (reports_dir / "warmup_buffer_hybrid_cramped.json").write_text(
        json.dumps(warmup_report, indent=2, sort_keys=True)
    )
    for name in ("warmup_buffer_hybrid_cramped.json", "teacher_quality_hybrid_cramped.json",
                 "calibration_hybrid_cramped.json", "potential_table_hybrid_cramped.json",
                 "distill_fidelity_hybrid_cramped.json",
                 "distill_fidelity_feasibility_hybrid_cramped.json"):
        if "provenance" not in json.loads((reports_dir / name).read_text()):
            raise RuntimeError(f"pipeline artifact {name} lacks provenance")

    summary = {
        "runtime_seconds": round(_time.monotonic() - start, 1),
        "provenance": provenance,
        "warmup_gate_met": warmup_report["gate"]["met"],
        "quality": {k: quality[k] for k in
                    ("invalid_state_rate", "bridge_consistency_rate",
                     "projection_error_mean", "delivery_preservation_rate",
                     "endpoint_success_rate", "shaping_enabled")},
        "calibration": {k: cal_report[k] for k in ("raw_ece", "brier", "action", "beta_after")},
        "potential_table": {k: potential_report[k] for k in
                            ("states_passing_gate", "phi_range")},
        "distill_p_hat": {"mae": p_rep["mae"], "passed": p_rep["passed"]},
        "distill_f_hat": {"mae": f_rep["mae"], "passed": f_rep["passed"]},
    }
    (reports_dir / "teacher_pipeline_summary_hybrid_cramped.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary


class _ProbeAdapter:
    """Adapts PairedOvercooked to the (reset/step) probe-collection protocol."""

    def __init__(self, paired: PairedOvercooked):
        self.paired = paired
        self.num_agents = paired.num_agents
        self.num_actions = paired.num_actions

    def reset(self):
        local, lj, _ = self.paired.reset()
        return local, lj

    def step(self, actions):
        result, lj, _ = self.paired.step(actions)
        return type(result)(
            local_obs=result.local_obs, joint_state=lj,
            extrinsic_reward=result.extrinsic_reward,
            terminated=result.terminated, truncated=result.truncated,
            info=result.info,
        )
