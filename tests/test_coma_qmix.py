"""COMA and QMIX trunk tests: shapes, semantics, resume fidelity, budgets."""

import numpy as np
import pytest
import torch

from reap.algos.coma import ComaNets, ComaTrainer
from reap.algos.qmix import QmixNets, QmixTrainer, ReplayBuffer
from reap.config import (
    AlgoConfig,
    CheckpointConfig,
    Config,
    EnvConfig,
    LoggingConfig,
    RunConfig,
)
from reap.metrics import deterministic_view, read_jsonl
from tests.chain_env import ChainEnv

COMA_TINY = {"rollout_length": 32, "hidden_size": 32}
QMIX_TINY = {
    "rollout_length": 32, "hidden_size": 32, "mixing_embed_dim": 8,
    "hypernet_hidden": 16, "buffer_capacity": 2_000, "batch_size": 32,
    "gradient_steps": 2, "min_buffer_size": 64, "eps_anneal_steps": 500,
}


def make_cfg(tmp_path, algo, name="t", seed=0, total_steps=192, params=None,
             log_every=64, ckpt_every=96):
    base = COMA_TINY if algo == "coma" else QMIX_TINY
    return Config(
        run=RunConfig(
            name=name, seed=seed, mode="smoke", out_dir=str(tmp_path / "runs"),
            max_wall_clock_minutes=30.0, device="cpu",
        ),
        env=EnvConfig(id="mpe_spread", horizon=25, num_agents=3),
        algo=AlgoConfig(name=algo, total_env_steps=total_steps,
                        params={**base, **(params or {})}),
        logging=LoggingConfig(interval_env_steps=log_every),
        checkpoint=CheckpointConfig(interval_env_steps=ckpt_every, keep_last=2),
    )


def run_dir(cfg):
    from pathlib import Path

    return Path(cfg.run.out_dir) / cfg.run.name / f"seed{cfg.run.seed}"


# -- net-level semantics -------------------------------------------------------


def test_coma_critic_input_encodes_other_agents_actions():
    nets = ComaNets(local_obs_dim=4, joint_state_dim=6, num_agents=3,
                    num_actions=5, hidden=16)
    joint = torch.zeros(2, 6)
    actions = torch.tensor([[1, 2, 3], [0, 4, 0]])
    feats = nets.critic_input(joint, actions)
    assert feats.shape == (2, 3, 6 + 3 + 2 * 5)
    # agent 0's block holds one-hots of agents 1 and 2 (actions 2 and 3)
    block = feats[0, 0, 9:]
    expected = torch.zeros(10)
    expected[2] = 1.0  # agent 1 took action 2
    expected[5 + 3] = 1.0  # agent 2 took action 3
    assert torch.equal(block, expected)
    # the one-hot block never contains the agent's OWN action
    own = feats[0, 0, 9:].view(2, 5)
    assert own.sum() == 2.0


def test_coma_counterfactual_baseline_is_policy_expectation():
    env = ChainEnv(4, horizon=6)
    trainer = ComaTrainer(env, dict(COMA_TINY), seed=0)
    rollout = trainer.collect_rollout()
    joint = torch.as_tensor(rollout["joint_states"])
    acts = torch.as_tensor(rollout["actions"])
    critic_in = trainer.nets.critic_input(joint, acts)
    q_all = trainer.nets.critic(critic_in)
    obs = torch.as_tensor(rollout["local_obs"])
    T, N = acts.shape
    ident = torch.eye(N).expand(T, N, N)
    pi = torch.distributions.Categorical(
        logits=trainer.nets.actor(torch.cat([obs, ident], dim=-1))
    ).probs
    baseline = (pi * q_all).sum(-1)
    # hand-check one entry
    t, i = 3, 1
    manual = sum(float(pi[t, i, a]) * float(q_all[t, i, a]) for a in range(2))
    assert float(baseline[t, i]) == pytest.approx(manual, rel=1e-5)


def test_qmix_mixer_is_monotone_in_agent_utilities():
    nets = QmixNets(local_obs_dim=4, joint_state_dim=6, num_agents=3,
                    num_actions=5, hidden=16, embed=8, hyper_hidden=16)
    state = torch.randn(4, 6)
    qs = torch.randn(4, 3, requires_grad=True)
    q_tot = nets.mix(qs, state).sum()
    q_tot.backward()
    assert (qs.grad >= 0).all()  # abs() hypernet weights guarantee monotonicity


def test_qmix_replay_buffer_roundtrip_and_ring():
    buf = ReplayBuffer(capacity=4, num_agents=2, obs_dim=3, state_dim=5)
    for k in range(6):  # overfill to exercise the ring
        buf.push(np.full((2, 3), k), np.full(5, k), [k % 2, k % 2], float(k),
                 np.full((2, 3), k + 1), np.full(5, k + 1), 0.0)
    assert buf.size == 4 and buf.pos == 2
    state = buf.state_dict()
    buf2 = ReplayBuffer(capacity=4, num_agents=2, obs_dim=3, state_dim=5)
    buf2.load_state_dict(state)
    assert buf2.size == 4 and buf2.pos == 2
    assert np.array_equal(buf2.rewards[:4], buf.rewards[:4])
    batch = buf2.sample(8, np.random.default_rng(0))
    assert batch["obs"].shape == (8, 2, 3)


def test_unknown_params_rejected():
    env = ChainEnv()
    with pytest.raises(ValueError, match="unknown coma params"):
        ComaTrainer(env, {"nope": 1}, seed=0)
    with pytest.raises(ValueError, match="unknown qmix params"):
        QmixTrainer(env, {"nope": 1}, seed=0)


# -- runner-level protocol semantics ---------------------------------------------


@pytest.mark.parametrize("algo", ["coma", "qmix"])
def test_runner_smoke_and_extrinsic_channel_purity(tmp_path, algo):
    from reap.train import run_from_config

    cfg = make_cfg(tmp_path, algo, total_steps=160)
    summary = run_from_config(cfg)
    assert summary["env_step"] == 160
    records = read_jsonl(run_dir(cfg) / "metrics.jsonl")
    assert records[-1]["env_step"] == 160  # exact-budget final record
    for rec in records:
        # COMA/QMIX add no shaping or intrinsic terms: channels stay zero
        assert rec["shaped"]["term_mean"] == 0.0
        assert rec["intrinsic"]["bonus_mean"] == 0.0
        assert np.isfinite(rec["extrinsic"]["episode_return_mean"])


@pytest.mark.parametrize("algo", ["coma", "qmix"])
def test_runner_stops_exactly_at_non_divisible_budget(tmp_path, algo):
    from reap.train import run_from_config

    cfg = make_cfg(tmp_path, algo, total_steps=70, log_every=64, ckpt_every=64)
    summary = run_from_config(cfg)
    assert summary["env_step"] == 70
    records = read_jsonl(run_dir(cfg) / "metrics.jsonl")
    assert records[-1]["env_step"] == 70


@pytest.mark.parametrize("algo", ["coma", "qmix"])
def test_runner_periodic_eval_lands_in_extrinsic_channel(tmp_path, algo):
    from reap.train import run_from_config

    cfg = make_cfg(tmp_path, algo, total_steps=128,
                   params={"eval_interval_env_steps": 64, "eval_episodes": 2})
    summary = run_from_config(cfg)
    assert "eval_return_mean" in summary
    records = read_jsonl(run_dir(cfg) / "metrics.jsonl")
    final = records[-1]["extrinsic"]
    assert "eval_return_mean" in final and "eval_success_rate" in final
    assert final["eval_episodes"] == 2.0


@pytest.mark.parametrize("algo", ["coma", "qmix"])
def test_same_seed_determinism(tmp_path, algo):
    from reap.train import run_from_config

    cfg_a = make_cfg(tmp_path, algo, name="a", total_steps=96)
    cfg_b = make_cfg(tmp_path, algo, name="b", total_steps=96)
    run_from_config(cfg_a)
    run_from_config(cfg_b)
    rec_a = read_jsonl(run_dir(cfg_a) / "metrics.jsonl")
    rec_b = read_jsonl(run_dir(cfg_b) / "metrics.jsonl")
    assert deterministic_view(rec_a) == deterministic_view(rec_b)


@pytest.mark.parametrize("algo", ["coma", "qmix"])
def test_resume_is_trajectory_faithful_mid_episode(tmp_path, algo):
    """Interrupted+resumed must equal uninterrupted (BL-20260603-resume-env-state).

    The interruption point (128) is deliberately not aligned to the episode
    horizon (25), so the resume checkpoint lands mid-episode.
    """
    import dataclasses

    from reap.train import run_from_config

    full = make_cfg(tmp_path, algo, name="full", total_steps=192, ckpt_every=96)
    run_from_config(full)

    part = make_cfg(tmp_path, algo, name="part", total_steps=128, ckpt_every=96)
    run_from_config(part)
    resumed = dataclasses.replace(
        part, algo=dataclasses.replace(part.algo, total_env_steps=192)
    )
    run_from_config(resumed, resume=True)

    rec_full = read_jsonl(run_dir(full) / "metrics.jsonl")
    rec_part = read_jsonl(run_dir(part) / "metrics.jsonl")
    full_view = deterministic_view(rec_full)
    part_view = deterministic_view(rec_part)
    assert part_view == full_view[: len(part_view)]


@pytest.mark.parametrize("trainer_cls,base", [(ComaTrainer, COMA_TINY),
                                              (QmixTrainer, QMIX_TINY)])
def test_standardise_rewards_state_in_checkpoint(trainer_cls, base):
    from reap.envs.mpe_spread import MpeSpreadEnv

    env = MpeSpreadEnv()
    trainer = trainer_cls(env, {**base, "standardise_rewards": True}, seed=0)
    trainer.update(trainer.collect_rollout())
    state = trainer.state_dict()
    assert state["reward_rms"] is not None
    assert state["reward_rms"]["count"] > 1  # statistics actually accumulated
    env2 = MpeSpreadEnv()
    restored = trainer_cls(env2, {**base, "standardise_rewards": True}, seed=1)
    restored.load_state_dict(state)
    assert restored.reward_rms.mean == pytest.approx(trainer.reward_rms.mean)
    # metrics stay on the raw extrinsic scale (standardization is target-only)
    assert trainer.episode_stats()["episode_return_mean"] < -50.0


def test_qmix_learns_chain_coordination():
    """Sanity: QMIX solves the 2-agent coordination chain from sparse reward."""
    torch.manual_seed(0)
    env = ChainEnv(3, horizon=8)
    trainer = QmixTrainer(env, {**QMIX_TINY, "eps_anneal_steps": 1_500,
                                "gradient_steps": 4, "lr": 1e-3,
                                "target_update_interval": 10}, seed=0)
    for _ in range(120):
        trainer.update(trainer.collect_rollout())
    assert trainer.episode_stats()["success_rate"] > 0.8


def test_coma_learns_chain_coordination():
    """Sanity: COMA improves the 2-agent coordination chain from sparse reward."""
    torch.manual_seed(0)
    env = ChainEnv(3, horizon=8)
    trainer = ComaTrainer(env, {**COMA_TINY, "rollout_length": 64,
                                "actor_lr": 1e-3, "critic_lr": 1e-3,
                                "target_update_interval": 20}, seed=0)
    for _ in range(150):
        trainer.update(trainer.collect_rollout())
    assert trainer.episode_stats()["success_rate"] > 0.5
