"""Multi-Agent PPO with centralized training and decentralized execution.

Actors are parameter-shared MLPs over per-agent local observations plus an
agent-identity one-hot; the critic is centralized over the wrapper's joint
state and predicts a single team value. Training follows PPO with GAE on the
team reward. An optional intrinsic bonus (RND / count-based) is added to the
reward used for advantage estimation only — episode metrics always report the
extrinsic (task) reward separately.

Episode horizon ends are treated as terminal for bootstrapping: Overcooked
episodes end exactly at the horizon, so there is no post-horizon return to
bootstrap toward.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import nn

from reap.algos.intrinsic import make_bonus
from reap.envs.base import CoopEnv

DEFAULT_PARAMS = {
    "rollout_length": 256,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "update_epochs": 4,
    "num_minibatches": 4,
    "hidden_size": 128,
    "intrinsic": "none",  # none | rnd | count
    "intrinsic_coef": 0.0,
    "shaping_potential": "",  # provider name from reap.shaping.hand; "" = off
    "shaping_beta": 0.0,
    "episode_window": 100,  # rolling window for reported episode stats
}


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class MappoNets(nn.Module):
    """Shared actor (local obs + agent one-hot) and centralized critic."""

    def __init__(self, local_obs_dim: int, joint_state_dim: int, num_agents: int,
                 num_actions: int, hidden: int):
        super().__init__()
        self.num_agents = num_agents
        self.actor = _mlp(local_obs_dim + num_agents, hidden, num_actions)
        self.critic = _mlp(joint_state_dim, hidden, 1)

    def actor_input(self, local_obs: np.ndarray) -> torch.Tensor:
        """Stack per-agent observations with identity one-hots: (N, obs+N)."""
        obs = torch.as_tensor(np.asarray(local_obs), dtype=torch.float32)
        ident = torch.eye(self.num_agents, dtype=torch.float32)
        return torch.cat([obs, ident], dim=-1)

    def act(self, local_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample one action per agent; returns (actions, log probs)."""
        logits = self.actor(self.actor_input(local_obs))
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        return actions.numpy(), dist.log_prob(actions).detach().numpy()

    def value(self, joint_state: np.ndarray) -> float:
        x = torch.as_tensor(np.asarray(joint_state), dtype=torch.float32)
        return float(self.critic(x).item())


class MappoTrainer:
    """Rollout collection + PPO updates with trajectory-faithful checkpoints."""

    def __init__(self, env: CoopEnv, params: dict, seed: int):
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError(f"unknown mappo params: {sorted(unknown)}")
        self.p = {**DEFAULT_PARAMS, **params}
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.nets = MappoNets(
            env.local_obs_dim, env.joint_state_dim, env.num_agents,
            env.num_actions, self.p["hidden_size"],
        )
        self.optimizer = torch.optim.Adam(self.nets.parameters(), lr=self.p["lr"])
        self.bonus = make_bonus(self.p["intrinsic"], env.joint_state_dim)
        self.potential_fn = None
        if self.p["shaping_potential"]:
            from reap.shaping.hand import make_hand_potential

            if self.p["shaping_beta"] < 0:
                raise ValueError("shaping_beta must be non-negative")
            self.potential_fn = make_hand_potential(self.p["shaping_potential"])
        elif self.p["shaping_beta"]:
            raise ValueError("shaping_beta set but no shaping_potential configured")

        self.env_step = 0
        self.updates = 0
        self.episodes = 0
        self.successes = 0
        self._ep_return = 0.0
        self._recent_returns: deque = deque(maxlen=self.p["episode_window"])
        self._recent_successes: deque = deque(maxlen=self.p["episode_window"])
        self._obs, self._joint = env.reset(seed=seed)

    @property
    def gamma(self) -> float:
        return self.p["gamma"]

    # -- rollout -----------------------------------------------------------

    def collect_rollout(self, max_steps: int | None = None) -> dict:
        """Collect one on-policy rollout; returns tensors for the update.

        ``max_steps`` truncates the rollout so a run can stop at exactly its
        configured environment-step budget instead of overshooting by up to
        one rollout length.
        """
        T, N = self.p["rollout_length"], self.env.num_agents
        if max_steps is not None:
            T = min(T, int(max_steps))
        if T <= 0:
            raise ValueError(f"rollout length must be positive, got {T}")
        local_obs = np.empty((T, N, self.env.local_obs_dim), dtype=np.float32)
        joint_states = np.empty((T, self.env.joint_state_dim), dtype=np.float32)
        next_joints = np.empty_like(joint_states)
        actions = np.empty((T, N), dtype=np.int64)
        logps = np.empty((T, N), dtype=np.float32)
        extrinsic = np.empty(T, dtype=np.float32)
        shaping = np.zeros(T, dtype=np.float32)
        dones = np.empty(T, dtype=np.float32)
        values = np.empty(T, dtype=np.float32)

        for t in range(T):
            local_obs[t] = np.asarray(self._obs)
            joint_states[t] = self._joint
            with torch.no_grad():
                acts, lps = self.nets.act(self._obs)
                values[t] = self.nets.value(self._joint)
            actions[t] = acts
            logps[t] = lps
            if self.potential_fn is not None:
                phi_s = self.potential_fn(self.env, self._joint, self.env.steps_remaining)
            result = self.env.step(acts.tolist())
            self.env_step += 1
            extrinsic[t] = result.extrinsic_reward
            done = result.terminated or result.truncated
            dones[t] = float(done)
            next_joints[t] = result.joint_state
            if self.potential_fn is not None:
                # potential at any episode end is zero (termination and timeout)
                phi_next = (
                    0.0
                    if done
                    else self.potential_fn(
                        self.env, result.joint_state, self.env.steps_remaining
                    )
                )
                shaping[t] = self.p["shaping_beta"] * (self.gamma * phi_next - phi_s)
            self._ep_return += result.extrinsic_reward

            if done:
                self.episodes += 1
                success = bool(result.info.get("success", False))
                self.successes += int(success)
                self._recent_returns.append(self._ep_return)
                self._recent_successes.append(float(success))
                self._ep_return = 0.0
                self._obs, self._joint = self.env.reset()
            else:
                self._obs, self._joint = result.local_obs, result.joint_state

        intrinsic = self.bonus.compute(joint_states) * self.p["intrinsic_coef"]
        bonus_diag = self.bonus.update(joint_states)
        with torch.no_grad():
            last_value = self.nets.value(self._joint)

        return {
            "local_obs": local_obs,
            "joint_states": joint_states,
            "actions": actions,
            "logps": logps,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic.astype(np.float32),
            "shaping": shaping,
            "dones": dones,
            "values": values,
            "last_value": last_value,
            "bonus_diag": bonus_diag,
        }

    # -- learning ----------------------------------------------------------

    def _gae(self, rewards, dones, values, last_value):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones[t]
            next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + self.gamma * self.p["gae_lambda"] * next_nonterminal * last_gae
            advantages[t] = last_gae
        return advantages, advantages + values

    def update(self, rollout: dict) -> dict:
        """One PPO update on a collected rollout; returns loss diagnostics."""
        T, N = len(rollout["extrinsic"]), self.env.num_agents
        rewards = rollout["extrinsic"] + rollout["intrinsic"] + rollout["shaping"]
        advantages, returns = self._gae(
            rewards, rollout["dones"], rollout["values"], rollout["last_value"]
        )

        obs = torch.as_tensor(rollout["local_obs"])  # (T, N, obs)
        ident = torch.eye(N, dtype=torch.float32).expand(T, N, N)
        actor_in = torch.cat([obs, ident], dim=-1)
        joint = torch.as_tensor(rollout["joint_states"])
        acts = torch.as_tensor(rollout["actions"])
        old_logps = torch.as_tensor(rollout["logps"])
        adv = torch.as_tensor(advantages).unsqueeze(-1).expand(T, N)
        ret = torch.as_tensor(returns)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = np.arange(T)
        batch_size = max(1, T // self.p["num_minibatches"])  # tiny final rollouts
        diags = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        steps = 0
        for _ in range(self.p["update_epochs"]):
            self.rng.shuffle(idx)
            for start in range(0, T, batch_size):
                mb = idx[start : start + batch_size]
                logits = self.nets.actor(actor_in[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logps = dist.log_prob(acts[mb])
                ratio = (new_logps - old_logps[mb]).exp()
                clipped = torch.clamp(
                    ratio, 1 - self.p["clip_coef"], 1 + self.p["clip_coef"]
                )
                policy_loss = -torch.min(ratio * adv[mb], clipped * adv[mb]).mean()
                value_pred = self.nets.critic(joint[mb]).squeeze(-1)
                value_loss = 0.5 * ((value_pred - ret[mb]) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = (
                    policy_loss
                    + self.p["value_coef"] * value_loss
                    - self.p["entropy_coef"] * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.nets.parameters(), self.p["max_grad_norm"])
                self.optimizer.step()
                diags["policy_loss"] += float(policy_loss.item())
                diags["value_loss"] += float(value_loss.item())
                diags["entropy"] += float(entropy.item())
                steps += 1
        self.updates += 1
        return {k: v / steps for k, v in diags.items()}

    # -- reporting ---------------------------------------------------------

    def episode_stats(self) -> dict:
        window_n = len(self._recent_returns)
        return {
            "episodes": float(self.episodes),
            "episode_return_mean": (
                float(np.mean(self._recent_returns)) if window_n else 0.0
            ),
            "success_rate": (
                float(np.mean(self._recent_successes)) if window_n else 0.0
            ),
        }

    # -- checkpointing (BL-20260603-resume-env-state: carry ALL state) ------

    def state_dict(self) -> dict:
        return {
            "nets": self.nets.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "bonus": self.bonus.state_dict(),
            "env_step": self.env_step,
            "updates": self.updates,
            "episodes": self.episodes,
            "successes": self.successes,
            "ep_return": self._ep_return,
            "recent_returns": list(self._recent_returns),
            "recent_successes": list(self._recent_successes),
            "obs": np.asarray(self._obs),
            "joint": self._joint,
            "env_snapshot": self.env.get_state(),
            "numpy_rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.nets.load_state_dict(state["nets"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.bonus.load_state_dict(state["bonus"])
        self.env_step = state["env_step"]
        self.updates = state["updates"]
        self.episodes = state["episodes"]
        self.successes = state["successes"]
        self._ep_return = state["ep_return"]
        self._recent_returns = deque(
            state["recent_returns"], maxlen=self.p["episode_window"]
        )
        self._recent_successes = deque(
            state["recent_successes"], maxlen=self.p["episode_window"]
        )
        self._obs = [row for row in np.asarray(state["obs"])]
        self._joint = state["joint"]
        self.env.set_state(state["env_snapshot"])
        self.rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
