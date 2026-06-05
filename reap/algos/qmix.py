"""QMIX: monotonic value factorisation (Rashid et al. 2018).

Per-agent utility network (parameter-shared MLP over local observation +
agent one-hot, matching the trunk's actor featurization) combined by a mixing
network whose weights are produced by hypernetworks conditioned on the
centralized joint state; absolute values on the mixing weights enforce
monotonicity. Training is off-policy DQN-style on a transition replay buffer
with epsilon-greedy exploration, double-Q target action selection, and a
periodically synced target network.

The team extrinsic reward is the only learning signal; episode metrics report
the extrinsic channel exactly as the MAPPO trunk does. Checkpoints carry every
stateful component — including the replay buffer and exploration state — so
resume is trajectory-faithful.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import nn

from reap.algos.coma import _clone_module, greedy_evaluate
from reap.algos.mappo import _mlp
from reap.envs.base import CoopEnv

DEFAULT_PARAMS = {
    "rollout_length": 256,  # env steps collected between update() calls
    "lr": 3e-4,
    "gamma": 0.99,
    "hidden_size": 128,
    "mixing_embed_dim": 32,
    "hypernet_hidden": 64,
    "buffer_capacity": 100_000,
    "batch_size": 128,
    "gradient_steps": 8,  # per update() call
    "min_buffer_size": 1_000,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_anneal_steps": 50_000,
    "target_update_interval": 200,  # hard target sync, in update() calls
    "max_grad_norm": 10.0,
    "episode_window": 100,
}


class QmixNets(nn.Module):
    """Shared per-agent utility net + state-conditioned monotonic mixer."""

    def __init__(self, local_obs_dim: int, joint_state_dim: int, num_agents: int,
                 num_actions: int, hidden: int, embed: int, hyper_hidden: int):
        super().__init__()
        self.num_agents = num_agents
        self.embed = embed
        self.agent = _mlp(local_obs_dim + num_agents, hidden, num_actions)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(joint_state_dim, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, num_agents * embed),
        )
        self.hyper_b1 = nn.Linear(joint_state_dim, embed)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(joint_state_dim, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, embed),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(joint_state_dim, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, 1),
        )

    def agent_input(self, local_obs: torch.Tensor) -> torch.Tensor:
        """(B, N, obs) -> (B, N, obs+N) with identity one-hots appended."""
        B, N = local_obs.shape[0], self.num_agents
        ident = torch.eye(N, dtype=torch.float32).expand(B, N, N)
        return torch.cat([local_obs, ident], dim=-1)

    def agent_qs(self, local_obs: torch.Tensor) -> torch.Tensor:
        """(B, N, obs) -> per-agent action values (B, N, A)."""
        return self.agent(self.agent_input(local_obs))

    def mix(self, chosen_qs: torch.Tensor, joint_state: torch.Tensor) -> torch.Tensor:
        """Monotonic mixing: (B, N) utilities + (B, S) state -> (B,) Q_tot."""
        B, N = chosen_qs.shape
        w1 = torch.abs(self.hyper_w1(joint_state)).view(B, N, self.embed)
        b1 = self.hyper_b1(joint_state).view(B, 1, self.embed)
        hidden = torch.nn.functional.elu(chosen_qs.unsqueeze(1) @ w1 + b1)
        w2 = torch.abs(self.hyper_w2(joint_state)).view(B, self.embed, 1)
        b2 = self.hyper_b2(joint_state).view(B, 1, 1)
        return (hidden @ w2 + b2).view(B)

    def act(self, local_obs: np.ndarray, greedy: bool = True) -> np.ndarray:
        obs = torch.as_tensor(np.asarray(local_obs), dtype=torch.float32).unsqueeze(0)
        return self.agent_qs(obs)[0].argmax(dim=-1).numpy()


class ReplayBuffer:
    """Fixed-capacity transition ring buffer with full state serialization."""

    def __init__(self, capacity: int, num_agents: int, obs_dim: int, state_dim: int):
        self.capacity = int(capacity)
        self.size = 0
        self.pos = 0
        self.obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros_like(self.state)
        self.actions = np.zeros((capacity, num_agents), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, obs, state, actions, reward, next_obs, next_state, done) -> None:
        i = self.pos
        self.obs[i] = obs
        self.state[i] = state
        self.actions[i] = actions
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.next_state[i] = next_state
        self.dones[i] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "state": torch.as_tensor(self.state[idx]),
            "actions": torch.as_tensor(self.actions[idx]),
            "rewards": torch.as_tensor(self.rewards[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "next_state": torch.as_tensor(self.next_state[idx]),
            "dones": torch.as_tensor(self.dones[idx]),
        }

    def state_dict(self) -> dict:
        n = self.size  # persist only the filled prefix (ring order irrelevant)
        return {
            "obs": self.obs[:n], "next_obs": self.next_obs[:n],
            "state": self.state[:n], "next_state": self.next_state[:n],
            "actions": self.actions[:n], "rewards": self.rewards[:n],
            "dones": self.dones[:n], "pos": self.pos, "size": n,
        }

    def load_state_dict(self, state: dict) -> None:
        n = int(state["size"])
        self.obs[:n] = state["obs"]
        self.next_obs[:n] = state["next_obs"]
        self.state[:n] = state["state"]
        self.next_state[:n] = state["next_state"]
        self.actions[:n] = state["actions"]
        self.rewards[:n] = state["rewards"]
        self.dones[:n] = state["dones"]
        self.size = n
        self.pos = int(state["pos"])


class QmixTrainer:
    """Collection + replay training with trajectory-faithful checkpoints."""

    def __init__(self, env: CoopEnv, params: dict, seed: int):
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError(f"unknown qmix params: {sorted(unknown)}")
        self.p = {**DEFAULT_PARAMS, **params}
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.nets = QmixNets(
            env.local_obs_dim, env.joint_state_dim, env.num_agents,
            env.num_actions, self.p["hidden_size"], self.p["mixing_embed_dim"],
            self.p["hypernet_hidden"],
        )
        self.target_nets = _clone_module(self.nets)
        self.optimizer = torch.optim.Adam(self.nets.parameters(), lr=self.p["lr"])
        self.buffer = ReplayBuffer(
            self.p["buffer_capacity"], env.num_agents, env.local_obs_dim,
            env.joint_state_dim,
        )

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

    @property
    def epsilon(self) -> float:
        """Linear anneal from eps_start to eps_end over eps_anneal_steps."""
        frac = min(1.0, self.env_step / max(1, self.p["eps_anneal_steps"]))
        return self.p["eps_start"] + frac * (self.p["eps_end"] - self.p["eps_start"])

    # -- rollout -----------------------------------------------------------------

    def _eps_greedy(self) -> np.ndarray:
        with torch.no_grad():
            greedy = self.nets.act(self._obs)
        explore = self.rng.random(self.env.num_agents) < self.epsilon
        random_acts = self.rng.integers(0, self.env.num_actions, self.env.num_agents)
        return np.where(explore, random_acts, greedy)

    def collect_rollout(self, max_steps: int | None = None) -> dict:
        """Collect transitions into the replay buffer; exact-budget truncation."""
        T = self.p["rollout_length"]
        if max_steps is not None:
            T = min(T, int(max_steps))
        if T <= 0:
            raise ValueError(f"rollout length must be positive, got {T}")

        extrinsic = np.empty(T, dtype=np.float32)
        for t in range(T):
            obs_now = np.asarray(self._obs, dtype=np.float32)
            joint_now = np.asarray(self._joint, dtype=np.float32)
            acts = self._eps_greedy()
            result = self.env.step(acts.tolist())
            self.env_step += 1
            extrinsic[t] = result.extrinsic_reward
            done = result.terminated or result.truncated
            self.buffer.push(
                obs_now, joint_now, acts, result.extrinsic_reward,
                np.asarray(result.local_obs, dtype=np.float32),
                np.asarray(result.joint_state, dtype=np.float32), float(done),
            )
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

        return {
            "extrinsic": extrinsic,
            # logging-compat channels: QMIX adds no shaping/intrinsic terms
            "intrinsic": np.zeros(T, dtype=np.float32),
            "shaping": np.zeros(T, dtype=np.float32),
            "bonus_diag": {"epsilon": float(self.epsilon)},
            "shaping_snapshot": None,
        }

    # -- learning -----------------------------------------------------------------

    def update(self, rollout: dict | None = None) -> dict:
        if self.buffer.size < self.p["min_buffer_size"]:
            return {"td_loss": 0.0, "q_tot_mean": 0.0, "skipped": 1.0}
        diags = {"td_loss": 0.0, "q_tot_mean": 0.0, "skipped": 0.0}
        for _ in range(self.p["gradient_steps"]):
            batch = self.buffer.sample(self.p["batch_size"], self.rng)
            qs = self.nets.agent_qs(batch["obs"])  # (B, N, A)
            chosen = qs.gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)
            q_tot = self.nets.mix(chosen, batch["state"])

            with torch.no_grad():
                # double-Q: online net selects, target net evaluates
                next_online = self.nets.agent_qs(batch["next_obs"])
                next_actions = next_online.argmax(dim=-1, keepdim=True)
                next_target = self.target_nets.agent_qs(batch["next_obs"])
                next_chosen = next_target.gather(-1, next_actions).squeeze(-1)
                next_q_tot = self.target_nets.mix(next_chosen, batch["next_state"])
                targets = batch["rewards"] + self.gamma * (1 - batch["dones"]) * next_q_tot

            loss = torch.nn.functional.mse_loss(q_tot, targets)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.nets.parameters(), self.p["max_grad_norm"])
            self.optimizer.step()
            diags["td_loss"] += float(loss.item())
            diags["q_tot_mean"] += float(q_tot.mean().item())

        self.updates += 1
        if self.updates % self.p["target_update_interval"] == 0:
            self.target_nets.load_state_dict(self.nets.state_dict())
        g = self.p["gradient_steps"]
        return {"td_loss": diags["td_loss"] / g, "q_tot_mean": diags["q_tot_mean"] / g,
                "skipped": 0.0}

    # -- reporting / evaluation ------------------------------------------------------

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

    def evaluate(self, env: CoopEnv, episodes: int, seed: int) -> dict:
        """Greedy (epsilon=0) evaluation on a separate env."""
        return greedy_evaluate(lambda obs: self.nets.act(obs), env, episodes, seed)

    # -- checkpointing (BL-20260603-resume-env-state: carry ALL state) --------------

    def state_dict(self) -> dict:
        return {
            "nets": self.nets.state_dict(),
            "target_nets": self.target_nets.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "buffer": self.buffer.state_dict(),
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
        self.target_nets.load_state_dict(state["target_nets"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.buffer.load_state_dict(state["buffer"])
        self.env_step = state["env_step"]
        self.updates = state["updates"]
        self.episodes = state["episodes"]
        self.successes = state["successes"]
        self._ep_return = state["ep_return"]
        self._recent_returns = deque(state["recent_returns"], maxlen=self.p["episode_window"])
        self._recent_successes = deque(state["recent_successes"], maxlen=self.p["episode_window"])
        self._obs = [row for row in np.asarray(state["obs"])]
        self._joint = state["joint"]
        self.env.set_state(state["env_snapshot"])
        self.rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
