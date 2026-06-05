"""COMA: counterfactual multi-agent policy gradients (Foerster et al. 2018).

Decentralized actors (parameter-shared MLP over local observation + agent
one-hot, as in the MAPPO trunk) with a centralized per-agent action-value
critic Q(s, a_i | u_{-i}, i). The actor gradient uses the counterfactual
advantage A_i = Q(s, u_i) - sum_u pi_i(u|o_i) Q(s, u), which marginalizes the
agent's own action out of the joint action-value while holding the other
agents' actions fixed.

On-policy: one rollout -> one critic regression pass (TD(lambda) targets from
a target critic) -> one actor gradient pass. The team extrinsic reward is the
only learning signal; episode metrics report the extrinsic channel exactly as
the MAPPO trunk does. Checkpoints carry every stateful component so resume is
trajectory-faithful.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import nn

from reap.algos.mappo import _mlp
from reap.envs.base import CoopEnv

DEFAULT_PARAMS = {
    "rollout_length": 256,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "gamma": 0.99,
    "td_lambda": 0.8,
    "entropy_coef": 0.01,
    "max_grad_norm": 10.0,
    "hidden_size": 128,
    "target_update_interval": 200,  # hard target-critic sync, in updates
    "episode_window": 100,
}


class ComaNets(nn.Module):
    """Shared actor + centralized counterfactual critic."""

    def __init__(self, local_obs_dim: int, joint_state_dim: int, num_agents: int,
                 num_actions: int, hidden: int):
        super().__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.actor = _mlp(local_obs_dim + num_agents, hidden, num_actions)
        # critic input: joint state, agent one-hot, other agents' actions one-hot
        critic_in = joint_state_dim + num_agents + (num_agents - 1) * num_actions
        self.critic = _mlp(critic_in, hidden, num_actions)

    def actor_input(self, local_obs: np.ndarray) -> torch.Tensor:
        obs = torch.as_tensor(np.asarray(local_obs), dtype=torch.float32)
        ident = torch.eye(self.num_agents, dtype=torch.float32)
        return torch.cat([obs, ident], dim=-1)

    def act(self, local_obs: np.ndarray, greedy: bool = False) -> np.ndarray:
        logits = self.actor(self.actor_input(local_obs))
        if greedy:
            return logits.argmax(dim=-1).numpy()
        return torch.distributions.Categorical(logits=logits).sample().numpy()

    def critic_input(self, joint: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Batch critic features for every agent: (T, N, critic_in).

        ``joint`` is (T, S); ``actions`` is (T, N) integer joint actions. For
        agent i the action block one-hot-encodes the OTHER agents' actions.
        """
        T, N, A = joint.shape[0], self.num_agents, self.num_actions
        onehot = torch.zeros(T, N, A)
        onehot.scatter_(-1, actions.unsqueeze(-1), 1.0)
        ident = torch.eye(N).expand(T, N, N)
        joint_rep = joint.unsqueeze(1).expand(T, N, joint.shape[-1])
        others = []
        for i in range(N):
            other_idx = [j for j in range(N) if j != i]
            others.append(onehot[:, other_idx, :].reshape(T, (N - 1) * A))
        others = torch.stack(others, dim=1)  # (T, N, (N-1)*A)
        return torch.cat([joint_rep, ident, others], dim=-1)


class ComaTrainer:
    """Rollout collection + COMA updates with trajectory-faithful checkpoints."""

    def __init__(self, env: CoopEnv, params: dict, seed: int):
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError(f"unknown coma params: {sorted(unknown)}")
        self.p = {**DEFAULT_PARAMS, **params}
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.nets = ComaNets(
            env.local_obs_dim, env.joint_state_dim, env.num_agents,
            env.num_actions, self.p["hidden_size"],
        )
        self.target_critic = _clone_module(self.nets.critic)
        self.actor_opt = torch.optim.Adam(self.nets.actor.parameters(), lr=self.p["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.nets.critic.parameters(), lr=self.p["critic_lr"])

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

    # -- rollout -------------------------------------------------------------

    def collect_rollout(self, max_steps: int | None = None) -> dict:
        """One on-policy rollout; truncated by ``max_steps`` for exact budgets."""
        T, N = self.p["rollout_length"], self.env.num_agents
        if max_steps is not None:
            T = min(T, int(max_steps))
        if T <= 0:
            raise ValueError(f"rollout length must be positive, got {T}")

        local_obs = np.empty((T, N, self.env.local_obs_dim), dtype=np.float32)
        joint_states = np.empty((T, self.env.joint_state_dim), dtype=np.float32)
        next_joints = np.empty_like(joint_states)
        next_actions = np.zeros((T, N), dtype=np.int64)
        actions = np.empty((T, N), dtype=np.int64)
        extrinsic = np.empty(T, dtype=np.float32)
        dones = np.empty(T, dtype=np.float32)

        for t in range(T):
            local_obs[t] = np.asarray(self._obs)
            joint_states[t] = self._joint
            with torch.no_grad():
                acts = self.nets.act(self._obs)
            actions[t] = acts
            result = self.env.step(acts.tolist())
            self.env_step += 1
            extrinsic[t] = result.extrinsic_reward
            done = result.terminated or result.truncated
            dones[t] = float(done)
            next_joints[t] = result.joint_state
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
            if not done:
                with torch.no_grad():
                    next_actions[t] = self.nets.act(self._obs)

        return {
            "local_obs": local_obs,
            "joint_states": joint_states,
            "next_joints": next_joints,
            "next_actions": next_actions,
            "actions": actions,
            "extrinsic": extrinsic,
            # logging-compat channels: COMA adds no shaping/intrinsic terms
            "intrinsic": np.zeros(T, dtype=np.float32),
            "shaping": np.zeros(T, dtype=np.float32),
            "dones": dones,
            "bonus_diag": {},
            "shaping_snapshot": None,
        }

    # -- learning --------------------------------------------------------------

    def _td_lambda_targets(self, rewards, dones, next_q_taken):
        """Backward TD(lambda) targets from target-critic bootstrap values."""
        T = len(rewards)
        lam, gamma = self.p["td_lambda"], self.gamma
        targets = np.zeros(T, dtype=np.float32)
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            one_step = rewards[t] + gamma * nonterminal * next_q_taken[t]
            if t == T - 1:
                targets[t] = one_step  # window edge: one-step bootstrap
            else:
                # G^lam_t = r + gamma[(1-lam) Q(s',u') + lam G^lam_{t+1}]
                targets[t] = one_step + gamma * lam * nonterminal * (
                    targets[t + 1] - next_q_taken[t]
                )
        return targets

    def update(self, rollout: dict) -> dict:
        T, N, A = len(rollout["extrinsic"]), self.env.num_agents, self.env.num_actions
        joint = torch.as_tensor(rollout["joint_states"])
        next_joint = torch.as_tensor(rollout["next_joints"])
        acts = torch.as_tensor(rollout["actions"])
        next_acts = torch.as_tensor(rollout["next_actions"])
        rewards = rollout["extrinsic"]
        dones = rollout["dones"]

        # bootstrap values: target critic evaluated at the NEXT joint action
        with torch.no_grad():
            next_in = self.nets.critic_input(next_joint, next_acts)
            next_q_all = self.target_critic(next_in)  # (T, N, A)
            next_q_taken = next_q_all.gather(-1, next_acts.unsqueeze(-1)).squeeze(-1)
            # team value: per-agent estimates agree in expectation; use agent 0
            next_q_team = next_q_taken[:, 0].numpy()
        targets = torch.as_tensor(self._td_lambda_targets(rewards, dones, next_q_team))

        # critic regression at taken actions, all agents
        critic_in = self.nets.critic_input(joint, acts)
        q_all = self.nets.critic(critic_in)  # (T, N, A)
        q_taken = q_all.gather(-1, acts.unsqueeze(-1)).squeeze(-1)  # (T, N)
        critic_loss = 0.5 * ((q_taken - targets.unsqueeze(-1)) ** 2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.nets.critic.parameters(), self.p["max_grad_norm"])
        self.critic_opt.step()

        # actor: counterfactual advantage from the refreshed critic
        obs = torch.as_tensor(rollout["local_obs"])
        ident = torch.eye(N, dtype=torch.float32).expand(T, N, N)
        logits = self.nets.actor(torch.cat([obs, ident], dim=-1))
        dist = torch.distributions.Categorical(logits=logits)
        with torch.no_grad():
            q_all_fresh = self.nets.critic(critic_in)  # (T, N, A)
        pi = dist.probs  # (T, N, A)
        baseline = (pi.detach() * q_all_fresh).sum(-1)  # (T, N)
        advantage = (
            q_all_fresh.gather(-1, acts.unsqueeze(-1)).squeeze(-1) - baseline
        )
        logp = dist.log_prob(acts)
        entropy = dist.entropy().mean()
        actor_loss = -(advantage.detach() * logp).mean() - self.p["entropy_coef"] * entropy
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.nets.actor.parameters(), self.p["max_grad_norm"])
        self.actor_opt.step()

        self.updates += 1
        if self.updates % self.p["target_update_interval"] == 0:
            self.target_critic.load_state_dict(self.nets.critic.state_dict())
        return {
            "policy_loss": float(actor_loss.item()),
            "value_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
            "advantage_abs_mean": float(advantage.abs().mean().item()),
        }

    # -- reporting / evaluation -------------------------------------------------

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
        """Greedy evaluation on a separate env (does not touch training state)."""
        return greedy_evaluate(lambda obs: self.nets.act(obs, greedy=True),
                               env, episodes, seed)

    # -- checkpointing (BL-20260603-resume-env-state: carry ALL state) -----------

    def state_dict(self) -> dict:
        return {
            "nets": self.nets.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
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
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
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


def _clone_module(module: nn.Module) -> nn.Module:
    import copy

    clone = copy.deepcopy(module)
    for param in clone.parameters():
        param.requires_grad_(False)
    return clone


def greedy_evaluate(policy_fn, env: CoopEnv, episodes: int, seed: int) -> dict:
    """Deterministic-policy evaluation episodes; extrinsic metrics only."""
    returns, succ = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return, done = 0.0, False
        while not done:
            with torch.no_grad():
                acts = policy_fn(obs)
            result = env.step(np.asarray(acts).tolist())
            ep_return += result.extrinsic_reward
            done = result.terminated or result.truncated
            obs = result.local_obs
        returns.append(ep_return)
        succ.append(float(result.info.get("success", False)))
    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_success_rate": float(np.mean(succ)),
        "eval_episodes": float(episodes),
    }
