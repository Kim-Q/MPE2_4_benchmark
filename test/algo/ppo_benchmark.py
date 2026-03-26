"""
PPO-benchmark
=============

Proximal Policy Optimization (Schulman et al., 2017) applied to the
MPE2 leader-follower environment.

All three controllers (adversary_0, agent_0, agent_1) use independent PPO
policies but are trained inside the same shared environment interaction loop.

Usage
-----
::

    from algo.ppo_benchmark import PPOBenchmark
    results = PPOBenchmark.run(num_episodes=200, seed=0)
    print(results["mean_rewards"])
"""

from __future__ import annotations

import sys
import os
import numpy as np

_API_DIR = os.path.join(os.path.dirname(__file__), "..", "api")
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from base import (  # noqa: E402
    GaussianMLP,
    ValueMLP,
    RLController,
    BaseBenchmark,
    AGENT_OBS_DIMS,
    compute_gae,
    collect_rollout,
)


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def _ppo_update(
    policy: GaussianMLP,
    obs_list: list,
    action_list: list,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    lr: float = 3e-4,
    clip_eps: float = 0.2,
    epochs: int = 4,
    minibatch: int = 32,
) -> None:
    """In-place PPO policy update using clipped surrogate objective.

    Parameters
    ----------
    policy        : GaussianMLP to update
    obs_list      : list of observations collected under the old policy
    action_list   : list of actions taken
    old_log_probs : log-probs under the old policy  (T,)
    advantages    : GAE advantages  (T,)
    lr            : learning rate
    clip_eps      : clipping radius ε
    epochs        : number of passes over the data
    minibatch     : mini-batch size
    """
    T = len(obs_list)
    # Normalise advantages
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        indices = np.random.permutation(T)
        for start in range(0, T, minibatch):
            batch_idx = indices[start: start + minibatch]
            grad = np.zeros(policy.num_params(), dtype=np.float64)

            for i in batch_idx:
                obs = obs_list[i]
                act = action_list[i]
                A_i = adv[i]
                old_lp = old_log_probs[i]

                new_lp = policy.log_prob(obs, act)
                ratio = np.exp(new_lp - old_lp)

                # Determine whether the clipped or unclipped term is active
                if A_i >= 0:
                    use_ratio = ratio if ratio <= 1.0 + clip_eps else 0.0
                else:
                    use_ratio = ratio if ratio >= 1.0 - clip_eps else 0.0

                if use_ratio != 0.0:
                    lp_grad = policy.log_prob_grad(obs, act)
                    grad += ratio * A_i * lp_grad

            grad /= max(len(batch_idx), 1)
            # Gradient ascent
            params = policy.get_flat_params()
            policy.set_flat_params(params + lr * grad)


# ---------------------------------------------------------------------------
# PPO controller
# ---------------------------------------------------------------------------

class PPOController(RLController):
    """Controller whose policy is updated via PPO.

    Parameters
    ----------
    obs_dim : int
        Observation dimension for this agent.
    seed : int
        Seed for the internal RNG.
    """

    def __init__(self, obs_dim: int, seed: int = 0) -> None:
        super().__init__(
            obs_dim=obs_dim,
            action_dim=2,
            hidden_dim=64,
            rng=np.random.default_rng(seed),
        )
        self.value_net = ValueMLP(obs_dim, hidden_dim=64, lr=1e-3, rng=np.random.default_rng(seed + 1))


# ---------------------------------------------------------------------------
# PPO Benchmark
# ---------------------------------------------------------------------------

class PPOBenchmark(BaseBenchmark):
    """Train and evaluate all agents with PPO.

    Parameters
    ----------
    num_good_agents : int
        Number of good (follower) agents (default 2).
    gamma : float
        Discount factor.
    lam : float
        GAE lambda.
    lr : float
        Policy learning rate.
    clip_eps : float
        PPO clipping radius.
    ppo_epochs : int
        Number of PPO update epochs per rollout.
    """

    NAME = "PPO-benchmark"

    def __init__(
        self,
        num_good_agents: int = 2,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
    ) -> None:
        super().__init__(num_good_agents=num_good_agents)
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs

    def _build_controllers(self, seed: int) -> dict[str, PPOController]:
        return {
            ag: PPOController(obs_dim=dim, seed=seed + i * 100)
            for i, (ag, dim) in enumerate(AGENT_OBS_DIMS.items())
        }

    def train(
        self,
        num_episodes: int = 200,
        seed: int = 0,
        verbose: bool = False,
    ) -> dict[str, PPOController]:
        """Train all controllers for ``num_episodes`` episodes."""
        controllers = self._build_controllers(seed)
        value_nets = {ag: controllers[ag].value_net for ag in controllers}
        episode_returns: list[dict] = []

        for ep in range(num_episodes):
            env = self._build_env(seed + ep)
            env.reset(seed=seed + ep)
            rollouts = collect_rollout(env, controllers, value_nets)
            env.close()

            ep_rewards: dict[str, float] = {}
            for ag, rollout in rollouts.items():
                if not rollout.rewards:
                    continue
                policy = controllers[ag].policy
                vnet = value_nets[ag]
                advantages, returns = compute_gae(
                    rollout.rewards, rollout.values, 0.0, self.gamma, self.lam,
                )
                _ppo_update(
                    policy,
                    rollout.observations,
                    rollout.actions,
                    np.array(rollout.log_probs, dtype=np.float64),
                    advantages,
                    lr=self.lr,
                    clip_eps=self.clip_eps,
                    epochs=self.ppo_epochs,
                )
                vnet.update(rollout.observations, returns, epochs=4)
                ep_rewards[ag] = float(np.sum(rollout.rewards))

            episode_returns.append(ep_rewards)
            if verbose and (ep + 1) % 20 == 0:
                avg = {ag: np.mean([r.get(ag, 0) for r in episode_returns[-20:]]) for ag in controllers}
                print(f"[{self.NAME}] ep {ep+1}/{num_episodes}  avg_reward={avg}")

        for ctrl in controllers.values():
            ctrl.deterministic = True
        return controllers


if __name__ == "__main__":
    results = PPOBenchmark.run(num_episodes=100, num_eval_episodes=5, seed=0, verbose=True)
    print("mean_rewards:", results["mean_rewards"])
