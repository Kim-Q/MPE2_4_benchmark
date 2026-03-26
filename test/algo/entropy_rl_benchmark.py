"""
EntropyRL-benchmark
===================

Policy gradient with entropy regularization applied to the MPE2
leader-follower environment.

The objective maximized is:

    J(θ) = E_π[Σ_t γ^t r_t]  +  β · H[π]

where H[π] is the differential entropy of the Gaussian policy.
The entropy bonus encourages exploration and prevents premature convergence.

All three controllers (adversary_0, agent_0, agent_1) use independent
EntropyRL policies trained inside the same shared environment loop.

Usage
-----
::

    from algo.entropy_rl_benchmark import EntropyRLBenchmark
    results = EntropyRLBenchmark.run(num_episodes=200, seed=0)
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
    RLController,
    BaseBenchmark,
    AGENT_OBS_DIMS,
    collect_rollout,
)


# ---------------------------------------------------------------------------
# Entropy RL update (REINFORCE + entropy bonus)
# ---------------------------------------------------------------------------

def _entropy_rl_update(
    policy: GaussianMLP,
    obs_list: list,
    action_list: list,
    returns: np.ndarray,
    lr: float = 3e-4,
    entropy_coeff: float = 0.01,
) -> None:
    """In-place Entropy RL update using REINFORCE with entropy regularization.

    Gradient of the objective:

        ∇_θ J = (1/T) Σ_t [ G_t · ∇_θ log π(a_t|s_t)
                             + β · ∇_θ H[π(·|s_t)] ]

    Parameters
    ----------
    policy        : GaussianMLP to update
    obs_list      : list of observations
    action_list   : list of actions
    returns       : discounted returns G_t  (T,)
    lr            : learning rate
    entropy_coeff : entropy regularisation coefficient β
    """
    T = len(obs_list)
    ret_norm = (returns - returns.mean()) / (returns.std() + 1e-8)
    grad = np.zeros(policy.num_params(), dtype=np.float64)
    for i in range(T):
        lp_grad = policy.log_prob_grad(obs_list[i], action_list[i])
        grad += ret_norm[i] * lp_grad
        grad += entropy_coeff * policy.entropy_grad()
    grad /= T
    params = policy.get_flat_params()
    policy.set_flat_params(params + lr * grad)


def _compute_returns(rewards: list, gamma: float = 0.99) -> np.ndarray:
    """Compute Monte Carlo discounted returns G_t = Σ_{k≥t} γ^{k-t} r_k."""
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float64)
    g = 0.0
    for t in reversed(range(T)):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns


# ---------------------------------------------------------------------------
# Entropy RL controller
# ---------------------------------------------------------------------------

class EntropyRLController(RLController):
    """Controller whose policy is updated via REINFORCE + entropy bonus.

    Parameters
    ----------
    obs_dim : int
    seed : int
    """

    def __init__(self, obs_dim: int, seed: int = 0) -> None:
        super().__init__(
            obs_dim=obs_dim,
            action_dim=2,
            hidden_dim=64,
            rng=np.random.default_rng(seed),
        )


# ---------------------------------------------------------------------------
# EntropyRL Benchmark
# ---------------------------------------------------------------------------

class EntropyRLBenchmark(BaseBenchmark):
    """Train and evaluate all agents with EntropyRL (REINFORCE + entropy bonus).

    Parameters
    ----------
    num_good_agents : int
    gamma : float
        Discount factor.
    lr : float
        Policy learning rate.
    entropy_coeff : float
        Entropy regularization coefficient β.
    """

    NAME = "EntropyRL-benchmark"

    def __init__(
        self,
        num_good_agents: int = 2,
        gamma: float = 0.99,
        lr: float = 3e-4,
        entropy_coeff: float = 0.01,
    ) -> None:
        super().__init__(num_good_agents=num_good_agents)
        self.gamma = gamma
        self.lr = lr
        self.entropy_coeff = entropy_coeff

    def _build_controllers(self, seed: int) -> dict[str, EntropyRLController]:
        return {
            ag: EntropyRLController(obs_dim=dim, seed=seed + i * 100)
            for i, (ag, dim) in enumerate(AGENT_OBS_DIMS.items())
        }

    def train(
        self,
        num_episodes: int = 200,
        seed: int = 0,
        verbose: bool = False,
    ) -> dict[str, EntropyRLController]:
        """Train all controllers for ``num_episodes`` episodes."""
        controllers = self._build_controllers(seed)
        episode_returns: list[dict] = []

        for ep in range(num_episodes):
            env = self._build_env(seed + ep)
            env.reset(seed=seed + ep)
            rollouts = collect_rollout(env, controllers, value_nets=None)
            env.close()

            ep_rewards: dict[str, float] = {}
            for ag, rollout in rollouts.items():
                if not rollout.rewards:
                    continue
                policy = controllers[ag].policy
                returns = _compute_returns(rollout.rewards, self.gamma)
                _entropy_rl_update(
                    policy,
                    rollout.observations,
                    rollout.actions,
                    returns,
                    lr=self.lr,
                    entropy_coeff=self.entropy_coeff,
                )
                ep_rewards[ag] = float(np.sum(rollout.rewards))

            episode_returns.append(ep_rewards)
            if verbose and (ep + 1) % 20 == 0:
                avg = {ag: np.mean([r.get(ag, 0) for r in episode_returns[-20:]]) for ag in controllers}
                print(f"[{self.NAME}] ep {ep+1}/{num_episodes}  avg_reward={avg}")

        for ctrl in controllers.values():
            ctrl.deterministic = True
        return controllers


if __name__ == "__main__":
    results = EntropyRLBenchmark.run(num_episodes=100, num_eval_episodes=5, seed=0, verbose=True)
    print("mean_rewards:", results["mean_rewards"])
