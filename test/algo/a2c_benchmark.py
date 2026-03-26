"""
A2C-benchmark
=============

Advantage Actor-Critic (Mnih et al., 2016) applied to the MPE2
leader-follower environment.

A2C maintains both an actor (policy) and a critic (value function).
The policy gradient uses TD-based advantages:

    A(s_t, a_t) = r_t + γ V(s_{t+1}) - V(s_t)

summed via GAE-lambda for variance reduction.  Unlike PPO there is no
importance-ratio clipping; a single gradient-ascent step is applied per
rollout (no mini-batch epochs).

All three controllers (adversary_0, agent_0, agent_1) use independent A2C
policies trained inside the same shared environment interaction loop.

Usage
-----
::

    from algo.a2c_benchmark import A2CBenchmark
    results = A2CBenchmark.run(num_episodes=200, seed=0)
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
# A2C update
# ---------------------------------------------------------------------------

def _a2c_update(
    policy: GaussianMLP,
    obs_list: list,
    action_list: list,
    advantages: np.ndarray,
    lr: float = 3e-4,
    entropy_coeff: float = 0.01,
) -> None:
    """In-place A2C actor update.

    A single gradient-ascent step on the policy objective:

        ∇_θ J ≈ (1/T) Σ_t [ A_t · ∇_θ log π(a_t|s_t)
                             + β · ∇_θ H[π(·|s_t)] ]

    Parameters
    ----------
    policy        : GaussianMLP to update
    obs_list      : list of observations
    action_list   : list of actions
    advantages    : GAE advantages  (T,)
    lr            : actor learning rate
    entropy_coeff : entropy regularisation coefficient β
    """
    T = len(obs_list)
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    grad = np.zeros(policy.num_params(), dtype=np.float64)
    for i in range(T):
        lp_grad = policy.log_prob_grad(obs_list[i], action_list[i])
        grad += adv_norm[i] * lp_grad
        grad += entropy_coeff * policy.entropy_grad()
    grad /= T
    params = policy.get_flat_params()
    policy.set_flat_params(params + lr * grad)


# ---------------------------------------------------------------------------
# A2C controller
# ---------------------------------------------------------------------------

class A2CController(RLController):
    """Controller whose policy is updated via A2C.

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
        self.value_net = ValueMLP(obs_dim, hidden_dim=64, lr=1e-3, rng=np.random.default_rng(seed + 1))


# ---------------------------------------------------------------------------
# A2C Benchmark
# ---------------------------------------------------------------------------

class A2CBenchmark(BaseBenchmark):
    """Train and evaluate all agents with A2C.

    Parameters
    ----------
    num_good_agents : int
    gamma : float
        Discount factor.
    lam : float
        GAE lambda for advantage estimation.
    lr : float
        Actor learning rate.
    entropy_coeff : float
        Entropy regularisation coefficient β.
    """

    NAME = "A2C-benchmark"

    def __init__(
        self,
        num_good_agents: int = 2,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 3e-4,
        entropy_coeff: float = 0.01,
    ) -> None:
        super().__init__(num_good_agents=num_good_agents)
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.entropy_coeff = entropy_coeff

    def _build_controllers(self, seed: int) -> dict[str, A2CController]:
        return {
            ag: A2CController(obs_dim=dim, seed=seed + i * 100)
            for i, (ag, dim) in enumerate(AGENT_OBS_DIMS.items())
        }

    def train(
        self,
        num_episodes: int = 200,
        seed: int = 0,
        verbose: bool = False,
    ) -> dict[str, A2CController]:
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
                _a2c_update(
                    policy,
                    rollout.observations,
                    rollout.actions,
                    advantages,
                    lr=self.lr,
                    entropy_coeff=self.entropy_coeff,
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
    results = A2CBenchmark.run(num_episodes=100, num_eval_episodes=5, seed=0, verbose=True)
    print("mean_rewards:", results["mean_rewards"])
