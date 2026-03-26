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
    collect_rollout,
)
from env_api import build_custom_env, LandmarkConfig  # noqa: E402
from controller_api import action2d_to_simple_adversary_continuous  # noqa: E402


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
    # Normalise returns (baseline)
    ret_norm = (returns - returns.mean()) / (returns.std() + 1e-8)

    grad = np.zeros(policy.num_params(), dtype=np.float64)

    for i in range(T):
        # Policy gradient term
        lp_grad = policy.log_prob_grad(obs_list[i], action_list[i])
        grad += ret_norm[i] * lp_grad

        # Entropy gradient term (independent of obs for diagonal Gaussian)
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

class EntropyRLBenchmark:
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
        self.num_good_agents = num_good_agents
        self.gamma = gamma
        self.lr = lr
        self.entropy_coeff = entropy_coeff

    _OBS_DIMS = {"adversary_0": 8, "agent_0": 10, "agent_1": 10}

    def _build_controllers(self, seed: int) -> dict[str, EntropyRLController]:
        return {
            ag: EntropyRLController(obs_dim=dim, seed=seed + i * 100)
            for i, (ag, dim) in enumerate(self._OBS_DIMS.items())
        }

    def _build_env(self, seed: int):
        lm_cfgs = [
            LandmarkConfig(position=np.array([0.6, 0.0]), name="goal_0"),
            LandmarkConfig(position=np.array([-0.6, 0.0]), name="goal_1"),
        ]
        follower_goal_weights = {
            "agent_0": np.array([0.8, 0.2]),
            "agent_1": np.array([0.2, 0.8]),
        }
        return build_custom_env(
            num_good_agents=self.num_good_agents,
            landmark_configs=lm_cfgs,
            follower_goal_weights=follower_goal_weights,
            max_cycles=25,
            continuous_actions=True,
            render_mode=None,
        )

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

            # EntropyRL does not use a value network (Monte Carlo returns)
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

    def evaluate(
        self,
        controllers: dict[str, EntropyRLController],
        seed: int = 9999,
        num_eval_episodes: int = 10,
    ) -> dict:
        """Evaluate trained controllers."""
        all_cum_rewards: list[dict] = []
        for ep in range(num_eval_episodes):
            env = self._build_env(seed + ep)
            env.reset(seed=seed + ep)
            cum_rewards = {ag: 0.0 for ag in env.possible_agents}
            for agent in env.agent_iter():
                obs, rew, term, trunc, _ = env.last()
                cum_rewards[agent] += rew
                if term or trunc:
                    env.step(None)
                else:
                    action = controllers[agent].get_action(obs)
                    env.step(action)
            env.close()
            all_cum_rewards.append(cum_rewards)

        mean_rewards = {
            ag: float(np.mean([r[ag] for r in all_cum_rewards]))
            for ag in all_cum_rewards[0]
        }
        return {"mean_rewards": mean_rewards, "all_episode_rewards": all_cum_rewards}

    @classmethod
    def run(
        cls,
        num_episodes: int = 200,
        num_eval_episodes: int = 10,
        seed: int = 0,
        verbose: bool = True,
        **kwargs,
    ) -> dict:
        """Train then evaluate all agents with EntropyRL.

        Parameters
        ----------
        num_episodes : int
        num_eval_episodes : int
        seed : int
        verbose : bool
        **kwargs
            Forwarded to :class:`EntropyRLBenchmark` constructor.

        Returns
        -------
        dict with keys ``"name"``, ``"controllers"``, ``"mean_rewards"``,
        ``"all_episode_rewards"``.
        """
        bench = cls(**kwargs)
        print(f"=== {bench.NAME}: training for {num_episodes} episodes ===")
        controllers = bench.train(num_episodes=num_episodes, seed=seed, verbose=verbose)
        print(f"=== {bench.NAME}: evaluating for {num_eval_episodes} episodes ===")
        eval_result = bench.evaluate(controllers, seed=seed + 10000, num_eval_episodes=num_eval_episodes)
        print(f"=== {bench.NAME}: mean_rewards = {eval_result['mean_rewards']} ===")
        return {
            "name": bench.NAME,
            "controllers": controllers,
            **eval_result,
        }


if __name__ == "__main__":
    results = EntropyRLBenchmark.run(num_episodes=100, num_eval_episodes=5, seed=0, verbose=True)
    print("mean_rewards:", results["mean_rewards"])
