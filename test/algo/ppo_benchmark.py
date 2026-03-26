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
    compute_gae,
    collect_rollout,
)
from env_api import build_custom_env, LandmarkConfig  # noqa: E402
from controller_api import action2d_to_simple_adversary_continuous  # noqa: E402


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
                clipped_ratio = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

                # Determine whether the clipped or unclipped term is active
                if A_i >= 0:
                    use_ratio = ratio if ratio <= 1.0 + clip_eps else 0.0
                else:
                    use_ratio = ratio if ratio >= 1.0 - clip_eps else 0.0

                if use_ratio != 0.0:
                    # Gradient of (ratio * A) = ratio * d(new_log_prob)/d(theta) * A
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

class PPOBenchmark:
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
        self.num_good_agents = num_good_agents
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs

    # obs_dim per agent role
    _OBS_DIMS = {"adversary_0": 8, "agent_0": 10, "agent_1": 10}

    def _build_controllers(self, seed: int) -> dict[str, PPOController]:
        return {
            ag: PPOController(obs_dim=dim, seed=seed + i * 100)
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
    ) -> dict[str, PPOController]:
        """Train all controllers for ``num_episodes`` episodes.

        Returns the trained controllers.
        """
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

                last_val = 0.0  # terminal bootstrap
                advantages, returns = compute_gae(
                    rollout.rewards, rollout.values, last_val,
                    self.gamma, self.lam,
                )

                old_log_probs = np.array(rollout.log_probs, dtype=np.float64)

                _ppo_update(
                    policy,
                    rollout.observations,
                    rollout.actions,
                    old_log_probs,
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

        # Switch to deterministic inference
        for ctrl in controllers.values():
            ctrl.deterministic = True

        return controllers

    def evaluate(
        self,
        controllers: dict[str, PPOController],
        seed: int = 9999,
        num_eval_episodes: int = 10,
    ) -> dict:
        """Evaluate trained controllers over several episodes."""
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
        """Train then evaluate all agents with PPO.

        Parameters
        ----------
        num_episodes : int
            Training episodes.
        num_eval_episodes : int
            Evaluation episodes after training.
        seed : int
            Master seed.
        verbose : bool
            Print progress during training.
        **kwargs
            Forwarded to the :class:`PPOBenchmark` constructor.

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
    results = PPOBenchmark.run(num_episodes=100, num_eval_episodes=5, seed=0, verbose=True)
    print("mean_rewards:", results["mean_rewards"])
