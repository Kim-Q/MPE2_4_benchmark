"""
TRPO-benchmark
==============

Trust Region Policy Optimization (Schulman et al., 2015) applied to the
MPE2 leader-follower environment.

All three controllers (adversary_0, agent_0, agent_1) use independent TRPO
policies trained inside the same shared environment interaction loop.

TRPO uses the natural policy gradient (conjugate gradient + backtracking line
search) to take the largest step that keeps the KL divergence below a
threshold ``delta``.

Usage
-----
::

    from algo.trpo_benchmark import TRPOBenchmark
    results = TRPOBenchmark.run(num_episodes=200, seed=0)
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
# TRPO helpers
# ---------------------------------------------------------------------------

def _policy_gradient(
    policy: GaussianMLP,
    obs_list: list,
    action_list: list,
    advantages: np.ndarray,
) -> np.ndarray:
    """Compute the policy gradient ∇_θ E[log π(a|s) · A] (mean over samples)."""
    T = len(obs_list)
    grad = np.zeros(policy.num_params(), dtype=np.float64)
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    for i in range(T):
        lp_grad = policy.log_prob_grad(obs_list[i], action_list[i])
        grad += adv_norm[i] * lp_grad
    return grad / T


def _fisher_vector_product(
    policy: GaussianMLP,
    obs_list: list,
    vector: np.ndarray,
    damping: float = 0.1,
) -> np.ndarray:
    """Approximate Fisher-vector product F·v using the empirical Fisher.

    F·v ≈ (1/T) Σ_t  (∇ log π_t) · (∇ log π_t)^T · v
    """
    T = len(obs_list)
    Fv = np.zeros_like(vector, dtype=np.float64)
    for i in range(T):
        j_i = policy.log_prob_grad(obs_list[i], _action_cache[i])
        Fv += np.dot(j_i, vector) * j_i
    Fv /= T
    Fv += damping * vector  # Tikhonov damping
    return Fv


# Thread-local cache for action list inside Fisher-vector product
_action_cache: list = []


def _conjugate_gradient(
    Av_fn,
    b: np.ndarray,
    cg_iters: int = 10,
    residual_tol: float = 1e-10,
) -> np.ndarray:
    """Solve Ax = b using the conjugate gradient method."""
    x = np.zeros_like(b)
    r = b.copy()
    p = b.copy()
    rdotr = np.dot(r, r)
    for _ in range(cg_iters):
        Ap = Av_fn(p)
        alpha = rdotr / (np.dot(p, Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = np.dot(r, r)
        if new_rdotr < residual_tol:
            break
        p = r + (new_rdotr / (rdotr + 1e-8)) * p
        rdotr = new_rdotr
    return x


def _surrogate_loss(
    policy: GaussianMLP,
    obs_list: list,
    action_list: list,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
) -> float:
    """Compute the surrogate objective L = E[r(θ) * A]."""
    T = len(obs_list)
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    total = 0.0
    for i in range(T):
        new_lp = policy.log_prob(obs_list[i], action_list[i])
        ratio = np.exp(new_lp - old_log_probs[i])
        total += ratio * adv_norm[i]
    return total / T


def _kl_divergence(
    policy: GaussianMLP,
    old_means: list,
    old_stds: list,
) -> float:
    """Average KL divergence KL(old || new) for Diagonal Gaussians."""
    total_kl = 0.0
    T = len(old_means)
    new_std = np.exp(policy.log_std)
    for i in range(T):
        mu1 = old_means[i]
        s1 = old_stds[i]
        mu2, _ = policy.forward(policy._obs_cache[i])
        kl = np.sum(
            np.log(new_std / s1)
            + (s1 ** 2 + (mu1 - mu2) ** 2) / (2.0 * new_std ** 2)
            - 0.5
        )
        total_kl += kl
    return total_kl / T


def _trpo_update(
    policy: GaussianMLP,
    obs_list: list,
    action_list: list,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    delta: float = 0.01,
    cg_iters: int = 10,
    backtrack_iters: int = 10,
    backtrack_coeff: float = 0.8,
    damping: float = 0.1,
) -> None:
    """In-place TRPO policy update.

    Steps:
    1. Compute policy gradient g.
    2. Compute the natural gradient direction via CG:  d = F⁻¹ g.
    3. Rescale d so that 0.5 d^T F d = δ.
    4. Back-tracking line search on the surrogate to find the largest
       step that does not violate the KL constraint.
    """
    global _action_cache
    _action_cache = action_list

    # Capture old means and stds (before update)
    old_means = []
    old_stds = []
    for obs in obs_list:
        m, s = policy.forward(obs)
        old_means.append(m.copy())
        old_stds.append(s.copy())
    policy._obs_cache = obs_list

    # 1. Policy gradient
    g = _policy_gradient(policy, obs_list, action_list, advantages)

    # 2. Natural gradient via CG
    def fvp(v):
        return _fisher_vector_product(policy, obs_list, v, damping=damping)

    nat_grad = _conjugate_gradient(fvp, g, cg_iters=cg_iters)

    # 3. Scale to satisfy KL budget
    sHs = float(np.dot(nat_grad, fvp(nat_grad)))
    if sHs <= 0.0:
        return  # degenerate step, skip
    scale = np.sqrt(2.0 * delta / (sHs + 1e-8))
    full_step = scale * nat_grad

    # 4. Back-tracking line search
    old_params = policy.get_flat_params().copy()
    old_loss = _surrogate_loss(policy, obs_list, action_list, old_log_probs, advantages)

    for k in range(backtrack_iters):
        step_size = backtrack_coeff ** k
        new_params = old_params + step_size * full_step
        policy.set_flat_params(new_params)

        new_loss = _surrogate_loss(policy, obs_list, action_list, old_log_probs, advantages)
        kl = _kl_divergence(policy, old_means, old_stds)

        if new_loss > old_loss and kl <= delta:
            return  # accepted

    # Revert if no acceptable step was found
    policy.set_flat_params(old_params)


# ---------------------------------------------------------------------------
# TRPO controller
# ---------------------------------------------------------------------------

class TRPOController(RLController):
    """Controller whose policy is updated via TRPO.

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
# TRPO Benchmark
# ---------------------------------------------------------------------------

class TRPOBenchmark(BaseBenchmark):
    """Train and evaluate all agents with TRPO.

    Parameters
    ----------
    num_good_agents : int
    gamma : float
    lam : float
    delta : float
        KL trust-region radius.
    cg_iters : int
        Conjugate gradient iterations.
    damping : float
        Fisher matrix damping coefficient.
    """

    NAME = "TRPO-benchmark"

    def __init__(
        self,
        num_good_agents: int = 2,
        gamma: float = 0.99,
        lam: float = 0.95,
        delta: float = 0.01,
        cg_iters: int = 10,
        damping: float = 0.1,
    ) -> None:
        super().__init__(num_good_agents=num_good_agents)
        self.gamma = gamma
        self.lam = lam
        self.delta = delta
        self.cg_iters = cg_iters
        self.damping = damping

    def _build_controllers(self, seed: int) -> dict[str, TRPOController]:
        return {
            ag: TRPOController(obs_dim=dim, seed=seed + i * 100)
            for i, (ag, dim) in enumerate(AGENT_OBS_DIMS.items())
        }

    def train(
        self,
        num_episodes: int = 200,
        seed: int = 0,
        verbose: bool = False,
    ) -> dict[str, TRPOController]:
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
                _trpo_update(
                    policy,
                    rollout.observations,
                    rollout.actions,
                    np.array(rollout.log_probs, dtype=np.float64),
                    advantages,
                    delta=self.delta,
                    cg_iters=self.cg_iters,
                    damping=self.damping,
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
    results = TRPOBenchmark.run(num_episodes=100, num_eval_episodes=5, seed=0, verbose=True)
    print("mean_rewards:", results["mean_rewards"])
