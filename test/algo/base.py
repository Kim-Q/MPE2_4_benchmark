"""
Shared building blocks for numpy-based RL algorithms.

Classes
-------
GaussianMLP
    Two-layer MLP Gaussian policy with manual forward pass, log-prob,
    entropy, and gradient computation.
ValueMLP
    Two-layer MLP value function used by PPO and TRPO critics.
RLController
    Subclass of :class:`BaseAgentController` that wraps a ``GaussianMLP``
    and implements :meth:`get_action_2d`.
Rollout
    Named-tuple holding one collected trajectory.

Helper functions
----------------
compute_gae   – Generalized Advantage Estimation (GAE-lambda).
collect_rollout – Run one episode and return a ``Rollout``.
"""

from __future__ import annotations

import sys
import os
import numpy as np
from typing import NamedTuple

# Allow importing controller_api from sibling api/ directory when running
# scripts directly from test/algo/
_API_DIR = os.path.join(os.path.dirname(__file__), "..", "api")
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from controller_api import BaseAgentController  # noqa: E402


# ---------------------------------------------------------------------------
# Neural network building blocks
# ---------------------------------------------------------------------------

class GaussianMLP:
    """Diagonal-Gaussian policy implemented as a two-layer MLP.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation input.
    action_dim : int
        Dimensionality of the 2-D action output (default 2).
    hidden_dim : int
        Number of hidden units (default 64).
    rng : np.random.Generator | None
        Random number generator for weight initialisation and sampling.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 64,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        rng = rng if rng is not None else np.random.default_rng(0)

        # He (Kaiming) initialization
        s1 = np.sqrt(2.0 / obs_dim)
        s2 = np.sqrt(2.0 / hidden_dim)
        self.W1: np.ndarray = (rng.standard_normal((hidden_dim, obs_dim)) * s1).astype(np.float64)
        self.b1: np.ndarray = np.zeros(hidden_dim, dtype=np.float64)
        self.W2: np.ndarray = (rng.standard_normal((action_dim, hidden_dim)) * s2).astype(np.float64)
        self.b2: np.ndarray = np.zeros(action_dim, dtype=np.float64)
        # Learnable log-std (initialized to log(0.5))
        self.log_std: np.ndarray = np.full(action_dim, np.log(0.5), dtype=np.float64)

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def get_flat_params(self) -> np.ndarray:
        """Return all parameters as a 1-D array."""
        return np.concatenate([
            self.W1.ravel(), self.b1.ravel(),
            self.W2.ravel(), self.b2.ravel(),
            self.log_std.ravel(),
        ])

    def set_flat_params(self, params: np.ndarray) -> None:
        """Set all parameters from a 1-D array (in the same order as ``get_flat_params``)."""
        idx = 0
        for attr, arr in [
            ("W1", self.W1), ("b1", self.b1),
            ("W2", self.W2), ("b2", self.b2),
            ("log_std", self.log_std),
        ]:
            size = arr.size
            getattr(self, attr)[:] = params[idx: idx + size].reshape(arr.shape)
            idx += size

    def num_params(self) -> int:
        return self.get_flat_params().size

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward_internals(self, obs: np.ndarray):
        """Return (pre_h, h, pre_mean, mean, std)."""
        obs = np.asarray(obs, dtype=np.float64).ravel()
        pre_h = self.W1 @ obs + self.b1          # (hidden_dim,)
        h = np.tanh(pre_h)                        # (hidden_dim,)
        pre_mean = self.W2 @ h + self.b2          # (action_dim,)
        mean = np.tanh(pre_mean)                  # (action_dim,)  in (-1, 1)
        std = np.exp(self.log_std)                # (action_dim,)
        return obs, pre_h, h, pre_mean, mean, std

    def forward(self, obs: np.ndarray):
        """Return (mean, std)."""
        _, _, _, _, mean, std = self._forward_internals(obs)
        return mean, std

    # ------------------------------------------------------------------
    # Log-probability
    # ------------------------------------------------------------------

    def log_prob(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Log-prob of ``action`` under the current policy."""
        mean, std = self.forward(obs)
        action = np.asarray(action, dtype=np.float64).ravel()
        return float(
            -0.5 * np.sum(((action - mean) / std) ** 2)
            - np.sum(self.log_std)
            - 0.5 * self.action_dim * np.log(2.0 * np.pi)
        )

    # ------------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------------

    def entropy(self) -> float:
        """Entropy of the Gaussian (independent of observation)."""
        return float(
            0.5 * self.action_dim * (1.0 + np.log(2.0 * np.pi))
            + np.sum(self.log_std)
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample an action from the policy and return as float32."""
        mean, std = self.forward(obs)
        action = mean + std * rng.standard_normal(self.action_dim)
        action = np.clip(action, -1.0, 1.0)
        return action.astype(np.float32)

    # ------------------------------------------------------------------
    # Gradient of log π(a|s) w.r.t. all parameters
    # ------------------------------------------------------------------

    def log_prob_grad(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Gradient of log π(a|s) w.r.t. the flat parameter vector."""
        obs, pre_h, h, pre_mean, mean, std = self._forward_internals(obs)
        action = np.asarray(action, dtype=np.float64).ravel()

        # d(log_prob)/d(mean_i)  = (a_i - mean_i) / std_i^2
        d_mean = (action - mean) / (std ** 2)               # (action_dim,)
        # d(log_prob)/d(log_std_i) = (a_i - mean_i)^2 / std_i^2 - 1
        d_log_std = (action - mean) ** 2 / (std ** 2) - 1.0  # (action_dim,)

        # Backprop through mean = tanh(pre_mean)
        d_pre_mean = d_mean * (1.0 - mean ** 2)             # (action_dim,)

        # Gradients for W2, b2
        d_W2 = np.outer(d_pre_mean, h)                      # (action_dim, hidden_dim)
        d_b2 = d_pre_mean.copy()

        # Backprop through h = tanh(pre_h)
        d_h = self.W2.T @ d_pre_mean                        # (hidden_dim,)
        d_pre_h = d_h * (1.0 - h ** 2)                     # (hidden_dim,)

        # Gradients for W1, b1
        d_W1 = np.outer(d_pre_h, obs)                       # (hidden_dim, obs_dim)
        d_b1 = d_pre_h.copy()

        return np.concatenate([
            d_W1.ravel(), d_b1.ravel(),
            d_W2.ravel(), d_b2.ravel(),
            d_log_std.ravel(),
        ])

    # ------------------------------------------------------------------
    # Gradient of entropy w.r.t. all parameters
    # ------------------------------------------------------------------

    def entropy_grad(self) -> np.ndarray:
        """Gradient of the entropy H[π] w.r.t. the flat parameter vector.

        Entropy = 0.5 * D * (1 + log 2π) + Σ log_std_i
        Only log_std parameters contribute.
        """
        grad = np.zeros(self.num_params(), dtype=np.float64)
        # log_std block sits at the end
        grad[-self.action_dim:] = 1.0
        return grad


class ValueMLP:
    """Two-layer MLP value (baseline) network used by PPO and TRPO.

    Parameters
    ----------
    obs_dim : int
    hidden_dim : int
    lr : float
        Learning rate for MSE updates.
    rng : np.random.Generator | None
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng if rng is not None else np.random.default_rng(0)
        # He (Kaiming) initialization
        s1 = np.sqrt(2.0 / obs_dim)
        s2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = (rng.standard_normal((hidden_dim, obs_dim)) * s1).astype(np.float64)
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = (rng.standard_normal((1, hidden_dim)) * s2).astype(np.float64)
        self.b2 = np.zeros(1, dtype=np.float64)
        self.lr = lr

    def predict(self, obs: np.ndarray) -> float:
        obs = np.asarray(obs, dtype=np.float64).ravel()
        h = np.tanh(self.W1 @ obs + self.b1)
        return float((self.W2 @ h + self.b2).squeeze())

    def update(self, obs_list: list, targets: np.ndarray, epochs: int = 5) -> None:
        """Update using stochastic gradient descent on MSE loss."""
        n = len(obs_list)
        for _ in range(epochs):
            indices = np.random.permutation(n)
            for i in indices:
                obs = np.asarray(obs_list[i], dtype=np.float64).ravel()
                target = float(targets[i])
                pre_h = self.W1 @ obs + self.b1
                h = np.tanh(pre_h)
                v = float((self.W2 @ h + self.b2).squeeze())
                err = v - target  # scalar

                # Backprop MSE loss
                d_v = 2.0 * err
                d_W2 = d_v * h.reshape(1, -1)
                d_b2 = np.array([d_v])
                d_h = self.W2.T.ravel() * d_v
                d_pre_h = d_h * (1.0 - h ** 2)
                d_W1 = np.outer(d_pre_h, obs)
                d_b1 = d_pre_h.copy()

                self.W1 -= self.lr * d_W1
                self.b1 -= self.lr * d_b1
                self.W2 -= self.lr * d_W2
                self.b2 -= self.lr * d_b2


# ---------------------------------------------------------------------------
# Controller base using a GaussianMLP
# ---------------------------------------------------------------------------

class RLController(BaseAgentController):
    """A controller that wraps a :class:`GaussianMLP` policy.

    During training, actions are sampled stochastically.
    After training, ``deterministic=True`` can be set so that
    ``get_action_2d`` returns the policy mean.

    Parameters
    ----------
    obs_dim : int
    action_dim : int
    hidden_dim : int
    rng : np.random.Generator | None
    deterministic : bool
        If ``True``, return the policy mean instead of sampling.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 64,
        rng: np.random.Generator | None = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__(rng=rng)
        self.policy = GaussianMLP(obs_dim, action_dim, hidden_dim, rng=np.random.default_rng(self.rng.integers(2**32)))
        self.deterministic = deterministic

    def get_action_2d(self, observation: np.ndarray) -> np.ndarray:
        if self.deterministic:
            mean, _ = self.policy.forward(observation)
            return mean.astype(np.float32)
        return self.policy.sample(observation, self.rng)


# ---------------------------------------------------------------------------
# Rollout data structure
# ---------------------------------------------------------------------------

class Rollout(NamedTuple):
    observations: list      # list of np.ndarray
    actions: list           # list of np.ndarray
    rewards: list           # list of float
    log_probs: list         # list of float
    values: list            # list of float  (empty if no value net)
    dones: list             # list of bool


# ---------------------------------------------------------------------------
# Generalised Advantage Estimation (GAE)
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: list,
    values: list,
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and value targets.

    Parameters
    ----------
    rewards : list of float
    values  : list of float  (V(s_t) for t = 0 … T-1)
    last_value : float       (V(s_T), bootstrap; 0 if terminal)
    gamma : float
    lam   : float

    Returns
    -------
    advantages : np.ndarray  shape (T,)
    returns    : np.ndarray  shape (T,)  (targets for value update)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    last_adv = 0.0
    vals = list(values) + [last_value]

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * vals[t + 1] - vals[t]
        advantages[t] = last_adv = delta + gamma * lam * last_adv

    returns = advantages + np.array(values, dtype=np.float64)
    return advantages, returns


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(
    env,
    controllers: dict,
    value_nets: dict | None = None,
) -> dict:
    """Run one episode and collect per-agent rollout data.

    Parameters
    ----------
    env : PettingZoo AEC environment (already reset)
    controllers : dict[agent_id -> RLController]
    value_nets  : dict[agent_id -> ValueMLP] | None

    Returns
    -------
    rollouts : dict[agent_id -> Rollout]
    """
    rollouts: dict[str, dict] = {
        ag: {"observations": [], "actions": [], "rewards": [], "log_probs": [], "values": [], "dones": []}
        for ag in controllers
    }

    for agent in env.agent_iter():
        obs, rew, term, trunc, _ = env.last()
        ctrl = controllers[agent]
        buf = rollouts[agent]

        if term or trunc:
            env.step(None)
            buf["dones"][-1] = True
        else:
            action_2d = ctrl.policy.sample(obs, ctrl.rng)
            lp = ctrl.policy.log_prob(obs, action_2d)

            from controller_api import action2d_to_simple_adversary_continuous
            env.step(action2d_to_simple_adversary_continuous(action_2d))

            val = value_nets[agent].predict(obs) if value_nets else 0.0

            buf["observations"].append(np.asarray(obs, dtype=np.float32))
            buf["actions"].append(action_2d)
            buf["rewards"].append(float(rew))
            buf["log_probs"].append(lp)
            buf["values"].append(val)
            buf["dones"].append(False)

    return {ag: Rollout(**d) for ag, d in rollouts.items()}
