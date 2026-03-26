from __future__ import annotations
import numpy as np


def action2d_to_simple_adversary_continuous(action_2d: np.ndarray) -> np.ndarray:
    """
    将 2D 连续动作 [ax, ay] 映射为 simple_adversary 连续动作 (5,)
    目标空间: Box(0, 1, (5,))
      idx0: no_action
      idx1: left
      idx2: right
      idx3: down
      idx4: up
    """
    a = np.asarray(action_2d, dtype=np.float32).reshape(2,)
    ax = float(np.clip(a[0], -1.0, 1.0))
    ay = float(np.clip(a[1], -1.0, 1.0))

    out = np.zeros(5, dtype=np.float32)
    if ax >= 0:
        out[2] = ax          # right
    else:
        out[1] = -ax         # left

    if ay >= 0:
        out[4] = ay          # up
    else:
        out[3] = -ay         # down

    return out


class BaseAgentController:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def get_action_2d(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        对外统一返回 simple_adversary 可接收的连续动作 (5,)
        """
        action_2d = self.get_action_2d(observation)
        return action2d_to_simple_adversary_continuous(action_2d)


class LeaderController(BaseAgentController):
    """
    占位策略：随机2D动作，再映射到(5,)
    """
    def get_action_2d(self, observation: np.ndarray) -> np.ndarray:
        # 随机范围 [-1,1]，表示二维速度/方向意图
        return self.rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)


class FollowerController(BaseAgentController):
    """
    占位策略：随机2D动作，再映射到(5,)
    """
    def get_action_2d(self, observation: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)


# ---------------------------------------------------------------------------
# RL algorithm registry
# ---------------------------------------------------------------------------

#: Mapping from short algorithm name to benchmark class.
#: Pass any of these names to :func:`build_rl_controllers`.
AVAILABLE_ALGOS: dict[str, str] = {
    "ppo":        "ppo_benchmark.PPOBenchmark",
    "trpo":       "trpo_benchmark.TRPOBenchmark",
    "a2c":        "a2c_benchmark.A2CBenchmark",
    "entropy_rl": "entropy_rl_benchmark.EntropyRLBenchmark",
    "reinforce":  "reinforce_benchmark.REINFORCEBenchmark",
}


def build_rl_controllers(
    algo: str = "ppo",
    num_episodes: int = 200,
    seed: int = 0,
    verbose: bool = False,
    **algo_kwargs,
) -> dict[str, "BaseAgentController"]:
    """Train RL controllers using the specified algorithm.

    This is the main entry point for obtaining trained controllers that can be
    passed directly to :func:`demo_api.run_episode`.

    Parameters
    ----------
    algo : str
        Algorithm name.  One of the keys in :data:`AVAILABLE_ALGOS`
        (``"ppo"``, ``"trpo"``, ``"a2c"``, ``"entropy_rl"``, ``"reinforce"``).
    num_episodes : int
        Number of training episodes.
    seed : int
        Master random seed.
    verbose : bool
        Print training progress.
    **algo_kwargs
        Extra keyword arguments forwarded to the benchmark constructor
        (e.g. ``lr=1e-3``, ``clip_eps=0.1`` for PPO).

    Returns
    -------
    dict[agent_id, BaseAgentController]
        Trained, deterministic controllers for every agent.

    Raises
    ------
    ValueError
        If ``algo`` is not in :data:`AVAILABLE_ALGOS`.

    Examples
    --------
    ::

        from controller_api import build_rl_controllers
        controllers = build_rl_controllers("ppo", num_episodes=100, seed=42)
    """
    import sys
    import os

    algo_lower = algo.lower()
    if algo_lower not in AVAILABLE_ALGOS:
        raise ValueError(
            f"Unknown algorithm '{algo}'. "
            f"Choose from: {sorted(AVAILABLE_ALGOS)}"
        )

    # Add algo directory to sys.path (sibling of api/)
    _algo_dir = os.path.join(os.path.dirname(__file__), "..", "algo")
    if _algo_dir not in sys.path:
        sys.path.insert(0, _algo_dir)

    module_name, class_name = AVAILABLE_ALGOS[algo_lower].split(".")
    import importlib
    module = importlib.import_module(module_name)
    BenchmarkClass = getattr(module, class_name)

    bench = BenchmarkClass(**algo_kwargs)
    return bench.train(num_episodes=num_episodes, seed=seed, verbose=verbose)


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    leader = LeaderController(rng=rng)
    follower = FollowerController(rng=rng)

    obs_leader = np.zeros(8, dtype=np.float32)
    obs_follower = np.zeros(10, dtype=np.float32)

    a2d_l = leader.get_action_2d(obs_leader)
    a5_l = leader.get_action(obs_leader)
    a2d_f = follower.get_action_2d(obs_follower)
    a5_f = follower.get_action(obs_follower)

    print("Leader 2D:", a2d_l, " -> mapped:", a5_l)
    print("Follower 2D:", a2d_f, " -> mapped:", a5_f)
    print("Available RL algos:", sorted(AVAILABLE_ALGOS))