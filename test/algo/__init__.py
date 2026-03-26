"""
RL benchmark algorithms for MPE2 leader-follower environment.

Each benchmark trains all controllers (adversary_0, agent_0, agent_1) using
the same algorithm and exposes a ``run()`` entry point for comparison.

Benchmarks
----------
- PPO-benchmark  : ``ppo_benchmark.PPOBenchmark``
- TRPO-benchmark : ``trpo_benchmark.TRPOBenchmark``
- EntropyRL-benchmark : ``entropy_rl_benchmark.EntropyRLBenchmark``
"""

from .ppo_benchmark import PPOBenchmark, PPOController
from .trpo_benchmark import TRPOBenchmark, TRPOController
from .entropy_rl_benchmark import EntropyRLBenchmark, EntropyRLController

__all__ = [
    "PPOBenchmark",
    "PPOController",
    "TRPOBenchmark",
    "TRPOController",
    "EntropyRLBenchmark",
    "EntropyRLController",
]
