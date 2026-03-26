"""
RL benchmark algorithms for MPE2 leader-follower environment.

Each benchmark trains all controllers (adversary_0, agent_0, agent_1) using
the same algorithm and exposes a ``run()`` entry point for comparison.

Benchmarks
----------
- PPO-benchmark       : ``ppo_benchmark.PPOBenchmark``
- TRPO-benchmark      : ``trpo_benchmark.TRPOBenchmark``
- A2C-benchmark       : ``a2c_benchmark.A2CBenchmark``
- EntropyRL-benchmark : ``entropy_rl_benchmark.EntropyRLBenchmark``
- REINFORCE-benchmark : ``reinforce_benchmark.REINFORCEBenchmark``
"""

from .ppo_benchmark import PPOBenchmark, PPOController
from .trpo_benchmark import TRPOBenchmark, TRPOController
from .a2c_benchmark import A2CBenchmark, A2CController
from .entropy_rl_benchmark import EntropyRLBenchmark, EntropyRLController
from .reinforce_benchmark import REINFORCEBenchmark, REINFORCEController

__all__ = [
    "PPOBenchmark",
    "PPOController",
    "TRPOBenchmark",
    "TRPOController",
    "A2CBenchmark",
    "A2CController",
    "EntropyRLBenchmark",
    "EntropyRLController",
    "REINFORCEBenchmark",
    "REINFORCEController",
]
