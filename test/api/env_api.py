from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mpe2._mpe_utils.core import Agent, Landmark, World
from mpe2._mpe_utils.simple_env import SimpleEnv
from mpe2.simple_adversary.simple_adversary import Scenario


@dataclass
class LandmarkConfig:
    position: np.ndarray | None = None
    is_goal: bool = False
    color: np.ndarray | None = None
    name: str | None = None


class CustomScenario(Scenario):
    """
    你的新设定：
    - adversary_0 作为 leader：奖励=全局奖励（独立于 follower）
    - agent_i 作为 follower：奖励形式相同，仅权重不同
    """

    def __init__(
        self,
        num_good_agents: int = 2,
        landmark_configs: list[LandmarkConfig] | None = None,
        follower_goal_weights: dict[str, np.ndarray] | None = None,
        leader_weight: float = 1.0,
        follower_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_good_agents = num_good_agents
        self.landmark_configs = landmark_configs or []
        self.follower_goal_weights = follower_goal_weights or {}
        self.leader_weight = leader_weight
        self.follower_weight = follower_weight

    def make_world(self, N=None) -> World:
        world = World()
        world.dim_c = 2

        num_adversaries = 1
        num_agents = num_adversaries + self.num_good_agents
        world.num_agents = num_agents

        num_landmarks = len(self.landmark_configs) if self.landmark_configs else self.num_good_agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, a in enumerate(world.agents):
            a.adversary = i < num_adversaries
            a.name = f"adversary_{i}" if a.adversary else f"agent_{i - num_adversaries}"
            a.collide = False
            a.silent = True
            a.size = 0.15

        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, lm in enumerate(world.landmarks):
            cfg = self.landmark_configs[i] if self.landmark_configs else None
            lm.name = cfg.name if cfg and cfg.name else f"goal_{i}"
            lm.collide = False
            lm.movable = False
            lm.size = 0.08
        return world

    def reset_world(self, world, np_random) -> None:
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])

        for i, lm in enumerate(world.landmarks):
            cfg = self.landmark_configs[i] if self.landmark_configs else None
            lm.color = cfg.color.copy() if (cfg and cfg.color is not None) else np.array([0.15, 0.65, 0.15])

        for i, lm in enumerate(world.landmarks):
            cfg = self.landmark_configs[i] if self.landmark_configs else None
            lm.state.p_pos = np.array(cfg.position, dtype=np.float64) if (cfg and cfg.position is not None) else np_random.uniform(-1, +1, world.dim_p)
            lm.state.p_vel = np.zeros(world.dim_p)

        # 关键：simple_adversary.Scenario.observation() 依赖 goal_a
        if len(world.landmarks) > 0:
            default_goal = world.landmarks[0]
        else:
            default_goal = None

        for a in world.agents:
            a.goal_a = default_goal
            a.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            a.state.p_vel = np.zeros(world.dim_p)
            a.state.c = np.zeros(world.dim_c)

    # ---------- follower reward: 同形式，不同权重 ----------
    def follower_reward(self, agent, world) -> float:
        # 默认均匀权重
        num_goals = len(world.landmarks)
        w = self.follower_goal_weights.get(agent.name, np.ones(num_goals) / num_goals)
        w = np.asarray(w, dtype=np.float64)
        w = w / (w.sum() + 1e-12)

        dists = np.array([np.linalg.norm(agent.state.p_pos - lm.state.p_pos) for lm in world.landmarks], dtype=np.float64)
        # 同一形式：加权负距离
        return self.follower_weight * (-float(np.dot(w, dists)))

    # ---------- leader reward: 全局状态 ----------
    def leader_global_reward(self, leader, world) -> float:
        followers = [a for a in world.agents if not a.adversary]
        if not followers:
            return 0.0

        # 示例全局项1：followers 到各自“最近goal”的平均距离（越小越好）
        nearest_goal_dists = []
        for f in followers:
            d = min(np.linalg.norm(f.state.p_pos - lm.state.p_pos) for lm in world.landmarks)
            nearest_goal_dists.append(d)
        term_followers = -float(np.mean(nearest_goal_dists))

        # 示例全局项2：leader 到 followers 质心距离（越小越好）
        centroid = np.mean([f.state.p_pos for f in followers], axis=0)
        term_cohesion = -float(np.linalg.norm(leader.state.p_pos - centroid))

        # 你可调系数
        return self.leader_weight * (0.7 * term_followers + 0.3 * term_cohesion)

    def agent_reward(self, agent, world):
        # follower 统一走同一奖励公式
        return self.follower_reward(agent, world)

    def adversary_reward(self, agent, world):
        # leader 完全独立奖励
        return self.leader_global_reward(agent, world)


class CustomRawEnv(SimpleEnv):
    def __init__(self, scenario: CustomScenario, max_cycles: int = 25, continuous_actions: bool = True, render_mode: str | None = None):
        world = scenario.make_world()
        super().__init__(
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=False,
        )
        self.metadata["name"] = "leader_follower_env"


def build_custom_env(
    num_good_agents: int = 2,
    landmark_configs: list[LandmarkConfig] | None = None,
    follower_goal_weights: dict[str, np.ndarray] | None = None,
    max_cycles: int = 25,
    continuous_actions: bool = True,
    render_mode: str | None = None,
):
    from pettingzoo.utils import wrappers

    scenario = CustomScenario(
        num_good_agents=num_good_agents,
        landmark_configs=landmark_configs,
        follower_goal_weights=follower_goal_weights,
    )
    env = CustomRawEnv(scenario=scenario, max_cycles=max_cycles, continuous_actions=continuous_actions, render_mode=render_mode)
    env = wrappers.ClipOutOfBoundsWrapper(env) if env.continuous_actions else wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

if __name__ == "__main__":
    # ========================= 演示入口说明 =========================
    # 本入口用于“环境 API 层”的最小可运行示例，核心目的是做快速自检：
    # 1) 验证 build_custom_env() 能正确构建封装后的 PettingZoo AEC 环境；
    # 2) 验证 reset(seed=0) 可以正常初始化世界（agent/landmark 状态可生成）；
    # 3) 打印 possible_agents，确认智能体命名与数量符合预期；
    # 4) 调用 close() 释放资源，避免渲染/句柄泄露。
    #
    # 注意：
    # - 这里不负责策略控制，不进行 episode 级别统计；
    # - 这里只验证“环境构建链路”是通的（Scenario -> RawEnv -> Wrapper）。
    # ================================================================
    env = build_custom_env(num_good_agents=2, max_cycles=5)
    env.reset(seed=0)
    print("custom_env_api demo ->", env.possible_agents)
    env.close()