# noqa: D212, D415
"""
Simple Adversary API 测试文档
==============================

本文档演示如何调用 Simple Adversary (``simple_adversary_v3``) 环境，并展示如何为
``adversary_0``、``agent_0``、``agent_1`` 编写自定义的控制器类，以替换默认的随机
动作策略，从而对 action、reward 等参数进行精细化调整。

环境概述
--------
- **智能体 (agents)**：``adversary_0``（红色对手）、``agent_0``、``agent_1``（蓝色合作智能体）
- **地标 (landmarks)**：2 个，其中 1 个为目标地标（绿色），另 1 个为普通地标
- **目标**：
  - 对手 (adversary_0)：尽可能靠近目标地标（但不知道哪个是目标）
  - 合作智能体 (agent_0/1)：最小化自身到目标地标的距离，同时最大化对手到目标地标的距离

观测空间 (Observation Spaces)
------------------------------
- ``agent_0`` / ``agent_1`` (shape: 10, N=2 默认)::

    [goal_rel_pos(2), landmark_rel_positions(num_landmarks×2), other_agent_rel_positions(num_agents_except_self×2)]

- ``adversary_0`` (shape: 8, N=2 默认)::

    [landmark_rel_positions(num_landmarks×2), other_agent_rel_positions(num_good_agents×2)]

动作空间 (Action Spaces)
--------------------------
- 离散模式 (discrete, 默认)：``Discrete(5)``
  - ``0``：不动 (no_action)
  - ``1``：向左移动 (move_left)
  - ``2``：向右移动 (move_right)
  - ``3``：向下移动 (move_down)
  - ``4``：向上移动 (move_up)
- 连续模式 (continuous)：``Box(0.0, 1.0, (5,))``
  - 每个维度对应一个推力大小，顺序同上

使用说明
--------
1. **安装依赖**::

       pip install mpe2 pettingzoo numpy imageio pillow

2. **直接运行演示**::

       python test/simple_adversary_api_test.py

3. **通过 pytest 运行所有测试**::

       pytest test/simple_adversary_api_test.py -v

4. **自定义控制算法**：
   继承 :class:`BaseAgentController` 并重写 ``get_action(observation)`` 方法，
   然后将自定义控制器注册到 :class:`SimpleAdversaryRunner` 的对应智能体槽位即可。
   如果需要修改奖励函数，继承 :class:`CustomScenario` 并覆写
   ``agent_reward``/``adversary_reward`` 方法，再传递给 ``raw_env``。

5. **GIF 保存**::

       runner = SimpleAdversaryRunner(save_gif=True, gif_path="out.gif", gif_fps=5)
       runner.run_episode()

6. **单步调试**::

       runner = SimpleAdversaryRunner(debug=True)
       runner.run_episode()

7. **自定义智能体数量和地标配置**::

       lm_cfgs = [
           LandmarkConfig(position=np.array([0.5, 0.5]), is_goal=True),
           LandmarkConfig(position=np.array([-0.5, -0.5])),
       ]
       runner = SimpleAdversaryRunner(num_good_agents=3, landmark_configs=lm_cfgs)
       runner.run_episode()

8. **按地标权重自定义奖励**::

       weights = {"agent_0": np.array([0.8, 0.2]), "agent_1": np.array([0.3, 0.7])}
       runner = SimpleAdversaryRunner(landmark_weights=weights)
       runner.run_episode()
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pytest

# 无头模式：在无显示器的服务器上运行时设置 SDL 环境变量
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from mpe2 import simple_adversary_v3
from mpe2._mpe_utils.core import Agent, Landmark, World
from mpe2._mpe_utils.simple_env import SimpleEnv, make_env
from mpe2.simple_adversary.simple_adversary import Scenario, raw_env

# 离散动作名称映射（模块级常量，避免在每次函数调用中重复创建）
_ACTION_NAMES: dict[int, str] = {
    0: "no_action",
    1: "move_left",
    2: "move_right",
    3: "move_down",
    4: "move_up",
}


# ---------------------------------------------------------------------------
# 地标配置数据类
# ---------------------------------------------------------------------------


@dataclass
class LandmarkConfig:
    """单个地标的配置。

    Parameters
    ----------
    position:
        地标的固定位置 ``[x, y]``（值域 ``[-1, 1]``）。
        若为 ``None``，则在每次 ``reset`` 时随机初始化。
    is_goal:
        若为 ``True``，此地标被指定为目标地标（绿色）。
        若所有地标均为 ``False``，则每次 ``reset`` 时随机选取一个目标地标。
    color:
        覆盖地标的默认颜色（RGB 数组，值域 ``[0, 1]``）。``None`` 表示使用默认颜色。
    name:
        地标名称；``None`` 时使用 ``"landmark <i>"`` 格式自动命名。
    """

    position: np.ndarray | None = None
    is_goal: bool = False
    color: np.ndarray | None = None
    name: str | None = None


# ---------------------------------------------------------------------------
# 可配置场景：支持自定义智能体数量、地标配置与地标权重
# ---------------------------------------------------------------------------


class ConfigurableScenario(Scenario):
    """扩展自 :class:`~mpe2.simple_adversary.simple_adversary.Scenario` 的可配置场景。

    与原始场景的区别：

    1. **智能体数量独立于地标数量**：``num_good_agents`` 参数控制合作智能体数量，
       ``adversary_0`` 始终固定为 1 个。
    2. **地标完全可配置**：通过 ``landmark_configs`` 列表设置地标数量、类型和位置。
    3. **按地标权重计算奖励**：``landmark_weights`` 为每个智能体提供地标权重向量，
       reward 与 landmark_rel_positions 和权重相关。
    4. **奖励计算接口**：可覆写 :meth:`compute_weighted_reward` 定制奖励逻辑。

    Parameters
    ----------
    num_good_agents:
        合作智能体数量（``agent_0``, ``agent_1``, ...）。``adversary_0`` 始终为 1 个。
    landmark_configs:
        地标配置列表（:class:`LandmarkConfig` 对象）。``None`` 时默认创建
        ``num_good_agents`` 个随机地标（与原始环境行为一致）。
    landmark_weights:
        每个智能体的地标权重字典，格式为 ``{agent_name: np.ndarray(shape=(num_landmarks,))}``。
        ``None`` 或未提供对应智能体的条目时，退回到标准奖励函数。
        权重应归一化（建议和为 1），否则奖励量级可能与默认值差异较大。
    num_agent_neighbors:
        部分可观测：每个智能体可观测的最近邻智能体数量上限。``None`` = 完全可观测。
    num_landmark_neighbors:
        部分可观测：每个智能体可观测的最近邻地标数量上限。``None`` = 完全可观测。

    示例
    ----
    固定地标位置并为不同 agent 指定不同权重::

        lm_cfgs = [
            LandmarkConfig(position=np.array([0.5,  0.5]), is_goal=True),
            LandmarkConfig(position=np.array([-0.5, 0.0])),
            LandmarkConfig(position=np.array([0.0, -0.5])),
        ]
        weights = {
            "agent_0": np.array([0.7, 0.2, 0.1]),
            "agent_1": np.array([0.1, 0.5, 0.4]),
        }
        scenario = ConfigurableScenario(
            num_good_agents=2,
            landmark_configs=lm_cfgs,
            landmark_weights=weights,
        )

    自定义奖励函数::

        class MyScenario(ConfigurableScenario):
            def compute_weighted_reward(self, agent, world, weights, landmark_dists):
                # 稀疏奖励：最近带权地标距离低于阈值时给予正奖励
                weighted_dist = float(np.dot(weights, landmark_dists))
                return 1.0 if weighted_dist < 0.2 else -weighted_dist
    """

    def __init__(
        self,
        num_good_agents: int = 2,
        landmark_configs: list[LandmarkConfig] | None = None,
        landmark_weights: dict[str, np.ndarray] | None = None,
        num_agent_neighbors: int | None = None,
        num_landmark_neighbors: int | None = None,
    ) -> None:
        super().__init__(
            num_agent_neighbors=num_agent_neighbors,
            num_landmark_neighbors=num_landmark_neighbors,
        )
        self.num_good_agents = num_good_agents
        self.landmark_configs: list[LandmarkConfig] = landmark_configs or []
        self.landmark_weights: dict[str, np.ndarray] = landmark_weights or {}

    # ------------------------------------------------------------------
    # 世界构建
    # ------------------------------------------------------------------

    def make_world(self, N=None) -> World:  # noqa: N803 (N accepted for signature compat, not used)
        """构建世界：1 个 adversary_0 + ``num_good_agents`` 个合作智能体。

        ``N`` 参数仅为与父类 :meth:`Scenario.make_world` 签名兼容而保留，不会被使用。
        智能体数量由构造函数的 ``num_good_agents`` 决定。
        地标数量由 ``landmark_configs`` 的长度决定；若 ``landmark_configs`` 为空，
        则使用与原始场景相同的逻辑（``num_good_agents`` 个随机地标）。
        """
        world = World()
        world.dim_c = 2

        num_adversaries = 1
        num_agents = num_adversaries + self.num_good_agents
        world.num_agents = num_agents

        num_landmarks = (
            len(self.landmark_configs) if self.landmark_configs else self.num_good_agents
        )

        # 添加智能体
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = i < num_adversaries
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.15

        # 添加地标
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            cfg = self.landmark_configs[i] if self.landmark_configs else None
            landmark.name = cfg.name if cfg and cfg.name else f"landmark {i}"
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08

        return world

    # ------------------------------------------------------------------
    # 世界重置
    # ------------------------------------------------------------------

    def reset_world(self, world, np_random) -> None:
        """重置世界状态，支持固定地标位置与指定目标地标。"""
        # 智能体颜色
        world.agents[0].color = np.array([0.85, 0.35, 0.35])   # adversary: 红
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])  # good: 蓝

        # 地标颜色与目标选择
        for i, landmark in enumerate(world.landmarks):
            cfg = self.landmark_configs[i] if self.landmark_configs else None
            if cfg and cfg.color is not None:
                landmark.color = cfg.color.copy()
            else:
                landmark.color = np.array([0.15, 0.15, 0.15])

        # 确定目标地标
        explicit_goals = [
            world.landmarks[i]
            for i, cfg in enumerate(self.landmark_configs)
            if cfg.is_goal
        ] if self.landmark_configs else []

        if explicit_goals:
            goal = explicit_goals[0]           # 取第一个显式目标
        else:
            goal = np_random.choice(world.landmarks)   # 随机选取
        goal.color = np.array([0.15, 0.65, 0.15])     # 目标地标: 绿

        for agent in world.agents:
            agent.goal_a = goal

        # 智能体初始位置（随机）
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # 地标初始位置（固定 or 随机）
        for i, landmark in enumerate(world.landmarks):
            cfg = self.landmark_configs[i] if self.landmark_configs else None
            if cfg and cfg.position is not None:
                landmark.state.p_pos = np.array(cfg.position, dtype=np.float64)
            else:
                landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    # ------------------------------------------------------------------
    # 奖励接口
    # ------------------------------------------------------------------

    def compute_weighted_reward(
        self,
        agent,
        world,
        weights: np.ndarray,
        landmark_dists: np.ndarray,
    ) -> float:
        """**可覆写的加权奖励接口**。

        当 ``landmark_weights`` 中存在对应智能体的权重向量时，此方法被调用。
        覆写此方法可自定义奖励计算逻辑（如稀疏奖励、非线性函数等）。

        默认实现：``reward = -dot(weights, landmark_dists)``
        （加权负距离，越近奖励越高）。

        Parameters
        ----------
        agent:
            当前计算奖励的智能体对象。
        world:
            当前世界状态。
        weights:
            该智能体的地标权重向量，形状为 ``(num_landmarks,)``。
        landmark_dists:
            该智能体到各地标的欧氏距离，形状为 ``(num_landmarks,)``。

        Returns
        -------
        float
            本时间步该智能体的奖励值。
        """
        return -float(np.dot(weights, landmark_dists))

    def agent_reward(self, agent, world) -> float:
        """合作智能体奖励：若存在地标权重则调用加权接口，否则退回标准逻辑。"""
        raw_weights = self.landmark_weights.get(agent.name)
        if raw_weights is not None:
            weights_arr = np.asarray(raw_weights, dtype=np.float64)
            landmark_dists = np.array(
                [
                    np.sqrt(np.sum(np.square(agent.state.p_pos - lm.state.p_pos)))
                    for lm in world.landmarks
                ]
            )
            return self.compute_weighted_reward(agent, world, weights_arr, landmark_dists)
        return super().agent_reward(agent, world)


# ---------------------------------------------------------------------------
# 可配置环境（raw_env 变体，使用 ConfigurableScenario）
# ---------------------------------------------------------------------------


class ConfigurableRawEnv(SimpleEnv):
    """与 :class:`~mpe2.simple_adversary.simple_adversary.raw_env` 等价的可配置环境。

    使用 :class:`ConfigurableScenario` 替换内置场景，从而支持：

    - 独立设置合作智能体数量（``num_good_agents``）；
    - 自定义地标数量、种类和位置（``landmark_configs``）；
    - 按地标权重计算奖励（``landmark_weights``）。

    ``adversary_0`` 始终为 1 个。

    Parameters
    ----------
    num_good_agents:
        合作智能体数量，默认为 2。
    landmark_configs:
        地标配置列表；``None`` 时按 ``num_good_agents`` 个随机地标处理。
    landmark_weights:
        per-agent 地标权重字典；``None`` 时退回标准奖励。
    max_cycles:
        每轮最大时间步数，默认 25。
    continuous_actions:
        是否使用连续动作空间，默认 ``False``。
    render_mode:
        渲染模式（``None`` / ``"human"`` / ``"rgb_array"``）。
    dynamic_rescaling:
        是否根据屏幕大小动态调整实体尺寸，默认 ``False``。
    num_agent_neighbors:
        部分可观测邻居数（智能体），``None`` = 完全可观测。
    num_landmark_neighbors:
        部分可观测邻居数（地标），``None`` = 完全可观测。
    """

    def __init__(
        self,
        num_good_agents: int = 2,
        landmark_configs: list[LandmarkConfig] | None = None,
        landmark_weights: dict[str, np.ndarray] | None = None,
        max_cycles: int = 25,
        continuous_actions: bool = False,
        render_mode: str | None = None,
        dynamic_rescaling: bool = False,
        num_agent_neighbors: int | None = None,
        num_landmark_neighbors: int | None = None,
    ) -> None:
        scenario = ConfigurableScenario(
            num_good_agents=num_good_agents,
            landmark_configs=landmark_configs,
            landmark_weights=landmark_weights,
            num_agent_neighbors=num_agent_neighbors,
            num_landmark_neighbors=num_landmark_neighbors,
        )
        world = scenario.make_world()
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "configurable_simple_adversary"


def _make_configurable_env(
    num_good_agents: int = 2,
    landmark_configs: list[LandmarkConfig] | None = None,
    landmark_weights: dict[str, np.ndarray] | None = None,
    **kwargs,
):
    """工厂函数：创建经过标准包装的 :class:`ConfigurableRawEnv`。"""
    from pettingzoo.utils import wrappers

    env = ConfigurableRawEnv(
        num_good_agents=num_good_agents,
        landmark_configs=landmark_configs,
        landmark_weights=landmark_weights,
        **kwargs,
    )
    if env.continuous_actions:
        env = wrappers.ClipOutOfBoundsWrapper(env)
    else:
        env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


# ---------------------------------------------------------------------------
# 自定义控制器基类
# ---------------------------------------------------------------------------


class BaseAgentController:
    """所有自定义控制器的基类。

    子类需要实现 :meth:`get_action`，接收观测向量并返回一个合法动作。

    Parameters
    ----------
    continuous_actions:
        若为 ``True``，则动作空间为连续的 ``Box(5,)``；
        否则为离散的 ``Discrete(5)``。
    """

    def __init__(self, continuous_actions: bool = False) -> None:
        self.continuous_actions = continuous_actions

    def get_action(self, observation: np.ndarray) -> int | np.ndarray:
        """根据观测向量返回动作。

        Parameters
        ----------
        observation:
            当前时间步从环境中获取的观测向量。

        Returns
        -------
        action:
            - 离散模式：整数 ``0``–``4``
            - 连续模式：长度为 5 的 ``np.ndarray``，值域 ``[0, 1]``
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# adversary_0 控制器
# ---------------------------------------------------------------------------


class Adversary0Controller(BaseAgentController):
    """``adversary_0`` 的自定义控制器。

    对手的观测结构为::

        obs[0:4]  — 两个地标的相对位置 (landmark_rel_positions)
        obs[4:8]  — 两个合作智能体的相对位置 (other_agent_rel_positions)

    本控制器使用贪心策略：计算到两个地标的距离，并向距离更近的地标移动。
    由于对手不知道哪个是目标，此策略模拟一种"追踪最近地标"的基础启发式算法。
    用户可以继承本类并重写 :meth:`get_action` 以实现更复杂的控制逻辑。
    """

    # 观测向量切分索引
    N_LANDMARKS = 2
    OBS_LANDMARK_START = 0
    OBS_LANDMARK_END = 4   # 2 个地标 × 2 维

    def get_action(self, observation: np.ndarray) -> int | np.ndarray:
        """向最近的地标移动（贪心启发式）。"""
        landmark_positions = observation[
            self.OBS_LANDMARK_START : self.OBS_LANDMARK_END
        ].reshape(self.N_LANDMARKS, 2)

        # 计算到每个地标的欧氏距离
        distances = np.linalg.norm(landmark_positions, axis=1)
        target_idx = int(np.argmin(distances))
        target_rel_pos = landmark_positions[target_idx]  # (dx, dy)

        return self._direction_action(target_rel_pos)

    def _direction_action(self, rel_pos: np.ndarray) -> int | np.ndarray:
        """将相对位移转换为离散或连续动作。"""
        dx, dy = rel_pos
        if self.continuous_actions:
            # 连续模式：将目标方向映射为推力向量
            action = np.zeros(5, dtype=np.float32)
            if abs(dx) >= abs(dy):
                action[2 if dx > 0 else 1] = float(min(abs(dx), 1.0))
            else:
                action[4 if dy > 0 else 3] = float(min(abs(dy), 1.0))
            return action
        else:
            # 离散模式：选取主方向
            if abs(dx) >= abs(dy):
                return 2 if dx > 0 else 1  # 右/左
            else:
                return 4 if dy > 0 else 3  # 上/下


# ---------------------------------------------------------------------------
# agent_0 控制器
# ---------------------------------------------------------------------------


class Agent0Controller(BaseAgentController):
    """``agent_0`` 的自定义控制器。

    合作智能体的观测结构为::

        obs[0:2]  — 目标地标的相对位置 (goal_rel_pos)
        obs[2:6]  — 两个地标的相对位置 (landmark_rel_positions)
        obs[6:10] — 两个其他智能体的相对位置 (other_agent_rel_positions)

    本控制器使用直接追踪策略：直接向目标地标移动。
    这是一种最优合作智能体策略的简化版本。
    用户可以继承本类并重写 :meth:`get_action` 以实现更复杂的控制逻辑。
    """

    OBS_GOAL_START = 0
    OBS_GOAL_END = 2  # 目标相对位置 (dx, dy)

    def get_action(self, observation: np.ndarray) -> int | np.ndarray:
        """直接向目标地标移动。"""
        goal_rel_pos = observation[self.OBS_GOAL_START : self.OBS_GOAL_END]
        return self._direction_action(goal_rel_pos)

    def _direction_action(self, rel_pos: np.ndarray) -> int | np.ndarray:
        """将相对位移转换为离散或连续动作。"""
        dx, dy = rel_pos
        if self.continuous_actions:
            action = np.zeros(5, dtype=np.float32)
            if abs(dx) >= abs(dy):
                action[2 if dx > 0 else 1] = float(min(abs(dx), 1.0))
            else:
                action[4 if dy > 0 else 3] = float(min(abs(dy), 1.0))
            return action
        else:
            if abs(dx) >= abs(dy):
                return 2 if dx > 0 else 1
            else:
                return 4 if dy > 0 else 3


# ---------------------------------------------------------------------------
# agent_1 控制器
# ---------------------------------------------------------------------------


class Agent1Controller(BaseAgentController):
    """``agent_1`` 的自定义控制器。

    观测结构与 :class:`Agent0Controller` 相同。

    本控制器使用"欺骗性覆盖"策略：若 ``agent_0`` 已经很接近目标地标，
    则 ``agent_1`` 转而驻守另一个（非目标）地标以迷惑对手。
    若 ``agent_0`` 距离目标较远，则与 ``agent_0`` 一同向目标移动。
    用户可以继承本类并重写 :meth:`get_action` 以实现更复杂的控制逻辑。

    Parameters
    ----------
    coverage_threshold:
        当 ``agent_0`` 到目标地标的距离小于该阈值时，认为目标已被覆盖，
        ``agent_1`` 转为覆盖另一个地标。
    """

    OBS_GOAL_START = 0
    OBS_GOAL_END = 2
    OBS_LANDMARK_START = 2
    OBS_LANDMARK_END = 6   # 两个地标相对位置
    OBS_OTHER_START = 6
    OBS_OTHER_END = 10     # 其他两个智能体相对位置

    N_LANDMARKS = 2

    def __init__(
        self,
        continuous_actions: bool = False,
        coverage_threshold: float = 0.3,
    ) -> None:
        super().__init__(continuous_actions)
        self.coverage_threshold = coverage_threshold

    def get_action(self, observation: np.ndarray) -> int | np.ndarray:
        """欺骗性覆盖策略。"""
        goal_rel_pos = observation[self.OBS_GOAL_START : self.OBS_GOAL_END]
        landmark_positions = observation[
            self.OBS_LANDMARK_START : self.OBS_LANDMARK_END
        ].reshape(self.N_LANDMARKS, 2)
        other_agent_positions = observation[
            self.OBS_OTHER_START : self.OBS_OTHER_END
        ].reshape(2, 2)

        # agent_0 相对于 agent_1 的位置 —— 第一个条目为 adversary_0，第二个为 agent_0
        agent0_rel_pos = other_agent_positions[1]
        # agent_0 到目标的估算距离（通过 goal_rel_pos - agent0_rel_pos）
        agent0_to_goal_dist = np.linalg.norm(goal_rel_pos - agent0_rel_pos)

        if agent0_to_goal_dist < self.coverage_threshold:
            # agent_0 已接近目标，agent_1 转而覆盖距自身最远的地标
            distances = np.linalg.norm(landmark_positions, axis=1)
            decoy_idx = int(np.argmax(distances))
            target_rel_pos = landmark_positions[decoy_idx]
        else:
            # 与 agent_0 一起追踪目标地标
            target_rel_pos = goal_rel_pos

        return self._direction_action(target_rel_pos)

    def _direction_action(self, rel_pos: np.ndarray) -> int | np.ndarray:
        dx, dy = rel_pos
        if self.continuous_actions:
            action = np.zeros(5, dtype=np.float32)
            if abs(dx) >= abs(dy):
                action[2 if dx > 0 else 1] = float(min(abs(dx), 1.0))
            else:
                action[4 if dy > 0 else 3] = float(min(abs(dy), 1.0))
            return action
        else:
            if abs(dx) >= abs(dy):
                return 2 if dx > 0 else 1
            else:
                return 4 if dy > 0 else 3


# ---------------------------------------------------------------------------
# 自定义奖励场景（可选扩展点）
# ---------------------------------------------------------------------------


class CustomScenario(Scenario):
    """在 :class:`~mpe2.simple_adversary.simple_adversary.Scenario` 基础上扩展的
    自定义奖励场景。

    用户可以覆写 :meth:`agent_reward` 或 :meth:`adversary_reward` 来调整奖励
    计算逻辑，例如：引入稀疏奖励、改变奖励权重、添加惩罚项等。

    示例
    ----
    ::

        class MyScenario(CustomScenario):
            def adversary_reward(self, agent, world):
                # 自定义对手奖励：抵近任意合作智能体也获得部分奖励
                base_reward = super().adversary_reward(agent, world)
                bonus = -min(
                    np.linalg.norm(agent.state.p_pos - g.state.p_pos)
                    for g in self.good_agents(world)
                )
                return base_reward + 0.1 * bonus
    """

    # 奖励权重：可在子类或实例化时调整
    adv_reward_weight: float = 1.0
    good_reward_weight: float = 1.0

    def adversary_reward(self, agent, world):
        """对手奖励（可覆写）。

        默认行为与原始 Scenario 相同，但乘以 :attr:`adv_reward_weight`。
        """
        base = super().adversary_reward(agent, world)
        return self.adv_reward_weight * base

    def agent_reward(self, agent, world):
        """合作智能体奖励（可覆写）。

        默认行为与原始 Scenario 相同，但乘以 :attr:`good_reward_weight`。
        """
        base = super().agent_reward(agent, world)
        return self.good_reward_weight * base


# ---------------------------------------------------------------------------
# 环境运行器
# ---------------------------------------------------------------------------


class SimpleAdversaryRunner:
    """封装 Simple Adversary AEC 环境的完整运行流程。

    Parameters
    ----------
    N:
        **仅在不使用 ``num_good_agents`` 时有效**。合作智能体数量（同时也是地标数量），
        默认为 2。若同时传入 ``num_good_agents``，则 ``N`` 被忽略。
    max_cycles:
        每轮的最大时间步数，默认为 25。
    continuous_actions:
        是否使用连续动作空间，默认为 ``False``（离散）。
    render_mode:
        渲染模式，``None``（不渲染）、``"human"`` 或 ``"rgb_array"``。
        GIF 功能会自动将 ``render_mode`` 切换为 ``"rgb_array"``。
    seed:
        随机种子，用于环境复现。
    adversary_controller:
        ``adversary_0`` 的控制器实例，默认为 :class:`Adversary0Controller`。
    agent0_controller:
        ``agent_0`` 的控制器实例，默认为 :class:`Agent0Controller`。
    agent1_controller:
        ``agent_1`` 的控制器实例，默认为 :class:`Agent1Controller`。
    num_good_agents:
        合作智能体数量（独立于地标数量）。当指定此参数时，
        环境将使用 :class:`ConfigurableScenario` 而非默认场景。
    landmark_configs:
        地标配置列表（:class:`LandmarkConfig` 对象列表）。
        若提供，则可独立设置每个地标的位置、种类和颜色。
    landmark_weights:
        per-agent 地标权重字典，格式 ``{agent_name: np.ndarray(num_landmarks,)}``。
        当提供此参数时，奖励函数通过 :meth:`ConfigurableScenario.compute_weighted_reward`
        计算。
    debug:
        若为 ``True``，每个世界步完成后打印所有智能体和地标的状态信息与奖励。
        设置为 ``"step"`` 可在每个 **agent 步**（而非 world 步）后打印动作信息。
    save_gif:
        若为 ``True``，将每个世界步的渲染帧收集为 GIF 并保存到 ``gif_path``。
        要求安装 ``imageio`` 和 ``pillow``。
    gif_path:
        GIF 文件保存路径，默认为 ``"simulation.gif"``。
    gif_fps:
        GIF 帧率（帧/秒），默认为 5。

    使用示例
    --------
    **基本使用**::

        runner = SimpleAdversaryRunner(max_cycles=50, seed=42)
        stats = runner.run_episode()
        print("累计奖励:", stats["cumulative_rewards"])
        print("每步奖励:", stats["step_rewards"])

    **自定义控制器**::

        class MyAdversary(Adversary0Controller):
            def get_action(self, obs):
                return np.random.randint(0, 5)

        runner = SimpleAdversaryRunner(adversary_controller=MyAdversary(), seed=0)
        stats = runner.run_episode()

    **连续动作模式**::

        runner = SimpleAdversaryRunner(continuous_actions=True, seed=0)
        stats = runner.run_episode()

    **使用自定义奖励场景**::

        # 通过 raw_env 直接使用 CustomScenario（绕过工厂函数）
        # 见 test_custom_scenario_reward_weights() 测试

    **GIF 保存**::

        runner = SimpleAdversaryRunner(save_gif=True, gif_path="sim.gif", gif_fps=5)
        stats = runner.run_episode()

    **单步调试**::

        runner = SimpleAdversaryRunner(debug=True)
        stats = runner.run_episode()

    **自定义智能体与地标**::

        lm_cfgs = [
            LandmarkConfig(position=np.array([0.5, 0.5]), is_goal=True),
            LandmarkConfig(position=np.array([-0.5, -0.5])),
        ]
        weights = {"agent_0": np.array([0.8, 0.2]), "agent_1": np.array([0.3, 0.7])}
        runner = SimpleAdversaryRunner(
            num_good_agents=2,
            landmark_configs=lm_cfgs,
            landmark_weights=weights,
        )
        stats = runner.run_episode()
    """

    def __init__(
        self,
        N: int = 2,
        max_cycles: int = 25,
        continuous_actions: bool = False,
        render_mode: str | None = None,
        seed: int | None = 0,
        adversary_controller: BaseAgentController | None = None,
        agent0_controller: BaseAgentController | None = None,
        agent1_controller: BaseAgentController | None = None,
        # ---- 新增：可配置智能体/地标 ----
        num_good_agents: int | None = None,
        landmark_configs: list[LandmarkConfig] | None = None,
        landmark_weights: dict[str, np.ndarray] | None = None,
        # ---- 新增：调试与 GIF ----
        debug: bool | str = False,
        save_gif: bool = False,
        gif_path: str = "simulation.gif",
        gif_fps: int = 5,
    ) -> None:
        # 若明确传入 num_good_agents，则覆盖 N
        effective_n = num_good_agents if num_good_agents is not None else N
        self.N = effective_n
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.seed = seed
        self.debug = debug
        self.save_gif = save_gif
        self.gif_path = gif_path
        self.gif_fps = gif_fps

        # GIF 需要 rgb_array 渲染模式
        if save_gif:
            render_mode = "rgb_array"
        self.render_mode = render_mode

        # 注册控制器（若未提供则使用默认实现）
        self.controllers: dict[str, BaseAgentController] = {
            "adversary_0": adversary_controller
            or Adversary0Controller(continuous_actions),
            "agent_0": agent0_controller or Agent0Controller(continuous_actions),
            "agent_1": agent1_controller or Agent1Controller(continuous_actions),
        }

        # 判断是否使用可配置场景
        use_configurable = (
            num_good_agents is not None
            or landmark_configs is not None
            or landmark_weights is not None
        )

        if use_configurable:
            self.env = _make_configurable_env(
                num_good_agents=effective_n,
                landmark_configs=landmark_configs,
                landmark_weights=landmark_weights,
                max_cycles=max_cycles,
                continuous_actions=continuous_actions,
                render_mode=render_mode,
            )
        else:
            self.env = simple_adversary_v3.env(
                N=N,
                max_cycles=max_cycles,
                continuous_actions=continuous_actions,
                render_mode=render_mode,
            )

    # ------------------------------------------------------------------
    # 内部辅助：打印世界状态
    # ------------------------------------------------------------------

    @staticmethod
    def _print_world_state(env, step_idx: int, rewards: dict) -> None:
        """打印当前世界步的智能体位置、地标位置和奖励。"""
        # 通过 AEC 包装器访问底层世界（剥去 OrderEnforcingWrapper 等）
        inner = env
        while hasattr(inner, "env"):
            inner = inner.env
        world = inner.world

        print(f"\n{'─' * 50}")
        print(f"[World Step {step_idx}]")
        print("  智能体位置:")
        for agent in world.agents:
            pos = agent.state.p_pos
            vel = agent.state.p_vel
            role = "adversary" if agent.adversary else "good"
            print(f"    {agent.name:<14} ({role})  pos=({pos[0]:+.4f}, {pos[1]:+.4f})  "
                  f"vel=({vel[0]:+.4f}, {vel[1]:+.4f})")
        print("  地标位置:")
        for lm in world.landmarks:
            is_goal = hasattr(world.agents[0], "goal_a") and world.agents[0].goal_a is lm
            pos = lm.state.p_pos
            goal_tag = " ← GOAL" if is_goal else ""
            print(f"    {lm.name:<14}  pos=({pos[0]:+.4f}, {pos[1]:+.4f}){goal_tag}")
        print("  本步奖励:")
        for agent_name, rew in rewards.items():
            print(f"    {agent_name:<14}: {rew:+.6f}")

    @staticmethod
    def _print_agent_action(agent_name: str, action, obs: np.ndarray) -> None:
        """打印单个 agent 步的动作信息（debug='step' 模式）。"""
        if isinstance(action, (int, np.integer)):
            action_str = f"{_ACTION_NAMES.get(int(action), str(action))} ({action})"
        else:
            action_str = np.array2string(np.asarray(action), precision=3)
        print(f"  [AgentStep] {agent_name:<14} → action={action_str}  "
              f"obs_shape={obs.shape}")

    # ------------------------------------------------------------------
    # episode 运行
    # ------------------------------------------------------------------

    def run_episode(self) -> dict:
        """运行一个完整的 episode 并收集统计信息。

        Returns
        -------
        stats : dict
            包含以下键：

            - ``cumulative_rewards`` (dict[str, float])：每个智能体的累计奖励
            - ``step_rewards`` (list[dict[str, float]])：每个世界时间步的奖励
            - ``total_steps`` (int)：执行的世界步数
            - ``observations`` (dict[str, list[np.ndarray]])：每个智能体所有时间步的观测
            - ``gif_path`` (str | None)：若保存了 GIF 则为其路径，否则为 ``None``
        """
        self.env.reset(seed=self.seed)

        cumulative_rewards: dict[str, float] = {
            agent: 0.0 for agent in self.env.possible_agents
        }
        step_rewards: list[dict[str, float]] = []
        observations: dict[str, list[np.ndarray]] = {
            agent: [] for agent in self.env.possible_agents
        }
        frames: list[np.ndarray] = []   # 用于 GIF

        # AEC 循环：每次迭代对应一个智能体的行动
        world_steps = 0
        last_agent = self.env.possible_agents[-1]

        for agent_name in self.env.agent_iter():
            obs, reward, termination, truncation, info = self.env.last()

            # 记录观测与奖励
            observations[agent_name].append(obs.copy())
            cumulative_rewards[agent_name] += reward

            if termination or truncation:
                self.env.step(None)
            else:
                # 使用对应控制器生成动作
                controller = self.controllers.get(agent_name)
                if controller is not None:
                    action = controller.get_action(obs)
                else:
                    action = self.env.action_space(agent_name).sample()

                # 单步调试：打印每个 agent 步的动作
                if self.debug == "step":
                    self._print_agent_action(agent_name, action, obs)

                self.env.step(action)

                # 最后一个智能体行动后，世界完成一步
                if agent_name == last_agent:
                    world_steps += 1
                    current_rewards = dict(self.env.rewards)
                    step_rewards.append(current_rewards)

                    # 单步调试（world-level）
                    if self.debug is True or self.debug == "world":
                        self._print_world_state(self.env, world_steps, current_rewards)

                    # 收集 GIF 帧（rgb_array 渲染）
                    if self.save_gif:
                        frame = self.env.render()
                        if frame is not None:
                            frames.append(frame)

        # 保存 GIF
        saved_gif_path: str | None = None
        if self.save_gif and frames:
            saved_gif_path = self._save_gif(frames)

        return {
            "cumulative_rewards": cumulative_rewards,
            "step_rewards": step_rewards,
            "total_steps": world_steps,
            "observations": observations,
            "gif_path": saved_gif_path,
        }

    def _save_gif(self, frames: list[np.ndarray]) -> str:
        """将 rgb_array 帧列表保存为 GIF 文件。

        Parameters
        ----------
        frames:
            形状为 ``(H, W, 3)`` 的 numpy 数组列表（uint8）。

        Returns
        -------
        str
            保存的 GIF 文件路径。
        """
        try:
            import imageio
        except ImportError as exc:
            raise ImportError(
                "保存 GIF 需要 imageio 库。请运行: pip install imageio pillow"
            ) from exc

        # 确保输出目录存在
        gif_dir = os.path.dirname(os.path.abspath(self.gif_path))
        os.makedirs(gif_dir, exist_ok=True)

        # 将 float 帧转为 uint8
        processed: list[np.ndarray] = []
        for f in frames:
            arr = np.asarray(f)
            if arr.dtype != np.uint8:
                # 若为 [0, 1] 范围的浮点数组，先缩放到 [0, 255]
                if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            processed.append(arr)

        duration = 1.0 / self.gif_fps
        imageio.mimsave(self.gif_path, processed, duration=duration)
        print(f"GIF 已保存至: {self.gif_path}（{len(processed)} 帧，{self.gif_fps} fps）")
        return self.gif_path

    def close(self) -> None:
        """释放环境资源。"""
        self.env.close()


# ---------------------------------------------------------------------------
# pytest 测试函数
# ---------------------------------------------------------------------------


def test_aec_discrete_default_controllers():
    """AEC 离散模式：使用默认（内置）控制器运行一个完整 episode，
    验证奖励和观测维度符合预期。"""
    runner = SimpleAdversaryRunner(max_cycles=25, seed=42)
    stats = runner.run_episode()
    runner.close()

    # 所有智能体都应出现在累计奖励中
    assert set(stats["cumulative_rewards"].keys()) == {"adversary_0", "agent_0", "agent_1"}

    # episode 应正好运行 max_cycles 步
    assert stats["total_steps"] == 25

    # 观测维度检查
    assert stats["observations"]["adversary_0"][0].shape == (8,)
    assert stats["observations"]["agent_0"][0].shape == (10,)
    assert stats["observations"]["agent_1"][0].shape == (10,)


def test_aec_continuous_default_controllers():
    """AEC 连续模式：使用默认控制器验证 episode 可正常完成。"""
    runner = SimpleAdversaryRunner(
        max_cycles=25, continuous_actions=True, seed=0
    )
    stats = runner.run_episode()
    runner.close()

    assert stats["total_steps"] == 25
    assert stats["observations"]["adversary_0"][0].shape == (8,)
    assert stats["observations"]["agent_0"][0].shape == (10,)


def test_custom_adversary_controller():
    """自定义对手控制器：使用随机策略的 adversary_0，验证环境仍能正常运行。"""

    class RandomAdversary(Adversary0Controller):
        """始终返回随机离散动作的对手。"""

        def get_action(self, observation: np.ndarray) -> int:
            return int(np.random.randint(0, 5))

    runner = SimpleAdversaryRunner(
        max_cycles=20,
        seed=1,
        adversary_controller=RandomAdversary(),
    )
    stats = runner.run_episode()
    runner.close()

    assert stats["total_steps"] == 20


def test_custom_agent0_controller():
    """自定义 agent_0 控制器：使用"静止不动"策略，验证奖励逻辑正常工作。"""

    class StationaryAgent(Agent0Controller):
        """始终选择 no_action（动作 0）的合作智能体。"""

        def get_action(self, observation: np.ndarray) -> int:
            return 0  # no_action

    runner = SimpleAdversaryRunner(
        max_cycles=20,
        seed=2,
        agent0_controller=StationaryAgent(),
    )
    stats = runner.run_episode()
    runner.close()

    assert stats["total_steps"] == 20


def test_custom_agent1_controller():
    """自定义 agent_1 控制器：使用"始终向上"策略，验证环境接受连续动作。"""

    class AlwaysUpAgent(Agent1Controller):
        """始终选择"向上"动作的合作智能体。"""

        def get_action(self, observation: np.ndarray) -> int:
            return 4  # move_up

    runner = SimpleAdversaryRunner(
        max_cycles=20,
        seed=3,
        agent1_controller=AlwaysUpAgent(),
    )
    stats = runner.run_episode()
    runner.close()

    assert stats["total_steps"] == 20


def test_custom_scenario_reward_weights():
    """自定义奖励场景：直接验证 CustomScenario 的奖励权重参数能正确缩放奖励。

    在相同的世界状态下，分别用 weight=1 和 weight=2 调用奖励函数，
    断言 weight=2 时的奖励恰好等于 weight=1 时的两倍。
    """
    # 构造两个场景：权重分别为 1 和 2
    scenario_1x = CustomScenario()
    scenario_1x.adv_reward_weight = 1.0

    scenario_2x = CustomScenario()
    scenario_2x.adv_reward_weight = 2.0

    # 构造一个共享的世界（两个场景使用相同的 make_world 逻辑）
    world = scenario_1x.make_world(2)
    rng = np.random.default_rng(42)

    class _MockRng:
        """轻量级 np_random 兼容对象。"""
        def uniform(self, low, high, size=None):
            return rng.uniform(low, high, size)
        def choice(self, arr):
            return rng.choice(arr)

    scenario_1x.reset_world(world, _MockRng())

    # 取对手智能体和当前奖励基准
    adversary = world.agents[0]
    assert adversary.adversary, "第一个智能体应为对手"

    rew_1x = scenario_1x.adversary_reward(adversary, world)
    rew_2x = scenario_2x.adversary_reward(adversary, world)

    assert rew_1x != 0.0, "基准对手奖励不应为零"
    assert abs(rew_2x / rew_1x - 2.0) < 1e-9, (
        f"期望 2× 奖励倍率，实际比值 = {rew_2x / rew_1x:.6f} "
        f"(1×={rew_1x:.6f}, 2×={rew_2x:.6f})"
    )

    # 合作智能体奖励不受 adv_reward_weight 影响
    for agent in world.agents[1:]:
        rew_a_1x = scenario_1x.agent_reward(agent, world)
        rew_a_2x = scenario_2x.agent_reward(agent, world)
        assert abs(rew_a_1x - rew_a_2x) < 1e-9, (
            f"{agent.name} 的奖励在 adv_reward_weight 变化时不应改变"
        )


def test_parallel_api():
    """Parallel API：验证并行模式下的 episode 也可正常运行，
    所有智能体同时接收动作并返回观测与奖励。"""
    parallel_env = simple_adversary_v3.parallel_env(N=2, max_cycles=25)
    obs, infos = parallel_env.reset(seed=0)

    # 验证初始观测维度
    assert obs["adversary_0"].shape == (8,)
    assert obs["agent_0"].shape == (10,)
    assert obs["agent_1"].shape == (10,)

    # 构建控制器
    controllers: dict[str, BaseAgentController] = {
        "adversary_0": Adversary0Controller(),
        "agent_0": Agent0Controller(),
        "agent_1": Agent1Controller(),
    }

    cumulative_rewards = {agent: 0.0 for agent in parallel_env.possible_agents}
    steps = 0
    while parallel_env.agents:
        actions = {
            agent: controllers[agent].get_action(obs[agent])
            for agent in parallel_env.agents
        }
        obs, rewards, terminations, truncations, infos = parallel_env.step(actions)
        for agent, rew in rewards.items():
            cumulative_rewards[agent] += rew
        steps += 1

    parallel_env.close()

    assert steps == 25
    for agent, total_rew in cumulative_rewards.items():
        assert np.isfinite(total_rew), f"{agent} 的累计奖励不是有限值: {total_rew}"


def test_observation_matches_declared_space():
    """验证每个智能体的实际观测维度与环境声明的观测空间维度完全一致。"""
    env = simple_adversary_v3.env(N=2, max_cycles=5)
    env.reset(seed=7)

    for agent_name in env.possible_agents:
        declared_shape = env.observation_space(agent_name).shape
        obs = env.observe(agent_name)
        assert obs.shape == declared_shape, (
            f"{agent_name}: 声明观测维度 {declared_shape} != 实际维度 {obs.shape}"
        )

    env.close()


def test_n_larger_than_2():
    """验证 N=3（3 个合作智能体，3 个地标）时自定义控制器同样有效。"""

    class GenericGoalChaser(BaseAgentController):
        """通用目标追踪器（适用于任意合作智能体）。"""

        def get_action(self, observation: np.ndarray) -> int:
            # 合作智能体观测的前 2 维始终是目标相对位置
            dx, dy = observation[0], observation[1]
            if abs(dx) >= abs(dy):
                return 2 if dx > 0 else 1
            else:
                return 4 if dy > 0 else 3

    N = 3
    env = simple_adversary_v3.env(N=N, max_cycles=20)
    env.reset(seed=5)

    chaser = GenericGoalChaser()
    adversary_ctrl = Adversary0Controller()

    # 手动构建控制器映射（N=3 时有 agent_0, agent_1, agent_2）
    agent_controllers: dict[str, BaseAgentController] = {
        "adversary_0": adversary_ctrl,
        "agent_0": chaser,
        "agent_1": chaser,
        "agent_2": chaser,
    }

    for agent_name in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        if termination or truncation:
            env.step(None)
        else:
            ctrl = agent_controllers.get(agent_name)
            action = ctrl.get_action(obs) if ctrl else env.action_space(agent_name).sample()
            env.step(action)

    env.close()


# ---------------------------------------------------------------------------
# 新功能测试：GIF、调试、可配置场景、地标权重
# ---------------------------------------------------------------------------


def test_save_gif(tmp_path):
    """GIF 保存：验证 save_gif=True 时生成有效的 GIF 文件。"""
    gif_file = str(tmp_path / "test_sim.gif")
    runner = SimpleAdversaryRunner(
        max_cycles=5,
        seed=10,
        save_gif=True,
        gif_path=gif_file,
        gif_fps=5,
    )
    stats = runner.run_episode()
    runner.close()

    assert stats["gif_path"] == gif_file, "stats 应包含 gif_path"
    assert os.path.isfile(gif_file), f"GIF 文件应存在: {gif_file}"
    # 检查文件非空
    assert os.path.getsize(gif_file) > 0, "GIF 文件不应为空"


def test_debug_world_mode(capsys):
    """单步调试（world 模式）：debug=True 时应在 stdout 打印世界状态信息。"""
    runner = SimpleAdversaryRunner(max_cycles=3, seed=20, debug=True)
    runner.run_episode()
    runner.close()

    captured = capsys.readouterr()
    assert "[World Step" in captured.out, "应包含 [World Step ...] 调试输出"
    assert "adversary_0" in captured.out, "应包含 adversary_0 的状态信息"
    assert "landmark" in captured.out, "应包含 landmark 的状态信息"


def test_debug_step_mode(capsys):
    """单步调试（step 模式）：debug='step' 时应在每个 agent 步后打印动作信息。"""
    runner = SimpleAdversaryRunner(max_cycles=2, seed=21, debug="step")
    runner.run_episode()
    runner.close()

    captured = capsys.readouterr()
    assert "[AgentStep]" in captured.out, "应包含 [AgentStep] 调试输出"
    assert "action=" in captured.out, "应包含 action= 字段"


def test_configurable_num_good_agents():
    """可配置场景：设置 num_good_agents=3，验证有 4 个智能体（1 adversary + 3 good）。"""
    runner = SimpleAdversaryRunner(num_good_agents=3, max_cycles=10, seed=30)
    possible = runner.env.possible_agents
    runner.close()

    assert "adversary_0" in possible, "应包含 adversary_0"
    assert "agent_0" in possible, "应包含 agent_0"
    assert "agent_1" in possible, "应包含 agent_1"
    assert "agent_2" in possible, "应包含 agent_2"
    assert len(possible) == 4, f"应有 4 个智能体，实际: {len(possible)}"


def test_configurable_landmark_fixed_positions():
    """可配置场景：固定地标位置，验证 reset 后地标位置与配置一致。"""
    pos0 = np.array([0.5, 0.5])
    pos1 = np.array([-0.5, -0.5])
    lm_cfgs = [
        LandmarkConfig(position=pos0, is_goal=True),
        LandmarkConfig(position=pos1),
    ]
    runner = SimpleAdversaryRunner(
        num_good_agents=2,
        landmark_configs=lm_cfgs,
        max_cycles=5,
        seed=40,
    )
    # 访问底层世界
    inner = runner.env
    while hasattr(inner, "env"):
        inner = inner.env

    runner.env.reset(seed=40)

    lm0_pos = inner.world.landmarks[0].state.p_pos
    lm1_pos = inner.world.landmarks[1].state.p_pos
    runner.close()

    np.testing.assert_allclose(lm0_pos, pos0, atol=1e-6,
                                err_msg="地标 0 位置应与配置一致")
    np.testing.assert_allclose(lm1_pos, pos1, atol=1e-6,
                                err_msg="地标 1 位置应与配置一致")


def test_configurable_landmark_goal_assignment():
    """可配置场景：通过 is_goal=True 指定目标地标，验证 goal_a 指向正确的地标。"""
    lm_cfgs = [
        LandmarkConfig(is_goal=False, name="decoy"),
        LandmarkConfig(is_goal=True, name="target"),
    ]
    runner = SimpleAdversaryRunner(
        num_good_agents=2,
        landmark_configs=lm_cfgs,
        max_cycles=5,
        seed=41,
    )
    inner = runner.env
    while hasattr(inner, "env"):
        inner = inner.env
    runner.env.reset(seed=41)

    goal_name = inner.world.agents[0].goal_a.name
    runner.close()

    assert goal_name == "target", f"目标地标应为 'target'，实际为 '{goal_name}'"


def test_landmark_weights_affect_reward():
    """地标权重：验证权重不同时，奖励值发生变化（相对于默认奖励）。"""
    # 默认场景（无权重）
    runner_default = SimpleAdversaryRunner(max_cycles=10, seed=50)
    stats_default = runner_default.run_episode()
    runner_default.close()

    # 使用权重的场景（全权重给第一个地标）
    weights = {
        "agent_0": np.array([1.0, 0.0]),
        "agent_1": np.array([1.0, 0.0]),
    }
    runner_weighted = SimpleAdversaryRunner(
        num_good_agents=2,
        landmark_weights=weights,
        max_cycles=10,
        seed=50,
    )
    stats_weighted = runner_weighted.run_episode()
    runner_weighted.close()

    # 二者的合作智能体累计奖励不应完全相同（权重改变了奖励计算方式）
    default_rew = stats_default["cumulative_rewards"]["agent_0"]
    weighted_rew = stats_weighted["cumulative_rewards"]["agent_0"]
    assert np.isfinite(weighted_rew), "加权奖励应为有限值"
    assert np.isfinite(default_rew), "默认奖励应为有限值"
    # 加权奖励（仅关注第一个地标）与默认奖励（基于随机目标地标）的计算方式不同，
    # 两者的累计值应该不相等
    assert not np.isclose(weighted_rew, default_rew), (
        f"加权奖励与默认奖励不应相等 (weighted={weighted_rew:.4f}, default={default_rew:.4f})"
    )


def test_configurable_scenario_weighted_reward_direct():
    """直接测试 ConfigurableScenario 的加权奖励计算接口。"""
    lm_cfgs = [
        LandmarkConfig(position=np.array([0.3, 0.3]), is_goal=True),
        LandmarkConfig(position=np.array([-0.3, -0.3])),
    ]
    weights = {
        "agent_0": np.array([0.9, 0.1]),
        "agent_1": np.array([0.1, 0.9]),
    }
    scenario = ConfigurableScenario(
        num_good_agents=2,
        landmark_configs=lm_cfgs,
        landmark_weights=weights,
    )
    world = scenario.make_world()
    rng = np.random.default_rng(99)

    class _MockRng:
        def uniform(self, low, high, size=None):
            return rng.uniform(low, high, size)
        def choice(self, arr):
            return rng.choice(arr)

    scenario.reset_world(world, _MockRng())

    # 取 agent_0 和 agent_1
    agent0 = next(a for a in world.agents if a.name == "agent_0")
    agent1 = next(a for a in world.agents if a.name == "agent_1")

    rew0 = scenario.agent_reward(agent0, world)
    rew1 = scenario.agent_reward(agent1, world)

    assert np.isfinite(rew0), "agent_0 的加权奖励应为有限值"
    assert np.isfinite(rew1), "agent_1 的加权奖励应为有限值"

    # 两个智能体权重不同，奖励应该不同（除非位置完全一致）
    # 权重差异：agent_0=[0.9, 0.1], agent_1=[0.1, 0.9]，指向不同地标，奖励必然不同
    assert not np.isclose(rew0, rew1), (
        f"agent_0 和 agent_1 权重不同，奖励应不相等 (rew0={rew0:.6f}, rew1={rew1:.6f})"
    )

    # 验证接口：子类化 ConfigurableScenario 并覆写 compute_weighted_reward
    class SparseRewardScenario(ConfigurableScenario):
        def compute_weighted_reward(self, agent, world, weights, landmark_dists):
            weighted_dist = float(np.dot(weights, landmark_dists))
            return 1.0 if weighted_dist < 0.5 else -weighted_dist

    sparse_scenario = SparseRewardScenario(
        num_good_agents=2,
        landmark_configs=lm_cfgs,
        landmark_weights=weights,
    )
    world2 = sparse_scenario.make_world()
    sparse_scenario.reset_world(world2, _MockRng())
    agent0_w2 = next(a for a in world2.agents if a.name == "agent_0")
    sparse_rew = sparse_scenario.agent_reward(agent0_w2, world2)
    # 稀疏奖励只能返回 1.0 或 负数
    assert np.isfinite(sparse_rew), "稀疏奖励场景应返回有限值"
    assert sparse_rew == 1.0 or sparse_rew < 0, "稀疏奖励应为 1.0（近时）或负数（远时）"


def test_runner_with_configurable_and_gif(tmp_path):
    """端到端测试：3 个合作智能体 + 固定地标 + 地标权重 + GIF 保存。"""
    gif_file = str(tmp_path / "configurable_sim.gif")
    lm_cfgs = [
        LandmarkConfig(position=np.array([0.4, 0.0]), is_goal=True, name="lm_goal"),
        LandmarkConfig(position=np.array([-0.4, 0.0]), name="lm_decoy0"),
        LandmarkConfig(position=np.array([0.0, 0.4]), name="lm_decoy1"),
    ]
    weights = {
        "agent_0": np.array([0.8, 0.1, 0.1]),
        "agent_1": np.array([0.1, 0.8, 0.1]),
        "agent_2": np.array([0.1, 0.1, 0.8]),
    }
    runner = SimpleAdversaryRunner(
        num_good_agents=3,
        landmark_configs=lm_cfgs,
        landmark_weights=weights,
        max_cycles=5,
        seed=60,
        save_gif=True,
        gif_path=gif_file,
        gif_fps=3,
    )
    stats = runner.run_episode()
    runner.close()

    assert stats["total_steps"] == 5
    assert set(stats["cumulative_rewards"].keys()) == {
        "adversary_0", "agent_0", "agent_1", "agent_2"
    }
    assert os.path.isfile(gif_file), "GIF 文件应存在"
    assert os.path.getsize(gif_file) > 0


# ---------------------------------------------------------------------------
# 命令行演示入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """直接运行此脚本时，执行演示 episode（含单步调试与 GIF 保存）。

    用法::

        python test/simple_adversary_api_test.py

    """
    import tempfile

    print("====== Simple Adversary 演示 ======")

    # ── 演示 1：标准场景（N=2）──────────────────────────────────────────
    print("\n[演示 1] 标准场景（N=2，调试模式=world）")
    runner1 = SimpleAdversaryRunner(
        N=2,
        max_cycles=5,
        continuous_actions=False,
        seed=0,
        debug=True,      # 每步打印世界状态
    )
    print(f"智能体列表: {runner1.env.possible_agents}")
    stats1 = runner1.run_episode()
    runner1.close()
    print(f"\nEpisode 完成，共 {stats1['total_steps']} 步")
    print("── 累计奖励 ──")
    for agent, total_rew in stats1["cumulative_rewards"].items():
        print(f"  {agent:<14}: {total_rew:>10.4f}")

    # ── 演示 2：GIF 保存 ────────────────────────────────────────────────
    print("\n[演示 2] GIF 保存演示")
    with tempfile.TemporaryDirectory() as tmpdir:
        gif_path = os.path.join(tmpdir, "demo.gif")
        runner2 = SimpleAdversaryRunner(
            N=2, max_cycles=10, seed=1,
            save_gif=True, gif_path=gif_path, gif_fps=5,
        )
        stats2 = runner2.run_episode()
        runner2.close()
        if stats2["gif_path"]:
            size_kb = os.path.getsize(gif_path) // 1024
            print(f"  GIF 保存成功: {gif_path}（{size_kb} KB，{stats2['total_steps']} 帧）")

    # ── 演示 3：自定义地标和权重 ────────────────────────────────────────
    print("\n[演示 3] 3 个合作智能体 + 固定地标 + 权重奖励")
    lm_cfgs = [
        LandmarkConfig(position=np.array([0.5,  0.5]), is_goal=True, name="goal"),
        LandmarkConfig(position=np.array([-0.5, 0.0]), name="decoy_A"),
        LandmarkConfig(position=np.array([0.0, -0.5]), name="decoy_B"),
    ]
    weights = {
        "agent_0": np.array([0.7, 0.2, 0.1]),
        "agent_1": np.array([0.1, 0.7, 0.2]),
        "agent_2": np.array([0.2, 0.1, 0.7]),
    }
    runner3 = SimpleAdversaryRunner(
        num_good_agents=3,
        landmark_configs=lm_cfgs,
        landmark_weights=weights,
        max_cycles=10,
        seed=2,
        debug="step",    # 每个 agent 步打印动作
    )
    print(f"智能体列表: {runner3.env.possible_agents}")
    stats3 = runner3.run_episode()
    runner3.close()
    print(f"\nEpisode 完成，共 {stats3['total_steps']} 步")
    print("── 累计奖励 ──")
    for agent, total_rew in stats3["cumulative_rewards"].items():
        print(f"  {agent:<14}: {total_rew:>10.4f}")

    print("\n── 前 3 步的奖励详情 ──")
    for i, step_rew in enumerate(stats3["step_rewards"][:3]):
        print(f"  step {i + 1}: { {k: f'{v:.4f}' for k, v in step_rew.items()} }")

    print("\n── 观测维度 ──")
    for agent, obs_list in stats3["observations"].items():
        print(f"  {agent:<14}: 观测维度={obs_list[0].shape}, 总步数={len(obs_list)}")

    print("\n──────────────────────────────────")
    print("演示完成。")
