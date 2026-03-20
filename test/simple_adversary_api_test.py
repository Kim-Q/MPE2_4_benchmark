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
- ``agent_0`` / ``agent_1`` (shape: 10)::

    [goal_rel_pos(2), landmark_rel_positions(2×2=4), other_agent_rel_positions(2×2=4)]

- ``adversary_0`` (shape: 8)::

    [landmark_rel_positions(2×2=4), other_agent_rel_positions(2×2=4)]
    # 注意：对手可以观测所有其他智能体（即两个合作智能体），共 2×2=4 维

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

       pip install mpe2 pettingzoo numpy

2. **直接运行演示**::

       python test/simple_adversary_api_test.py

3. **通过 pytest 运行所有测试**::

       pytest test/simple_adversary_api_test.py -v

4. **自定义控制算法**：
   继承 :class:`BaseAgentController` 并重写 ``get_action(observation)`` 方法，
   然后将自定义控制器注册到 :class:`SimpleAdversaryRunner` 的对应智能体槽位即可。
   如果需要修改奖励函数，继承 :class:`CustomScenario` 并覆写
   ``agent_reward``/``adversary_reward`` 方法，再传递给 ``raw_env``。
"""

from __future__ import annotations

import numpy as np
import pytest

from mpe2 import simple_adversary_v3
from mpe2.simple_adversary.simple_adversary import Scenario, raw_env


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
        合作智能体数量（同时也是地标数量），默认为 2。
    max_cycles:
        每轮的最大时间步数，默认为 25。
    continuous_actions:
        是否使用连续动作空间，默认为 ``False``（离散）。
    render_mode:
        渲染模式，``None``（不渲染）、``"human"`` 或 ``"rgb_array"``。
    seed:
        随机种子，用于环境复现。
    adversary_controller:
        ``adversary_0`` 的控制器实例，默认为 :class:`Adversary0Controller`。
    agent0_controller:
        ``agent_0`` 的控制器实例，默认为 :class:`Agent0Controller`。
    agent1_controller:
        ``agent_1`` 的控制器实例，默认为 :class:`Agent1Controller`。

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
                # 完全随机策略
                return np.random.randint(0, 5)

        runner = SimpleAdversaryRunner(
            adversary_controller=MyAdversary(),
            seed=0,
        )
        stats = runner.run_episode()

    **连续动作模式**::

        runner = SimpleAdversaryRunner(continuous_actions=True, seed=0)
        stats = runner.run_episode()

    **使用自定义奖励场景**::

        # 通过 raw_env 直接使用 CustomScenario（绕过工厂函数）
        # 见 test_custom_scenario_reward_weights() 测试
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
    ) -> None:
        self.N = N
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode
        self.seed = seed

        # 注册控制器（若未提供则使用默认实现）
        self.controllers: dict[str, BaseAgentController] = {
            "adversary_0": adversary_controller
            or Adversary0Controller(continuous_actions),
            "agent_0": agent0_controller or Agent0Controller(continuous_actions),
            "agent_1": agent1_controller or Agent1Controller(continuous_actions),
        }

        # 创建环境
        self.env = simple_adversary_v3.env(
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )

    def run_episode(self) -> dict:
        """运行一个完整的 episode 并收集统计信息。

        Returns
        -------
        stats : dict
            包含以下键：

            - ``cumulative_rewards`` (dict[str, float])：每个智能体的累计奖励
            - ``step_rewards`` (list[dict[str, float]])：每个世界时间步的奖励
            - ``total_steps`` (int)：执行的世界步数（= AEC 步数 / agent 数量）
            - ``observations`` (dict[str, list[np.ndarray]])：每个智能体所有时间步的观测
        """
        self.env.reset(seed=self.seed)

        cumulative_rewards: dict[str, float] = {
            agent: 0.0 for agent in self.env.possible_agents
        }
        step_rewards: list[dict[str, float]] = []
        observations: dict[str, list[np.ndarray]] = {
            agent: [] for agent in self.env.possible_agents
        }

        # AEC 循环：每次迭代对应一个智能体的行动
        world_steps = 0
        last_agent = self.env.possible_agents[-1]

        for agent_name in self.env.agent_iter():
            obs, reward, termination, truncation, info = self.env.last()

            # 记录观测与奖励
            observations[agent_name].append(obs.copy())
            cumulative_rewards[agent_name] += reward

            if termination or truncation:
                # 智能体已终止，传入 None
                self.env.step(None)
            else:
                # 使用对应控制器生成动作
                controller = self.controllers.get(agent_name)
                if controller is not None:
                    action = controller.get_action(obs)
                else:
                    # 未注册控制器时退回随机策略
                    action = self.env.action_space(agent_name).sample()
                self.env.step(action)

                # 最后一个智能体行动后世界完成一步，记录本步奖励
                if agent_name == last_agent:
                    world_steps += 1
                    step_rewards.append(dict(self.env.rewards))

        total_steps = world_steps
        return {
            "cumulative_rewards": cumulative_rewards,
            "step_rewards": step_rewards,
            "total_steps": total_steps,
            "observations": observations,
        }

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

    class _Rng:
        """轻量级 np_random 兼容对象。"""
        def uniform(self, low, high, size=None):
            return rng.uniform(low, high, size)
        def choice(self, arr):
            return rng.choice(arr)

    scenario_1x.reset_world(world, _Rng())

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
# 命令行演示入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """直接运行此脚本时，执行一次带统计输出的演示 episode。

    用法::

        python test/simple_adversary_api_test.py

    输出示例::

        ====== Simple Adversary 演示 ======
        智能体列表: ['adversary_0', 'agent_0', 'agent_1']
        运行 episode（max_cycles=25, seed=0）...
        Episode 完成，共 25 步
        ── 累计奖励 ──
          adversary_0 : -12.3456
          agent_0     :   3.2100
          agent_1     :   2.9876
        ──────────────────────────────────
    """
    print("====== Simple Adversary 演示 ======")

    runner = SimpleAdversaryRunner(
        N=2,
        max_cycles=25,
        continuous_actions=False,
        seed=0,
    )

    print(f"智能体列表: {runner.env.possible_agents}")
    print(f"运行 episode（max_cycles={runner.max_cycles}, seed={runner.seed}）...")

    stats = runner.run_episode()
    runner.close()

    print(f"Episode 完成，共 {stats['total_steps']} 步")
    print("── 累计奖励 ──")
    for agent, total_rew in stats["cumulative_rewards"].items():
        print(f"  {agent:<14}: {total_rew:>10.4f}")

    print("\n── 前 5 步的奖励详情 ──")
    for i, step_rew in enumerate(stats["step_rewards"][:5]):
        print(f"  step {i + 1}: { {k: f'{v:.4f}' for k, v in step_rew.items()} }")

    print("\n── 观测维度 ──")
    for agent, obs_list in stats["observations"].items():
        print(f"  {agent:<14}: 观测维度={obs_list[0].shape}, 总步数={len(obs_list)}")

    print("──────────────────────────────────")
    print("演示完成。")
