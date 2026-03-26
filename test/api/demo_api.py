from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from env_api import *
from controller_api import *

from pathlib import Path
DEFAULT_GIF_DIR = Path(__file__).parent

def _unwrap_to_world(env):
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    return inner.world

def render_stable_frame_from_world(world, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), figsize=(6, 6), dpi=150):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    # 固定世界坐标范围，不跟随对象缩放
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    # 可选：固定边距，避免自动tight造成视觉跳动
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.96)

    for lm in world.landmarks:
        ax.scatter(lm.state.p_pos[0], lm.state.p_pos[1], c="green", s=160, marker="s")
        ax.text(lm.state.p_pos[0] + 0.02, lm.state.p_pos[1] + 0.02, lm.name, fontsize=8)

    for a in world.agents:
        c = "red" if a.adversary else "blue"
        ax.scatter(a.state.p_pos[0], a.state.p_pos[1], c=c, s=100)
        ax.text(a.state.p_pos[0] + 0.02, a.state.p_pos[1] + 0.02, a.name, fontsize=8)

    canvas = FigureCanvas(fig)
    canvas.draw()

    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)  # (H,W,4)
    data = rgba[:, :, :3].copy()  # RGB

    plt.close(fig)
    return data

def save_episode_gif(frames: list[np.ndarray], gif_path: str, fps: int = 5) -> str:
    import imageio.v2 as imageio
    if not frames:
        raise ValueError("frames为空")
    os.makedirs(os.path.dirname(os.path.abspath(gif_path)), exist_ok=True)
    # 这里所有帧已由固定画布产生，shape天然一致
    imageio.mimsave(
        gif_path,
        frames,
        format="GIF",
        duration=1.0 / max(fps, 1),
        loop=0,
        subrectangles=False,   # 禁止局部块更新
    )
    return gif_path


def run_episode(max_cycles=25, seed=0, save_gif=True, gif_path="outputs/episode.gif", gif_fps=5, controllers=None):
    lm_cfgs = [
        LandmarkConfig(position=np.array([0.6, 0.0]), name="goal_0"),
        LandmarkConfig(position=np.array([-0.6, 0.0]), name="goal_1"),
    ]
    follower_goal_weights = {
        "agent_0": np.array([0.8, 0.2]),
        "agent_1": np.array([0.2, 0.8]),
    }

    env = build_custom_env(
        num_good_agents=2,
        landmark_configs=lm_cfgs,
        follower_goal_weights=follower_goal_weights,
        max_cycles=max_cycles,
        continuous_actions=True,
        render_mode=None,  # 不用内置render，完全自绘
    )

    if controllers is None:
        controllers = {
            "adversary_0": LeaderController(rng=np.random.default_rng(seed + 101)),
            "agent_0": FollowerController(rng=np.random.default_rng(seed + 201)),
            "agent_1": FollowerController(rng=np.random.default_rng(seed + 301)),
        }

    env.reset(seed=seed)
    cumulative_rewards = {a: 0.0 for a in env.possible_agents}
    step_rewards = []
    frames = []
    last_agent = env.possible_agents[-1]

    # 初始帧
    world = _unwrap_to_world(env)
    frames.append(render_stable_frame_from_world(world))

    for agent in env.agent_iter():
        obs, rew, term, trunc, _ = env.last()
        cumulative_rewards[agent] += rew

        if term or trunc:
            env.step(None)
        else:
            action = controllers[agent].get_action(obs)
            env.step(action)

            if agent == last_agent:
                world = _unwrap_to_world(env)

                # 关键：限制所有agent位置，确保都在画布内
                clamp_world_agents(world, low=-1.0, high=1.0)

                step_rewards.append(dict(env.rewards))
                frames.append(render_stable_frame_from_world(world, xlim=(-1.1, 1.1), ylim=(-1.1, 1.1)))

    env.close()
    gif_out = save_episode_gif(frames, gif_path, gif_fps) if save_gif else None
    plt_out = plot_rewards(step_rewards, fig_path="outputs/episode_rewards.png")
    return {"cumulative_rewards": cumulative_rewards, "step_rewards": step_rewards, "gif_path": gif_out, "plt_path": plt_out}

def clamp_world_agents(world, low=-1.0, high=1.0):
    for a in world.agents:
        a.state.p_pos = np.clip(a.state.p_pos, low, high)

def plot_rewards(step_rewards: list[dict[str, float]], fig_path: str = "outputs/episode_rewards.png") -> str | None:
    if not step_rewards:
        return None
    x = np.arange(1, len(step_rewards) + 1)
    plt.figure(figsize=(8, 4))
    for agent in sorted(step_rewards[0].keys()):
        y = [sr.get(agent, 0.0) for sr in step_rewards]
        plt.plot(x, y, label=agent)
    plt.xlabel("World Step")
    plt.ylabel("Reward")
    plt.title("Episode Rewards")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def run_rl_episode(
    algo: str = "ppo",
    num_episodes: int = 200,
    max_cycles: int = 25,
    seed: int = 0,
    save_gif: bool = True,
    gif_path: str | None = None,
    gif_fps: int = 5,
    verbose: bool = False,
    **algo_kwargs,
) -> dict:
    """Train RL controllers then run one evaluation episode.

    This is the main entry point for running any RL algorithm end-to-end
    from ``demo_api``.

    Parameters
    ----------
    algo : str
        Algorithm name – one of ``"ppo"``, ``"trpo"``, ``"a2c"``,
        ``"entropy_rl"``, ``"reinforce"``.
        See :data:`controller_api.AVAILABLE_ALGOS` for the full list.
    num_episodes : int
        Number of training episodes.
    max_cycles : int
        Steps per evaluation episode.
    seed : int
        Master random seed.
    save_gif : bool
        Whether to save the episode as a GIF.
    gif_path : str | None
        Output GIF path.  When ``None`` (default), saves to
        ``outputs/<algo>_episode.gif``.
    gif_fps : int
        Frames per second for the GIF.
    verbose : bool
        Print training progress.
    **algo_kwargs
        Extra keyword arguments forwarded to the algorithm benchmark
        constructor (e.g. ``lr=1e-3``, ``clip_eps=0.1`` for PPO).

    Returns
    -------
    dict with keys ``"cumulative_rewards"``, ``"step_rewards"``,
    ``"gif_path"``, ``"plt_path"``.

    Examples
    --------
    ::

        from demo_api import run_rl_episode
        out = run_rl_episode("ppo", num_episodes=100, seed=0)
        print(out["cumulative_rewards"])
    """
    if gif_path is None:
        gif_path = f"outputs/{algo.lower()}_episode.gif"

    controllers = build_rl_controllers(
        algo=algo,
        num_episodes=num_episodes,
        seed=seed,
        verbose=verbose,
        **algo_kwargs,
    )
    return run_episode(
        max_cycles=max_cycles,
        seed=seed,
        save_gif=save_gif,
        gif_path=gif_path,
        gif_fps=gif_fps,
        controllers=controllers,
    )


if __name__ == "__main__":
    out = run_episode(max_cycles=20, seed=0, save_gif=True, gif_path="outputs/episode.gif", gif_fps=5)
    fig = plot_rewards(out["step_rewards"], fig_path="outputs/episode_rewards.png")
    print("GIF:", out["gif_path"])
    print("FIG:", fig)