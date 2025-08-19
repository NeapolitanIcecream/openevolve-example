from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional


def _load_agent_class(agent_repo_dir: Path):
    agent_file = agent_repo_dir / "agent.py"
    if not agent_file.exists():
        raise FileNotFoundError(f"Agent not found: {agent_file}")
    spec = importlib.util.spec_from_file_location("agent_module", str(agent_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load agent module from {agent_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "Agent"):
        raise AttributeError("agent.py must define a class named 'Agent'")
    return getattr(module, "Agent")


def visualize(agent_repo_dir: Path, episodes: int, max_steps: int, render_mode: str, seed: Optional[int]) -> None:
    import gymnasium as gym

    env = gym.make("LunarLander-v3", render_mode=render_mode)
    Agent = _load_agent_class(agent_repo_dir)
    agent = Agent(env.action_space, getattr(env, "observation_space", None), config=None)

    try:
        for ep in range(episodes):
            s = (seed + ep) if seed is not None else None
            try:
                obs, _info = env.reset(seed=s)
            except TypeError:
                obs, _info = env.reset()

            if hasattr(agent, "reset"):
                agent.reset()

            terminated = False
            truncated = False
            total_r = 0.0
            steps = 0
            while not (terminated or truncated):
                action = agent.act(obs) if hasattr(agent, "act") else env.action_space.sample()
                obs, r, terminated, truncated, _info = env.step(action)
                total_r += float(r)
                steps += 1
                if steps >= max_steps:
                    break

            print(f"Episode {ep+1}/{episodes}: return={total_r:.2f}, steps={steps}")
    finally:
        try:
            if hasattr(agent, "close"):
                agent.close()
        except Exception:
            pass
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize current agent behavior (LunarLander)")
    default_agent_dir = Path(__file__).parent / "openevolve-example-lunarlander"
    parser.add_argument("--agent-path", type=str, default=str(default_agent_dir), help="Path to agent repository directory (must contain agent.py)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "rgb_array"], help="Render mode for Gym environment")
    parser.add_argument("--seed", type=int, default=42, help="Base seed (episode i uses seed+i)")
    args = parser.parse_args()

    agent_repo_dir = Path(args.agent_path).resolve()
    visualize(agent_repo_dir, episodes=args.episodes, max_steps=args.max_steps, render_mode=args.render_mode, seed=args.seed)


if __name__ == "__main__":
    main()
