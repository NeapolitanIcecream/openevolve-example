"""
LunarLander evaluator for OpenEvolve (env side)

Contract:
- Exposes evaluate(repo_root: str) -> dict
- Loads Agent from the evolving repository root (repo_root/agent.py)
- Runs N episodes in Gymnasium LunarLander-v3 and returns metrics with 'combined_score'

Notes:
- Keep this file outside the evolving repo. It is stable and not subject to evolution.
- Defaults can be overridden via env/env_config.yaml colocated with this file.
"""

from __future__ import annotations

import importlib.util
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class EvalConfig:
    env_id: str = "LunarLander-v3"
    num_episodes: int = 5
    max_steps: int = 1000
    seed: int = 42
    render_mode: Optional[str] = None  # "rgb_array" | "human" | None


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        if config_path.exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
        return {}
    except Exception:
        return {}


def _merge_config(defaults: EvalConfig, overrides: Dict[str, Any]) -> EvalConfig:
    cfg = EvalConfig(**{**defaults.__dict__, **{k: v for k, v in (overrides or {}).items() if v is not None}})
    return cfg


def _load_agent_class(repo_root: Path):
    agent_path = repo_root / "agent.py"
    if not agent_path.exists():
        raise FileNotFoundError(f"Agent file not found: {agent_path}")

    spec = importlib.util.spec_from_file_location("agent_module", str(agent_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load agent module from {agent_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    if not hasattr(module, "Agent"):
        raise AttributeError("agent.py must define a class named 'Agent'")

    return getattr(module, "Agent")


def _make_env(env_id: str, render_mode: Optional[str]):
    import gymnasium as gym  # Imported lazily to keep openevolve core lightweight

    kwargs: Dict[str, Any] = {}
    if render_mode:
        kwargs["render_mode"] = render_mode
    return gym.make(env_id, **kwargs)


def evaluate(repo_root: str) -> Dict[str, Any]:
    start_time = time.time()
    eval_dir = Path(__file__).parent
    repo_path = Path(repo_root)

    defaults = EvalConfig()
    overrides = _load_yaml_config(eval_dir / "env" / "env_config.yaml")
    cfg = _merge_config(defaults, overrides)

    env = None
    agent = None
    try:
        env = _make_env(cfg.env_id, cfg.render_mode)

        AgentCls = _load_agent_class(repo_path)
        agent = AgentCls(env.action_space, getattr(env, "observation_space", None), config=None)

        episode_rewards = []
        episode_steps = []

        for i in range(cfg.num_episodes):
            # Seed per episode for stability; action sampling randomness can still vary
            seed_i = (cfg.seed or 0) + i
            try:
                obs, _info = env.reset(seed=seed_i)
            except TypeError:
                # Older gymnasium versions might not accept seed kwarg
                obs, _info = env.reset()

            # Give the agent a chance to reset internal state
            if hasattr(agent, "reset"):
                agent.reset()

            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.act(obs) if hasattr(agent, "act") else env.action_space.sample()
                obs, reward, terminated, truncated, _info = env.step(action)
                total_reward += float(reward)
                steps += 1
                if steps >= cfg.max_steps:
                    break

            episode_rewards.append(total_reward)
            episode_steps.append(steps)

        mean_r = float(np.mean(episode_rewards)) if episode_rewards else float("nan")
        std_r = float(np.std(episode_rewards)) if len(episode_rewards) > 1 else 0.0
        best_r = float(np.max(episode_rewards)) if episode_rewards else float("nan")
        worst_r = float(np.min(episode_rewards)) if episode_rewards else float("nan")
        mean_steps = float(np.mean(episode_steps)) if episode_steps else 0.0

        duration = time.time() - start_time

        return {
            "combined_score": mean_r,
            "mean_reward": mean_r,
            "std_reward": std_r,
            "best_reward": best_r,
            "worst_reward": worst_r,
            "mean_steps": mean_steps,
            "episodes": int(cfg.num_episodes),
            "runtime_sec": float(duration),
        }

    except Exception as e:
        duration = time.time() - start_time
        # Penalize failures with a very low score to guide evolution away from errors
        return {
            "combined_score": -1e9,
            "error": str(e),
            "traceback": traceback.format_exc(limit=4),
            "runtime_sec": float(duration),
        }
    finally:
        try:
            if agent is not None and hasattr(agent, "close"):
                agent.close()
        except Exception:
            pass
        try:
            if env is not None:
                env.close()
        except Exception:
            pass


