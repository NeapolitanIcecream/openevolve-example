from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import importlib.util
import threading


# -------------------------
# Utilities
# -------------------------

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


def _merge_env_defaults(cli_defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Load env defaults from env/env_config.yaml and merge onto provided defaults.

    Recognized keys: env_id, num_episodes, max_steps, seed.
    """
    base = {
        "env_id": "LunarLander-v3",
        "num_episodes": 5,
        "max_steps": 1000,
        "seed": 42,
    }
    try:
        cfg_path = Path(__file__).parent / "env" / "env_config.yaml"
        overrides = _load_yaml_config(cfg_path)
        if overrides:
            for k in ["env_id", "num_episodes", "max_steps", "seed"]:
                v = overrides.get(k)
                if v is not None:
                    base[k] = v
    except Exception:
        pass
    base.update({k: v for k, v in cli_defaults.items() if v is not None})
    return base


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


# Map threads to stable worker indices to offset seeds per parallel job
_worker_map_lock = threading.Lock()
_thread_to_worker_index: Dict[int, int] = {}
_next_worker_index = 0


def _get_worker_index(n_jobs: int) -> int:
    try:
        tid = threading.get_ident()
        global _next_worker_index
        with _worker_map_lock:
            if tid not in _thread_to_worker_index:
                _thread_to_worker_index[tid] = _next_worker_index
                _next_worker_index += 1
            idx = _thread_to_worker_index[tid]
            return idx % max(1, int(n_jobs)) if n_jobs else idx
    except Exception:
        return 0


# -------------------------
# Objective callable
# -------------------------

@dataclass
class Objective:
    agent_repo_dir: Path
    episodes: int
    max_steps: int
    seed: Optional[int]
    env_id: str
    overrides: Optional[Dict[str, Any]]
    space: Optional[Dict[str, Any]]
    n_jobs: int

    def __call__(self, trial):  # type: ignore[override]
        import gymnasium as gym
        import optuna

        env = gym.make(self.env_id)
        Agent = _load_agent_class(self.agent_repo_dir)
        agent = None
        try:
            # Build agent using trial-suggested config
            if hasattr(Agent, "from_trial"):
                agent = Agent.from_trial(
                    env.action_space,
                    getattr(env, "observation_space", None),
                    trial=trial,
                    overrides=self.overrides,
                    space=self.space,
                )
            else:
                # Fallback: suggest from default_search_space
                if hasattr(Agent, "default_search_space") and hasattr(Agent, "default_config"):
                    spec = Agent.default_search_space()
                    if isinstance(self.space, dict) and self.space:
                        spec.update(self.space)
                    cfg = Agent.default_config()
                    for key, meta in spec.items():
                        t = meta.get("type") if isinstance(meta, dict) else None
                        if t == "float":
                            low_v = meta.get("low") if isinstance(meta, dict) else None
                            high_v = meta.get("high") if isinstance(meta, dict) else None
                            if low_v is None or high_v is None:
                                continue
                            step_v = meta.get("step") if isinstance(meta, dict) else None
                            low_f = float(low_v)
                            high_f = float(high_v)
                            if step_v is None:
                                cfg[key] = trial.suggest_float(key, low_f, high_f)
                            else:
                                cfg[key] = trial.suggest_float(key, low_f, high_f, step=float(step_v))
                        elif t == "int":
                            low_v = meta.get("low") if isinstance(meta, dict) else None
                            high_v = meta.get("high") if isinstance(meta, dict) else None
                            if low_v is None or high_v is None:
                                continue
                            step_v = meta.get("step") if isinstance(meta, dict) else None
                            step_i = int(step_v) if step_v is not None else 1
                            low_i = int(low_v)
                            high_i = int(high_v)
                            cfg[key] = trial.suggest_int(key, low_i, high_i, step=step_i)
                        elif t == "categorical":
                            choices = list(meta.get("choices", [])) if isinstance(meta, dict) else []
                            if choices:
                                cfg[key] = trial.suggest_categorical(key, choices)
                    if isinstance(self.overrides, dict) and self.overrides:
                        cfg.update(self.overrides)
                    agent = Agent(env.action_space, getattr(env, "observation_space", None), config=cfg)
                else:
                    # Last resort: plain agent with no tuning
                    agent = Agent(env.action_space, getattr(env, "observation_space", None), config=None)

            # Compute worker-dependent seed base to avoid duplication across parallel jobs
            worker_index = _get_worker_index(self.n_jobs)
            base_seed = (self.seed or 0) + (worker_index * self.episodes)

            episode_rewards = []
            for i in range(self.episodes):
                s = base_seed + i if self.seed is not None else None
                try:
                    obs, _info = env.reset(seed=s)
                except TypeError:
                    obs, _info = env.reset()

                if hasattr(agent, "reset"):
                    agent.reset()

                total_r = 0.0
                steps = 0
                terminated = False
                truncated = False
                while not (terminated or truncated):
                    action = agent.act(obs) if hasattr(agent, "act") else env.action_space.sample()
                    obs, r, terminated, truncated, _info = env.step(action)
                    total_r += float(r)
                    steps += 1
                    if steps >= self.max_steps:
                        break

                episode_rewards.append(total_r)
                # Report per-episode to enable pruning
                try:
                    trial.report(total_r, i)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                except Exception:
                    # Ignore pruning/report issues if any
                    pass

            # Maximize mean return
            mean_r = float(sum(episode_rewards) / max(len(episode_rewards), 1))
            return mean_r
        finally:
            try:
                if agent is not None and hasattr(agent, "close"):
                    agent.close()
            except Exception:
                pass
            env.close()


# -------------------------
# CLI / Runner
# -------------------------

def _load_space_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Search space JSON not found: {p}")
    with open(p, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Search space JSON must be an object mapping param -> spec")
    return data


def _load_overrides_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Overrides JSON not found: {p}")
    with open(p, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Overrides JSON must be an object mapping param -> value")
    return data


def _build_sampler(name: str, seed: Optional[int]):
    import optuna
    n = (name or "tpe").lower()
    if n == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if n == "cmaes":
        try:
            return optuna.samplers.CmaEsSampler(seed=seed)
        except Exception:
            return optuna.samplers.TPESampler(seed=seed)
    # default: TPE
    return optuna.samplers.TPESampler(seed=seed)


def _build_pruner(name: str, warmup_steps: int = 1):
    import optuna
    n = (name or "median").lower()
    if n in ("none", "nop"):
        return optuna.pruners.NopPruner()
    if n == "asha":
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
    # default: median
    return optuna.pruners.MedianPruner(n_warmup_steps=max(1, int(warmup_steps)))


def _export_trials_csv(study, csv_path: Path) -> None:
    import csv
    rows = []
    for t in study.trials:
        row = {
            "number": t.number,
            "state": str(t.state),
            "value": t.value if t.value is not None else "",
            "duration_sec": t.duration.total_seconds() if t.duration else "",
            "params": json.dumps(t.params, ensure_ascii=False),
        }
        rows.append(row)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["number", "state", "value", "duration_sec", "params"]) 
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _save_best_config(agent_repo_dir: Path, best_params: Dict[str, Any], overrides: Optional[Dict[str, Any]], out_path: Path) -> Dict[str, Any]:
    Agent = _load_agent_class(agent_repo_dir)
    cfg = {}
    if hasattr(Agent, "default_config"):
        cfg = Agent.default_config()
        if not isinstance(cfg, dict):
            cfg = {}
    cfg.update(best_params or {})
    if isinstance(overrides, dict) and overrides:
        cfg.update(overrides)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return cfg


def _maybe_save_plots(study, out_dir: Optional[str]) -> None:
    if not out_dir:
        return
    try:
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from typing import Any, Callable

        def _save_plot_object(obj: Any, outfile: Path) -> None:
            try:
                # Direct save if object supports it
                if hasattr(obj, "savefig"):
                    obj.savefig(str(outfile), dpi=150, bbox_inches="tight")
                    try:
                        plt.close()
                    except Exception:
                        pass
                    return
                # Axes-like: has .figure
                fig_attr = getattr(obj, "figure", None)
                if fig_attr is not None and hasattr(fig_attr, "savefig"):
                    fig_attr.savefig(str(outfile), dpi=150, bbox_inches="tight")
                    try:
                        plt.close()
                    except Exception:
                        pass
                    return
                # Fallback via get_figure
                getf: Optional[Callable[[], Any]] = getattr(obj, "get_figure", None)
                if callable(getf):
                    fig_obj = getf()
                    if fig_obj is not None and hasattr(fig_obj, "savefig"):
                        fig_obj.savefig(str(outfile), dpi=150, bbox_inches="tight")
                        try:
                            plt.close()
                        except Exception:
                            pass
            except Exception:
                pass

        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)

        ax1 = plot_optimization_history(study)
        _save_plot_object(ax1, d / "optimization_history.png")

        try:
            ax2 = plot_param_importances(study)
            _save_plot_object(ax2, d / "param_importances.png")
        except Exception:
            pass
    except Exception:
        # Visualization dependencies not available; ignore silently
        pass


def main():
    # Load env defaults from YAML, then feed into CLI defaults
    env_defaults = _merge_env_defaults({})

    parser = argparse.ArgumentParser(description="Optuna tuning frontend for LunarLander Agent")
    default_agent_dir = Path(__file__).parent / "openevolve-example-lunarlander"

    parser.add_argument("--agent-path", type=str, default=str(default_agent_dir), help="Path to agent repository directory (must contain agent.py)")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=float, default=None, help="Time budget in seconds (overrides n-trials if hit)")

    parser.add_argument("--episodes", type=int, default=int(env_defaults.get("num_episodes", 5)), help="Episodes per trial")
    parser.add_argument("--max-steps", type=int, default=int(env_defaults.get("max_steps", 1000)), help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=int(env_defaults.get("seed", 42)), help="Base seed (episode i uses seed+i)")
    parser.add_argument("--env-id", type=str, default=str(env_defaults.get("env_id", "LunarLander-v3")), help="Gymnasium environment id")

    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for Optuna (>=1)")
    parser.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random", "cmaes"], help="Sampler algorithm")
    parser.add_argument("--pruner", type=str, default="median", choices=["none", "median", "asha"], help="Pruning strategy")

    parser.add_argument("--storage", type=str, default="sqlite:///study.db", help="Optuna storage URL (e.g., sqlite:///study.db)")
    parser.add_argument("--study-name", type=str, default="lunarlander_tuning", help="Optuna study name")
    parser.add_argument("--resume", action="store_true", help="Resume study if it exists")

    parser.add_argument("--space-json", type=str, default=None, help="Path to JSON file overriding/defining search space")
    parser.add_argument("--overrides-json", type=str, default=None, help="Path to JSON file with fixed overrides applied after suggestion")

    parser.add_argument("--save-best-config", type=str, default=str(default_agent_dir / "best_config.json"), help="Path to save best config JSON")
    parser.add_argument("--export-csv", type=str, default=None, help="Path to export trials CSV (optional)")
    parser.add_argument("--save-plots", type=str, default=None, help="Directory to save optimization plots (optional)")

    args = parser.parse_args()

    agent_repo_dir = Path(args.agent_path).resolve()
    space = _load_space_json(args.space_json)
    overrides = _load_overrides_json(args.overrides_json)

    import optuna

    sampler = _build_sampler(args.sampler, args.seed)
    pruner = _build_pruner(args.pruner, warmup_steps=max(1, args.episodes // 2))

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=bool(args.resume),
    )

    objective = Objective(
        agent_repo_dir=agent_repo_dir,
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        seed=int(args.seed) if args.seed is not None else None,
        env_id=str(args.env_id),
        overrides=overrides,
        space=space,
        n_jobs=int(args.n_jobs),
    )

    t0 = time.time()
    study.optimize(objective, n_trials=int(args.n_trials), timeout=args.timeout, n_jobs=int(args.n_jobs), gc_after_trial=True, show_progress_bar=False)
    elapsed = time.time() - t0

    best = study.best_trial
    print(f"Best value: {best.value:.4f} (trial #{best.number}) in {elapsed:.1f}s")

    # Persist best config
    best_config_path = Path(args.save_best_config).resolve()
    saved_cfg = _save_best_config(agent_repo_dir, best.params, overrides, best_config_path)
    print(f"Saved best_config.json with {len(saved_cfg)} keys -> {best_config_path}")

    # Export CSV if requested
    if args.export_csv:
        csv_path = Path(args.export_csv).resolve()
        _export_trials_csv(study, csv_path)
        print(f"Exported trials CSV -> {csv_path}")

    # Optional plots
    if args.save_plots:
        _maybe_save_plots(study, args.save_plots)
        print(f"Saved plots to: {args.save_plots}")


if __name__ == "__main__":
    main()
