# OpenEvolve LunarLander Example (Env + Agent)

This example demonstrates how to use OpenEvolve to evolve a simple reinforcement-learning workload on Gymnasium's LunarLander environment. The workspace is split into two Git repositories (submodules) to clearly separate the stable evaluation environment from the evolving agent code.

## Why two repositories?

- Env (this repository): Stable evaluator and tooling. It should not be modified by the evolutionary process.
- Agent (submodule): The actual code under evolution. OpenEvolve operates on this Git repository and creates commits as organisms.

This separation ensures reproducible and comparable scoring while allowing free evolution of the agent implementation.

## Repository Layout

```
openevolve-example/
  evaluator.py                  # Entry point for OpenEvolve; returns metrics with 'combined_score'
  env/
    env_config.yaml             # Evaluator runtime config (episodes, steps, seed, render)
  main.py                       # Visualizer: run current agent with human/rgb_array rendering
  tune.py                       # Optuna tuning frontend (search best Agent config)
  pyproject.toml                # Env-side dependencies (gymnasium[box2d], swig)
  openevolve-example-lunarlander/  # Submodule: the evolving agent repository
    agent.py                    # Baseline agent
```

## Agent Contract

The agent repository must provide `agent.py` with the following minimal interface:

```python
class Agent:
    def __init__(self, action_space, observation_space=None, config=None):
        ...

    def reset(self):
        ...

    def act(self, observation):  # returns a valid action for the environment
        ...

    def close(self):
        ...
```

The baseline agent here is a heuristic controller with a tunable configuration. You can evolve or tune it to improve performance.

### Agent baselines (two variants)

This repository supports two agent baselines living in the submodule `openevolve-example-lunarlander`. Use the appropriate commit to reproduce the desired experiment:

1) Heuristic + hyperparameters (for Optuna tuning)
   - Submodule commit: `cec180024d02d4d11e36753b5bc88523ade2471e`
   - `agent.py` implements a richer heuristic policy and exposes a comprehensive set of hyperparameters via `default_config()`, `default_search_space()`, `suggest_config_from_trial()`, and `from_trial()`.
   - Recommended when you want to reproduce Optuna tuning results with `tune.py`.

2) Minimal random policy (for OpenEvolve autonomous evolution)
   - Submodule commit: `0cad981d6d26309ee9bbbc03919967b02f1383ea`
   - `agent.py` samples actions from the environment's action space; it serves as a simple starting point for commit-based evolutionary search.
   - Recommended when you want to reproduce OpenEvolve evolution results using `evaluator.py`.

#### How to switch the submodule to a specific baseline

From the repository root, run:

```bash
# Ensure submodule is initialized
git submodule update --init --recursive

# Switch to the Optuna-tuning baseline (heuristic with hyperparameters)
(cd openevolve-example-lunarlander && git fetch && git checkout cec180024d02d4d11e36753b5bc88523ade2471e)

# OR switch to the OpenEvolve-evolution baseline (random policy)
(cd openevolve-example-lunarlander && git fetch && git checkout 0cad981d6d26309ee9bbbc03919967b02f1383ea)
```

- After switching, you can verify by inspecting `openevolve-example-lunarlander/agent.py`.
- If you want to persist the submodule pointer in this repo, commit the change in the outer repository:

```bash
git add openevolve-example-lunarlander
git commit -m "point submodule to <baseline-commit>"
```

## Evaluator

- Entry: `evaluator.py` exposes `evaluate(repo_root: str) -> dict`
- Behavior: loads `Agent` from `repo_root/agent.py`, runs `num_episodes` of `LunarLander-v3`,
  and returns metrics including:
  - `combined_score`: mean total reward across episodes (primary fitness)
  - `mean_reward`, `std_reward`, `best_reward`, `worst_reward`, `mean_steps`, `episodes`, `runtime_sec`
- Config: set via `env/env_config.yaml` (keys: `env_id`, `num_episodes`, `max_steps`, `seed`, `render_mode`)

## Install

1) Install OpenEvolve (from the repo root):

```bash
pip install -e <path-to-openevolve-repo>
```

2) Install example dependencies (any one of the following):

```bash
# Option A: editable install of the example (uses pyproject.toml)
pip install -e <path-to-openevolve-repo>/openevolve-example

# Option B: install minimal extras directly
pip install "gymnasium[box2d]>=1.1.1" swig>=4.3.1
```

If you encounter build issues with Box2D on macOS, ensure `swig` is installed system-wide:

```bash
brew install swig
```

Optional (for tuning and plots):

```bash
# Required for tuning
pip install "optuna>=3.5.0"

# Optional for saving tuning plots
pip install "optuna[visualization]" matplotlib
```

## Visualize the Current Agent

Run the visualizer to observe the agent's behavior:

```bash
python <path-to-openevolve-repo>/openevolve-example/main.py \
  --agent-path <path-to-openevolve-repo>/openevolve-example/openevolve-example-lunarlander \
  --episodes 3 \
  --render-mode human
```

CLI flags:
- `--agent-path`: path to the agent repository (must contain `agent.py`)
- `--episodes`: number of episodes to run (default 1)
- `--max-steps`: step cap per episode (default 1000)
- `--render-mode`: `human` or `rgb_array` (default `human`)
- `--seed`: base seed; episode i uses `seed + i` (default 42)
- `--config`: path to JSON config for Agent (e.g., `openevolve-example-lunarlander/best_config.json`)

Visualize with a saved/tuned config:

```bash
python <path-to-openevolve-repo>/openevolve-example/main.py \
  --config <path-to-openevolve-repo>/openevolve-example/openevolve-example-lunarlander/best_config.json
```

## Tune the Agent with Optuna

To reproduce Optuna results, make sure the submodule is at the heuristic baseline `cec180024d02d4d11e36753b5bc88523ade2471e` (see above). Then run:

```bash
python <path-to-openevolve-repo>/openevolve-example/tune.py \
  --n-trials 80 \
  --episodes 5 \
  --max-steps 1000 \
  --seed 42
```

After tuning, visualize with the saved best config:

```bash
python <path-to-openevolve-repo>/openevolve-example/main.py \
  --config <path-to-openevolve-repo>/openevolve-example/openevolve-example-lunarlander/best_config.json
```

Resume, export, and plot:

```bash
python <path-to-openevolve-repo>/openevolve-example/tune.py \
  --n-trials 200 \
  --n-jobs 4 \
  --study-name lunarlander_tuning \
  --storage sqlite:///study.db \
  --resume \
  --export-csv trials.csv \
  --save-plots tune_plots
```

Customize the search space or pin parameters:

```bash
python <path-to-openevolve-repo>/openevolve-example/tune.py \
  --space-json path/to/space.json \
  --overrides-json path/to/pin.json
```

Outputs:
- `openevolve-example-lunarlander/best_config.json`: best-found config (merged over defaults)
- Optional: `trials.csv`, `tune_plots/` with optimization history and param importances

Notes:
- Defaults for `env_id`, `episodes`, `max_steps`, `seed` are loaded from `env/env_config.yaml` and can be overridden via CLI.
- Objective maximizes mean episode return (same metric as `evaluator.py`).

## Evolve the Agent with OpenEvolve

To reproduce OpenEvolve results, make sure the submodule is at the random baseline `0cad981d6d26309ee9bbbc03919967b02f1383ea`. Then run evolution with this example's `evaluator.py`:

```bash
python <path-to-openevolve-repo>/openevolve-run.py \
  <path-to-openevolve-repo>/openevolve-example/openevolve-example-lunarlander \
  <path-to-openevolve-repo>/openevolve-example/evaluator.py \
  --config <path-to-openevolve-repo>/configs/island_config_example.yaml \
  --iterations 50 \
  --evolution-target "Improve LunarLander mean return"
```

OpenEvolve will create commits in the agent repository, evaluate each candidate via `evaluator.py`, and track the best program by `combined_score`.

## Submodules

If you cloned this example separately and need to initialize submodules:

```bash
git submodule update --init --recursive
```

## License

This example is provided under the repository's license files.