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
  pyproject.toml                # Env-side dependencies (gymnasium[box2d], swig)
  openevolve-example-lunarlander/  # Submodule: the evolving agent repository
    agent.py                    # Minimal baseline agent (random policy)
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

The baseline provided here samples random actions. OpenEvolve will evolve commits in this repository to improve performance.

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

## Evolve the Agent with OpenEvolve

Use the agent submodule as the evolving repo and this evaluator as the scoring entry:

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