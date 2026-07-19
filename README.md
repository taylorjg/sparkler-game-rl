# Sparkler RL

Reinforcement learning spin-off of [sparkler-game-phaser](https://github.com/taylorjg/sparkler-game-phaser). Trains an agent to play the sparkler obstacle game in a headless Python simulator, then exports the policy for the Phaser browser game.

## Stack

| Component | Choice |
|-----------|--------|
| Simulator | Python port of Phaser game physics |
| RL API | [Gymnasium](https://gymnasium.farama.org/) |
| Training | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO) |
| Runtime | Docker (Python 3.12) |

## Quick start

```bash
# Run tests
docker compose run --rm test

# Evaluate rule-based baseline
docker compose run --rm heuristic

# Train BC agent (recommended)
docker compose run --rm bc-only

# Compare heuristic vs BC vs BC+PPO
docker compose run --rm eval

# Train PPO from scratch (slow; mostly a baseline)
docker compose run --rm train

# BC pre-train + PPO fine-tune (experimental)
docker compose run --rm bc-train

# Interactive shell
docker compose run --rm dev bash

# Export BC policy to JSON for the Phaser game
docker compose run --rm dev python -m train.export_policy
```

The export writes `models/sparkler_bc.json`. Copy it to `sparkler-game-phaser/public/assets/models/` and enable agent mode from the robot icon in the Phaser game.

## How training works

### Atari-style pixel RL (not used here)

Classic **Atari-style** game RL — the approach made famous by DeepMind’s DQN on the [Arcade Learning Environment (ALE)](https://github.com/Farama-Foundation/Arcade-Learning-Environment) — feeds the policy **raw screen pixels**, not game variables. Typically the agent sees a short stack of downsampled grayscale frames (for example 84×84×4) and a **convolutional neural network (CNN)** learns features and control end-to-end from those images alone.

**We are not doing that in this project.** Observations are a fixed vector of **7 normalised numbers** (ship height, velocity, gap geometry, and so on) from a headless physics simulator, and the policy is a small **MLP** (`MlpPolicy`).

Why:

| Pixel / CNN approach | This project’s approach |
|----------------------|-------------------------|
| Learns from rendered frames; needs a visual pipeline (render → preprocess → stack) | Reads structured state directly from the simulator |
| Sample-hungry — often millions of frames and GPU time | Trains in minutes on CPU with behavioural cloning |
| Hard to align training pixels with the live Phaser renderer (resolution, particles, rounded obstacles) | Same 7 features are built in Python (`simulator.py`) and TypeScript (`agent-observation.ts`) |
| CNN weights are heavy to run in the browser | BC exports a tiny MLP to JSON for in-game inference |
| Sparse rewards make pure RL slow on a game like this | A tuned rule-based expert + BC gave ~14 obstacles/ep quickly |

Pixel RL is the right default when you **cannot** expose meaningful state — classic Atari games hide RAM labels and you only get the bitmap. Sparkler is different: the physics are known, the gap and ship positions are available, and the goal includes **deploying the same policy back into Phaser**. Hand-crafted features plus imitation learning match that setup better than treating the problem like Breakout from pixels alone.

### What is PPO?

**PPO** (Proximal Policy Optimization) is a reinforcement learning algorithm that learns a **policy** — a function that decides what to do in each situation. In this project, the policy is a small neural network (`MlpPolicy`) that takes the current game state as input and outputs either **flap** or **don't flap**.

PPO is a good first choice for games like this because it:

- Handles discrete actions (flap / no flap) naturally
- Is stable and well-understood — the default algorithm in Stable-Baselines3 for this kind of problem
- Learns from trial and error: play episodes, observe rewards, adjust the network to favour actions that led to higher total reward

At a high level, each training step looks like this:

1. The agent plays the game using its current (imperfect) policy
2. It collects **observations**, **actions**, and **rewards** from those episodes
3. PPO updates the network to increase the probability of actions that worked well, while limiting how much the policy changes per update (the "proximal" part — this avoids catastrophic forgetting)

### The learning problem

The agent interacts with `SparklerEnv`, a [Gymnasium](https://gymnasium.farama.org/) wrapper around the headless simulator. Each **step** is one game frame (~60 fps).

**Actions** (discrete):

| Value | Meaning |
|-------|---------|
| `0` | Do nothing |
| `1` | Flap (short thrust burst, same as tap/up-arrow in the Phaser game) |

**Observations** (7 numbers, all normalised):

| Feature | What it tells the agent |
|---------|-------------------------|
| Ship height | Vertical position |
| Vertical velocity | How fast the ship is moving up or down |
| Gap centre | Where the safe passage is |
| Gap half-height | How tight the gap is |
| Distance to obstacle | How far ahead the next obstacle is |
| Scroll speed | How fast the world is moving (ramps over time) |
| Thrust active | Whether a flap burst is still in progress |

**Rewards** (defined in `sparkler/constants.py`):

| Event | Reward |
|-------|--------|
| Each frame survived | `0` |
| Obstacle cleared | `+5.0` |
| Collision (death) | `-1.0` |

An episode ends on collision. The agent's goal is to maximise **total reward** over an episode, which encourages staying alive and clearing obstacles.

### What `train_ppo.py` does

When you run `docker compose run --rm train`:

1. **Creates 8 parallel environments** — eight copies of the game run simultaneously to collect experience faster
2. **Initialises a PPO model** with an untrained neural network
3. **Runs 500,000 environment steps** (configurable with `--timesteps`), updating the policy as it goes
4. **Logs metrics to TensorBoard** in `logs/` (learning curves, episode reward, etc.)
5. **Evaluates the final policy** over 20 episodes and prints mean reward
6. **Saves the trained model** to `models/sparkler_ppo`

Customise training from inside the container:

```bash
docker compose run --rm train python -m train.train_ppo --timesteps 1000000 --n-envs 16 --seed 42
```

View training progress (with TensorBoard installed locally, or via a dev container):

```bash
docker compose up tensorboard
```

Then open http://localhost:6006

### Baseline comparison

Before (or alongside) training, run the rule-based heuristic to see what a simple hand-crafted strategy achieves:

```bash
docker compose run --rm heuristic
```

The heuristic flaps when the ship drops below the gap centre. PPO should eventually learn something smarter — timing flaps ahead of the gap, adapting to shrinking gaps and increasing scroll speed.

### Behavioral cloning (recommended)

The best results so far come from **behavioral cloning** on a tuned rule-based expert, not from PPO alone:

```bash
docker compose run --rm bc-only
docker compose run --rm eval
```

Export the trained BC weights for the browser game:

```bash
docker compose run --rm dev python -m train.export_policy
docker compose run --rm dev python -m train.export_policy --model models/sparkler_bc --output models/sparkler_bc.json
```

Copy `models/sparkler_bc.json` into `sparkler-game-phaser/public/assets/models/` and toggle agent mode from the robot icon in the HUD.

## Project layout

### Docker services (`docker-compose.yml`)

| Service | Command | Purpose |
|---------|---------|---------|
| `test` | `pytest -v` | Run unit tests |
| `heuristic` | `train.eval_heuristic` | Score the rule-based expert |
| `bc-only` | `train.train_bc_only` | **Recommended** — train BC (+ DAgger) on full difficulty |
| `eval` | `train.eval_compare` | Compare heuristic, BC, and BC+PPO checkpoints |
| `train` | `train.train_ppo` | Pure PPO from scratch (sparse-reward baseline) |
| `bc-train` | `train.train_bc_ppo` | BC pre-train then PPO fine-tune (experimental) |
| `dev` | shell | Interactive container with repo mounted |
| `tensorboard` | TensorBoard on `logs/` | View PPO training curves |

Generated output (gitignored): `models/` for checkpoints, `logs/` for TensorBoard.

### `sparkler/` — game simulation

| File | Purpose |
|------|---------|
| `constants.py` | Physics, layout, rewards, and expert tuning values — kept in sync with `sparkler-game-phaser/src/game/tuning.ts` |
| `simulator.py` | Headless game loop: flap, scroll, collision, gap shrink, speed ramp |
| `env.py` | Gymnasium wrapper (`SparklerEnv`) exposing the 7-feature observation vector and discrete flap action |
| `curriculum.py` | Difficulty presets (`EASY` for training demos, `FULL` for the real game) |
| `heuristic.py` | Rule-based expert used to generate BC demonstrations (`expert_action`) |
| `demos.py` | Collect expert/DAgger transitions and run BC pre-training on an SB3 model |
| `__init__.py` | Public exports: `SparklerEnv`, `SparklerSimulator` |

### `train/` — scripts and shared helpers

| File | Purpose | Typical command |
|------|---------|-----------------|
| `train_bc_only.py` | **Main training path** — expert demos, DAgger, BC; writes `models/sparkler_bc` | `docker compose run --rm bc-only` |
| `eval_compare.py` | Side-by-side eval of heuristic vs saved BC / BC+PPO models | `docker compose run --rm eval` |
| `export_policy.py` | Export SB3 `MlpPolicy` weights to browser JSON (`models/sparkler_bc.json`) | `python -m train.export_policy` |
| `bc_utils.py` | Shared BC training loop, checkpoint save/load, post-train evaluation |
| `training_utils.py` | Vec-env factory, PPO training phases, policy evaluation helpers |
| `eval_heuristic.py` | Run the rule-based expert for N episodes and print scores | `docker compose run --rm heuristic` |
| `tune_heuristic.py` | Grid search over expert margin/lookahead (dev utility; results inform `constants.py`) | `python -m train.tune_heuristic` |
| `train_ppo.py` | Pure PPO with optional easy-mode curriculum — struggled on sparse rewards | `docker compose run --rm train` |
| `train_bc_ppo.py` | BC pre-train then PPO fine-tune — fine-tuning hurt BC performance in practice | `docker compose run --rm bc-train` |

### `tests/`

| File | Purpose |
|------|---------|
| `test_simulator.py` | Simulator physics, env stepping, heuristic episodes, demo collection |

### Root files

| File | Purpose |
|------|---------|
| `Dockerfile` | Python 3.12 image with project dependencies |
| `requirements.txt` | Pip dependencies (Gymnasium, SB3, PyTorch, etc.) |
| `.dockerignore` | Files excluded from the Docker build context |

## Notes

- Collision uses simplified rectangular gap bounds (v1); rounded polygon corners from the Phaser game are not modelled yet.
- Constants are kept in sync with `sparkler-game-phaser/src/game/tuning.ts` and the agent observation builder in Phaser.
- Trained models are written to `models/`; TensorBoard logs to `logs/`.
