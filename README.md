# Sparkler RL

Reinforcement learning spin-off of [sparkler-game-phaser](https://github.com/taylorjg/sparkler-game-phaser). Trains an agent to play the sparkler obstacle game in a headless Python simulator, then (later) deploy the policy back into Phaser.

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

# Train PPO (CPU; takes a while)
docker compose run --rm train

# Interactive shell
docker compose run --rm dev bash

# Export BC policy to JSON for the Phaser game
docker compose run --rm dev python -m train.export_policy
```

The export writes `models/sparkler_bc.json`. Copy it to `sparkler-game-phaser/public/assets/models/` and open the Phaser game with `?agent=1`.

## How training works

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
| Each frame survived | `+0.01` |
| Obstacle cleared | `+1.0` |
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

Copy `models/sparkler_bc.json` into `sparkler-game-phaser/public/assets/models/` and play with `?agent=1`.

## Project layout

```
sparkler/
  constants.py    # Tuning values ported from game-scene.ts
  simulator.py    # Headless game loop
  env.py          # Gymnasium wrapper
train/
  eval_heuristic.py
  train_ppo.py
tests/
```

## Notes

- Collision uses simplified rectangular gap bounds (v1); rounded polygon corners from the Phaser game are not modelled yet.
- Constants are kept in sync with `sparkler-game-phaser/src/scenes/game-scene.ts`.
- Trained models are written to `models/`; TensorBoard logs to `logs/`.
