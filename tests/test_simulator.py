from sparkler.constants import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    INITIAL_GAP_PERCENT,
    MIN_GAP_PERCENT,
    REFERENCE_FRAME_MS,
)
from sparkler.simulator import SparklerSimulator


def test_obstacle_width_matches_game_formula() -> None:
    sim = SparklerSimulator(width=800, height=600, seed=1)
    assert sim._get_obstacle_width() == 40


def test_speed_ramps_over_time() -> None:
    sim = SparklerSimulator(width=800, height=600, seed=1)
    start_speed = sim._get_speed()
    sim.running_elapsed_ms = 60_000
    end_speed = sim._get_speed()
    assert end_speed > start_speed


def test_gap_shrinks_after_obstacle_cleared() -> None:
    sim = SparklerSimulator(seed=1)
    initial_gap = sim.gap_percent
    sim.obstacle_pair_cleared = True
    sim.scroll_x = sim.obstacle.right + 1
    sim.step(flap=False)
    assert sim.gap_percent == initial_gap - 2


def test_easy_curriculum_keeps_gap_and_speed_fixed() -> None:
    from sparkler.curriculum import EASY

    sim = SparklerSimulator(seed=1, curriculum=EASY)
    start_speed = sim._get_speed()
    sim.running_elapsed_ms = 120_000
    assert sim._get_speed() == start_speed

    sim.obstacle_pair_cleared = True
    sim.scroll_x = sim.obstacle.right + 1
    sim.step(flap=False)
    assert sim.gap_percent == INITIAL_GAP_PERCENT


def test_reset_restores_initial_state() -> None:
    sim = SparklerSimulator(seed=1)
    for _ in range(100):
        result = sim.step(flap=True)
        if result.terminated:
            break
    sim.reset()
    assert sim.score == 0
    assert sim.gap_percent == INITIAL_GAP_PERCENT
    assert sim.scroll_x == 0.0


def test_env_reset_and_step() -> None:
    from sparkler.env import SparklerEnv

    env = SparklerEnv(seed=7)
    obs, info = env.reset()
    assert obs.shape == (7,)
    assert info["score"] == 0

    obs, reward, terminated, truncated, info = env.step(1)
    assert obs.shape == (7,)
    assert isinstance(reward, float)


def test_heuristic_beats_random() -> None:
    from train.eval_heuristic import run_episode

    heuristic_scores = [run_episode(seed=i) for i in range(10)]
    random_scores = []
    for seed in range(10):
        sim = SparklerSimulator(seed=seed)
        for _ in range(20_000):
            result = sim.step(flap=sim.rng.random() < 0.05)
            if result.terminated:
                break
        random_scores.append(sim.score)

    assert sum(heuristic_scores) / len(heuristic_scores) > sum(random_scores) / len(
        random_scores
    )


def test_collect_demonstrations() -> None:
    from sparkler.curriculum import EASY
    from sparkler.demos import collect_demonstrations

    observations, actions = collect_demonstrations(3, curriculum=EASY, seed=99)
    assert observations.shape[1] == 7
    assert len(observations) == len(actions) > 0
    assert set(actions.tolist()).issubset({0, 1})
