"""Rule-based baseline agent."""

from __future__ import annotations

from sparkler import constants as C
from sparkler.simulator import SparklerSimulator


def heuristic_wants_flap(sim: SparklerSimulator, margin: float = C.EXPERT_MARGIN) -> bool:
    """True when the ship has dropped below the gap centre."""
    return sim.ship_y > sim.obstacle.gap_center + margin


def predict_ship_y(sim: SparklerSimulator, frames: int = C.EXPERT_LOOKAHEAD) -> float:
    """Predict ship height with constant gravity and no thrust."""
    dt = C.REFERENCE_FRAME_MS / 1000.0
    y = sim.ship_y
    velocity = sim.velocity_y
    for _ in range(frames):
        velocity += C.GRAVITY_Y * dt
        y += velocity * dt
    return y


def expert_action(
    sim: SparklerSimulator,
    margin: float = C.EXPERT_MARGIN,
    lookahead: int = C.EXPERT_LOOKAHEAD,
) -> bool:
    """Flap when a burst is needed to stay near the gap centre."""
    if sim.thrust_active:
        return False

    gap_center = sim.obstacle.gap_center
    predicted_y = predict_ship_y(sim, lookahead)
    below_now = sim.ship_y > gap_center + margin
    falling_toward_gap = sim.velocity_y > -100
    will_drop_below = predicted_y > gap_center + margin

    return below_now and falling_toward_gap and will_drop_below


def heuristic_action(
    sim: SparklerSimulator,
    margin: float = C.EXPERT_MARGIN,
    lookahead: int = C.EXPERT_LOOKAHEAD,
) -> bool:
    """Used for live heuristic play."""
    return expert_action(sim, margin, lookahead)
