"""Training curriculum settings."""

from __future__ import annotations

from dataclasses import dataclass

from sparkler.constants import INITIAL_GAP_PERCENT


@dataclass(frozen=True)
class CurriculumSettings:
    fixed_gap_percent: float | None = None
    enable_gap_shrink: bool = True
    enable_speed_ramp: bool = True
    centered_gaps: bool = False


EASY = CurriculumSettings(
    fixed_gap_percent=float(INITIAL_GAP_PERCENT),
    enable_gap_shrink=False,
    enable_speed_ramp=False,
    centered_gaps=True,
)

FULL = CurriculumSettings()
