"""Occupancy models used to feed the thermal simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import math
import random
from statistics import fmean

__all__ = [
    "OccupancyModel",
    "build_profile_from_periods",
]


def _interpolate_hourly(profile: Sequence[float], hour: float) -> float:
    hour = float(hour) % 24.0
    index = int(math.floor(hour))
    next_index = (index + 1) % 24
    fraction = hour - index
    return (1.0 - fraction) * profile[index] + fraction * profile[next_index]


def build_profile_from_periods(periods: Iterable[tuple[float, float, float | None]],
                               *, base_probability: float = 0.05,
                               presence_value: float = 0.9) -> List[float]:
    """Create an hourly profile from presence periods.

    Args:
        periods: Iterable of ``(start_hour, end_hour, probability)`` triples.  The
            probability is optional; when omitted ``presence_value`` is used.  If
            ``start_hour`` is greater than ``end_hour`` the interval is assumed to
            wrap to the next day.
        base_probability: Value assigned to hours outside any period.
        presence_value: Default probability used when the tuple only provides the
            start and end hour.
    """

    profile = [float(base_probability)] * 24
    periods = list(periods)

    for hour in range(24):
        hour_center = hour + 0.5
        probability = base_probability
        for start, end, *maybe_prob in periods:
            value = maybe_prob[0] if maybe_prob else presence_value
            if start <= end:
                inside = start <= hour_center < end
            else:  # wrap to the next day
                inside = hour_center >= start or hour_center < end
            if inside:
                probability = max(probability, float(value))
        profile[hour] = max(0.0, min(1.0, probability))
    return profile


@dataclass
class OccupancyModel:
    """Simple probabilistic occupancy model.

    The model stores two hourly profiles: one for weekdays, another for weekends.
    The :meth:`probability` method interpolates the profile to support sub-hourly
    time steps.  The model is intentionally lightweight because it is primarily
    used to provide soft constraints to the controller.
    """

    weekday_profile: Sequence[float]
    weekend_profile: Sequence[float]

    @classmethod
    def from_periods(cls, weekday_periods: Iterable[tuple[float, float, float | None]],
                     weekend_periods: Iterable[tuple[float, float, float | None]] | None = None,
                     *, base_probability: float = 0.05,
                     presence_value: float = 0.9) -> "OccupancyModel":
        weekday = build_profile_from_periods(
            weekday_periods, base_probability=base_probability, presence_value=presence_value
        )
        weekend = build_profile_from_periods(
            weekend_periods or weekday_periods,
            base_probability=base_probability,
            presence_value=presence_value,
        )
        return cls(weekday_profile=weekday, weekend_profile=weekend)

    # ------------------------------------------------------------------

    def _profile_for_day(self, day_index: int) -> Sequence[float]:
        day_index = int(day_index) % 7
        if day_index in (5, 6):
            return self.weekend_profile
        return self.weekday_profile

    def probability(self, day_index: int, hour: float) -> float:
        return float(_interpolate_hourly(self._profile_for_day(day_index), hour))

    def expected_probability(self, day_index: int, hour: float,
                             horizon_hours: float = 1.0,
                             step_hours: float = 0.25) -> float:
        if horizon_hours <= 0:
            return self.probability(day_index, hour)

        samples: List[float] = []
        steps = max(1, int(math.ceil(horizon_hours / step_hours)))
        for i in range(steps + 1):
            offset = min(horizon_hours, i * step_hours)
            samples.append(self.probability(day_index, hour + offset))
        return float(fmean(samples))

    def draw_presence(self, day_index: int, hour: float,
                       rng: random.Random | None = None) -> bool:
        rng = rng or random.Random()
        return bool(rng.random() < self.probability(day_index, hour))

    def heat_gain(self, day_index: int, hour: float,
                  sensible_gain: float = 100.0) -> float:
        """Return the expected sensible heat gain produced by occupants (W)."""

        return sensible_gain * self.probability(day_index, hour)
