"""Synthetic weather generation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import math
import random

__all__ = [
    "WeatherSample",
    "SyntheticWeatherProvider",
]


@dataclass(frozen=True)
class WeatherSample:
    """Representation of the external conditions for one hour."""

    outdoor_temp: float
    solar_irradiance: float
    wind_speed: float


class SyntheticWeatherProvider:
    """Generate pseudo realistic weather conditions without external data."""

    def __init__(self, *, base_temp: float = 8.0, diurnal_amplitude: float = 6.0,
                 seasonal_amplitude: float = 4.0, noise_std: float = 0.8,
                 solar_peak: float = 650.0, wind_mean: float = 2.5,
                 rng: random.Random | None = None) -> None:
        self.base_temp = float(base_temp)
        self.diurnal_amplitude = float(diurnal_amplitude)
        self.seasonal_amplitude = float(seasonal_amplitude)
        self.noise_std = float(noise_std)
        self.solar_peak = float(solar_peak)
        self.wind_mean = float(wind_mean)
        self.rng = rng or random.Random()

    def _daily_temperature(self, hour_of_day: float) -> float:
        phase = (hour_of_day - 7.0) / 24.0
        return math.sin(2.0 * math.pi * phase)

    def _seasonal_offset(self, absolute_hour: float) -> float:
        # Repeat every 90 days (~3 months) to add some variability to long
        # simulations while keeping the behaviour deterministic.
        phase = absolute_hour / (24.0 * 90.0)
        return math.sin(2.0 * math.pi * phase)

    def conditions_at(self, absolute_hour: float) -> WeatherSample:
        hour_of_day = absolute_hour % 24.0
        temp = (
            self.base_temp
            + self.diurnal_amplitude * self._daily_temperature(hour_of_day)
            + self.seasonal_amplitude * self._seasonal_offset(absolute_hour)
            + self.rng.gauss(0.0, self.noise_std)
        )

        solar = max(0.0, self.solar_peak * math.sin(math.pi * (hour_of_day - 6.0) / 12.0))
        wind = max(0.0, self.wind_mean + self.rng.gauss(0.0, 0.6))
        return WeatherSample(outdoor_temp=temp, solar_irradiance=solar, wind_speed=wind)

    def forecast(self, hours: int, *, start_hour: float = 0.0) -> List[WeatherSample]:
        return [self.conditions_at(start_hour + i) for i in range(int(hours))]

    def iter_forecast(self, hours: int, *, start_hour: float = 0.0) -> Iterator[WeatherSample]:
        for i in range(int(hours)):
            yield self.conditions_at(start_hour + i)
