"""Heating controller implementations."""
from __future__ import annotations

from dataclasses import dataclass

from .building import BuildingParameters, ThermalState
from .occupancy import OccupancyModel

__all__ = [
    "ControllerConfig",
    "AdaptivePreheatingController",
]


@dataclass
class ControllerConfig:
    comfort_temperature: float = 21.0
    setback_temperature: float = 17.0
    preheat_lead_hours: float = 1.5
    anticipation_horizon_hours: float = 2.0
    kp: float = 2200.0
    ki: float = 120.0
    integral_limit: float = 20000.0
    hvac_power_max: float | None = None


class AdaptivePreheatingController:
    """Simple PI controller with adaptive temperature set-point."""

    def __init__(self, config: ControllerConfig | None = None,
                 building: BuildingParameters | None = None) -> None:
        self.config = config or ControllerConfig()
        self.building = building or BuildingParameters()
        self._integral_error = 0.0

    def reset(self) -> None:
        self._integral_error = 0.0

    def compute_target(self, occupancy: OccupancyModel, day_index: int,
                        hour: float) -> float:
        cfg = self.config
        current = occupancy.probability(day_index, hour)
        anticipation = occupancy.expected_probability(
            day_index, hour,
            horizon_hours=cfg.preheat_lead_hours,
            step_hours=0.25,
        )
        preheat_factor = max(current, anticipation)
        return cfg.setback_temperature + preheat_factor * (
            cfg.comfort_temperature - cfg.setback_temperature
        )

    def compute_hvac_power(self, state: ThermalState, target_temp: float,
                            dt_hours: float) -> float:
        error = target_temp - state.indoor
        self._integral_error += error * dt_hours
        lower = -self.config.integral_limit
        upper = self.config.integral_limit
        if self._integral_error < lower:
            self._integral_error = lower
        elif self._integral_error > upper:
            self._integral_error = upper

        command = self.config.kp * error + self.config.ki * self._integral_error
        if command <= 0:
            self._integral_error = 0.0
            command = 0.0

        max_power = self.config.hvac_power_max or self.building.max_hvac_power
        if command < 0.0:
            return 0.0
        if command > max_power:
            return float(max_power)
        return float(command)
