"""Orchestrate the interactions between the building, controller and weather."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from statistics import fmean

from .building import BuildingThermalModel
from .controller import AdaptivePreheatingController
from .occupancy import OccupancyModel
from .weather import SyntheticWeatherProvider

__all__ = [
    "SimulationConfig",
    "SimulationResult",
    "run_closed_loop_simulation",
]


@dataclass
class SimulationConfig:
    duration_hours: int = 7 * 24
    time_step_hours: float = 0.25
    occupant_heat_gain: float = 120.0
    occupant_count: float = 1.0
    comfort_margin: float = 0.5
    start_hour: float = 0.0
    start_day_index: int = 0


@dataclass
class SimulationResult:
    times: List[float]
    indoor_temperatures: List[float]
    envelope_temperatures: List[float]
    outdoor_temperatures: List[float]
    target_temperatures: List[float]
    hvac_power: List[float]
    occupancy_probability: List[float]
    energy_consumption_kwh: float
    comfort_violations_hours: float

    def mean_indoor_temperature(self) -> float:
        return fmean(self.indoor_temperatures)


def run_closed_loop_simulation(model: BuildingThermalModel,
                               controller: AdaptivePreheatingController,
                               occupancy: OccupancyModel,
                               weather: SyntheticWeatherProvider,
                               config: SimulationConfig | None = None) -> SimulationResult:
    cfg = config or SimulationConfig()
    controller.reset()
    model.reset()

    steps = int(cfg.duration_hours / cfg.time_step_hours)
    times: List[float] = [i * cfg.time_step_hours for i in range(steps)]

    indoor: List[float] = [0.0] * steps
    envelope: List[float] = [0.0] * steps
    outdoor: List[float] = [0.0] * steps
    targets: List[float] = [0.0] * steps
    hvac: List[float] = [0.0] * steps
    occupancy_prob: List[float] = [0.0] * steps

    energy_kwh = 0.0
    comfort_violation = 0.0

    for i in range(steps):
        absolute_hour = cfg.start_hour + i * cfg.time_step_hours
        day_index = cfg.start_day_index + int((cfg.start_hour + i * cfg.time_step_hours) // 24)
        hour_of_day = absolute_hour % 24

        occ_prob = occupancy.probability(day_index, hour_of_day)
        target = controller.compute_target(occupancy, day_index, hour_of_day)
        hvac_power = controller.compute_hvac_power(model.state, target, cfg.time_step_hours)

        weather_sample = weather.conditions_at(absolute_hour)
        internal_gains = occ_prob * cfg.occupant_heat_gain * cfg.occupant_count
        state = model.step(
            cfg.time_step_hours,
            weather_sample.outdoor_temp,
            weather_sample.solar_irradiance,
            hvac_power,
            internal_gains,
        )

        indoor[i] = state.indoor
        envelope[i] = state.envelope
        outdoor[i] = weather_sample.outdoor_temp
        targets[i] = target
        hvac[i] = hvac_power
        occupancy_prob[i] = occ_prob

        energy_kwh += hvac_power * cfg.time_step_hours / 1000.0

        if state.indoor + cfg.comfort_margin < controller.config.comfort_temperature:
            comfort_violation += cfg.time_step_hours * occ_prob

    return SimulationResult(
        times=times,
        indoor_temperatures=indoor,
        envelope_temperatures=envelope,
        outdoor_temperatures=outdoor,
        target_temperatures=targets,
        hvac_power=hvac,
        occupancy_probability=occupancy_prob,
        energy_consumption_kwh=energy_kwh,
        comfort_violations_hours=comfort_violation,
    )
