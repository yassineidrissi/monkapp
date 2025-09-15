"""Core building thermal model used by monkapp.

The original repository mixed a lot of experimental code (training loops,
plotting scripts, etc.) inside a single gigantic module.  That made the real
physical model difficult to understand and almost impossible to unit test.  This
module focuses on the actual thermal dynamics of the building and exposes a
small, well documented API that can be reused by higher level components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple, Sequence

__all__ = [
    "AIR_DENSITY",
    "AIR_HEAT_CAPACITY",
    "BuildingParameters",
    "ThermalState",
    "BuildingThermalModel",
    "SimulationSample",
]

# -- Physical constants ------------------------------------------------------

#: Density of air at sea level (kg / m³).
AIR_DENSITY = 1.2041

#: Specific heat capacity of air (J / (kg·K)).
AIR_HEAT_CAPACITY = 1006.0


@dataclass(frozen=True)
class BuildingParameters:
    """Thermal parameters describing the building envelope.

    The default values roughly represent a well insulated 3 bedroom house.
    Values can easily be tweaked to model another building.  The parameters are
    expressed in SI units because they make the differential equations easier to
    follow:

    * ``thermal_mass_air`` and ``thermal_mass_envelope`` are thermal capacities
      expressed in Joule per Kelvin.  They represent how much energy must be
      injected to heat the indoor air / the envelope by 1°C.
    * ``conductance_envelope`` corresponds to the conductive heat transfer
      between the inner envelope node and the outdoor environment (W/K).
    * ``conductance_internal`` models the coupling between the indoor air and
      the envelope (W/K).
    * ``infiltration_air_changes`` indicates how many times the indoor air is
      renewed every hour because of infiltration (ACH).  The model converts it to
      an equivalent heat loss coefficient.
    * ``window_area`` (m²) and ``solar_heat_gain_coefficient`` (0-1) are used to
      compute solar gains from the incident irradiance.
    * ``hvac_efficiency`` is the ratio between electrical energy consumed by the
      HVAC system and the heat effectively injected indoors.
    * ``max_hvac_power`` (W) is only used by convenience helpers and does not
      constrain the differential model directly.
    """

    floor_area: float = 120.0
    ceiling_height: float = 2.6
    thermal_mass_air: float = 120.0 * 2.6 * AIR_DENSITY * AIR_HEAT_CAPACITY
    thermal_mass_envelope: float = 2.4e6
    conductance_envelope: float = 220.0
    conductance_internal: float = 180.0
    infiltration_air_changes: float = 0.4
    window_area: float = 24.0
    solar_heat_gain_coefficient: float = 0.62
    hvac_efficiency: float = 0.95
    max_hvac_power: float = 7500.0

    @property
    def infiltration_conductance(self) -> float:
        """Equivalent heat loss conductance for infiltration (W/K)."""

        if self.infiltration_air_changes <= 0:
            return 0.0

        volume = self.floor_area * self.ceiling_height
        mass_flow = self.infiltration_air_changes * volume * AIR_DENSITY / 3600.0
        return mass_flow * AIR_HEAT_CAPACITY


@dataclass
class ThermalState:
    """Temperatures of the two nodes used in the lumped RC model."""

    indoor: float = 20.0
    envelope: float = 18.0

    def as_array(self) -> List[float]:
        return [float(self.indoor), float(self.envelope)]


class SimulationSample(NamedTuple):
    """Small helper structure returned by :meth:`BuildingThermalModel.simulate`.

    Attributes:
        state: Thermal state after the step.
        energy_kwh: HVAC energy consumption for the step (kWh).
        solar_gain: Heat brought inside by solar radiation (W).
    """

    state: ThermalState
    energy_kwh: float
    solar_gain: float


class BuildingThermalModel:
    """Simple 2R2C building thermal model.

    The model uses two thermal nodes: indoor air and the building envelope.  A
    small number of parameters governs the heat exchanges between the nodes and
    with the outdoor environment.  Despite the apparent simplicity, the model is
    expressive enough to represent the impact of weather, occupancy and HVAC
    actions on the indoor comfort.
    """

    def __init__(self, params: BuildingParameters | None = None,
                 initial_state: ThermalState | None = None) -> None:
        self.params = params or BuildingParameters()
        base_state = initial_state or ThermalState()
        self._state = list(base_state.as_array())
        self._time_hours = 0.0

        # Pre-compute constants that never change during the simulation.
        self._inv_mass_air = 1.0 / self.params.thermal_mass_air
        self._inv_mass_env = 1.0 / self.params.thermal_mass_envelope

    # ------------------------------------------------------------------
    # Properties

    @property
    def state(self) -> ThermalState:
        return ThermalState(indoor=float(self._state[0]), envelope=float(self._state[1]))

    @property
    def time_hours(self) -> float:
        return self._time_hours

    # ------------------------------------------------------------------
    # Core dynamics

    def step(self, dt_hours: float, outdoor_temp: float, solar_irradiance: float,
             hvac_power: float = 0.0, internal_gains: float = 0.0) -> ThermalState:
        """Integrate the thermal model over ``dt_hours``.

        Args:
            dt_hours: Duration of the step in hours.
            outdoor_temp: Outdoor dry bulb temperature (°C).
            solar_irradiance: Global horizontal irradiance (W/m²).
            hvac_power: Thermal power injected by the HVAC system (W).  The raw
                power is automatically multiplied by :attr:`BuildingParameters.hvac_efficiency`.
            internal_gains: Thermal gains from occupants and appliances (W).

        Returns:
            The new :class:`ThermalState` after applying the step.
        """

        if dt_hours <= 0:
            raise ValueError("dt_hours must be positive")

        dt_seconds = dt_hours * 3600.0
        params = self.params

        indoor = float(self._state[0])
        envelope = float(self._state[1])

        hvac_heat = max(0.0, hvac_power) * params.hvac_efficiency
        solar_gain = max(0.0, solar_irradiance) * params.window_area * params.solar_heat_gain_coefficient

        # Heat flows in Watt (J/s)
        indoor_to_envelope = params.conductance_internal * (indoor - envelope)
        envelope_to_outdoor = params.conductance_envelope * (envelope - outdoor_temp)
        infiltration_loss = params.infiltration_conductance * (indoor - outdoor_temp)

        # Two coupled first order differential equations discretised with
        # explicit Euler integration which is perfectly adequate with time steps
        # ≥ 5 minutes.
        d_indoor = (hvac_heat + internal_gains + solar_gain - indoor_to_envelope - infiltration_loss)
        d_envelope = (indoor_to_envelope - envelope_to_outdoor)

        self._state[0] = indoor + d_indoor * self._inv_mass_air * dt_seconds
        self._state[1] = envelope + d_envelope * self._inv_mass_env * dt_seconds
        self._time_hours += dt_hours

        # Keep the values within realistic temperature bounds.  This prevents the
        # numerical integration from diverging when the model is misconfigured
        # without hiding issues: the clamp still allows the calling code to
        # detect unrealistic behaviour.
        self._state[0] = max(-50.0, min(60.0, self._state[0]))
        self._state[1] = max(-50.0, min(60.0, self._state[1]))
        return self.state

    # ------------------------------------------------------------------
    # Convenience helpers

    def reset(self, state: ThermalState | None = None) -> None:
        """Reset the simulation to ``state`` and set the internal clock to zero."""

        base_state = state or ThermalState()
        self._state = list(base_state.as_array())
        self._time_hours = 0.0

    def simulate(self, dt_hours: float, weather: Sequence[tuple[float, float]],
                 hvac_schedule: Sequence[float] | None = None,
                 internal_gains: Sequence[float] | None = None) -> List[SimulationSample]:
        """Run a simulation using precalculated schedules.

        Args:
            dt_hours: Time step (hours).
            weather: Sequence of ``(outdoor_temp, solar_irradiance)`` pairs.
            hvac_schedule: Optional sequence of HVAC powers (W).
            internal_gains: Optional sequence of internal gains (W).

        Returns:
            List of :class:`SimulationSample` objects collected during the
            simulation.
        """

        length = len(weather)
        hvac = hvac_schedule or [0.0] * length
        internal = internal_gains or [0.0] * length

        if len(hvac) != length or len(internal) != length:
            raise ValueError("Schedules must have the same length as weather data")

        samples: List[SimulationSample] = []
        for (temp_ext, irradiance), hvac_power, gains in zip(weather, hvac, internal):
            state = self.step(dt_hours, temp_ext, irradiance, hvac_power, gains)
            energy_kwh = max(0.0, hvac_power) * dt_hours / 1000.0
            solar_gain = max(0.0, irradiance) * self.params.window_area * self.params.solar_heat_gain_coefficient
            samples.append(SimulationSample(state=state, energy_kwh=energy_kwh, solar_gain=solar_gain))
        return samples

    # ------------------------------------------------------------------

    def equilibrium_temperature(self, outdoor_temp: float, gains: float = 0.0) -> float:
        """Approximate steady state indoor temperature for constant conditions."""

        params = self.params
        g_i = params.conductance_internal
        g_o = params.conductance_envelope
        g_inf = params.infiltration_conductance

        if g_i <= 0 and g_inf <= 0:
            return outdoor_temp

        # Solve the steady state of the 2R2C system analytically.  See the
        # derivation in the module documentation for the algebra.
        coupling = g_i * g_o / (g_i + g_o) if g_i > 0 and g_o > 0 else 0.0
        total_loss = coupling + g_inf
        if total_loss <= 0:
            return outdoor_temp

        return outdoor_temp + gains / total_loss
