"""Serverless entry point for Vercel to expose a lightweight simulation API."""
from __future__ import annotations

import json
import math
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

# Ensure the repository root is importable even when the file is executed directly
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:  # pragma: no cover - exercised indirectly when dependencies are available
    from user_models import BuildingParameters, BuildingThermalModel
except Exception as exc:
    BuildingParameters = None  # type: ignore[assignment]
    BuildingThermalModel = None  # type: ignore[assignment]
    MODEL_IMPORT_ERROR = exc
else:
    MODEL_IMPORT_ERROR = None

SECONDS_PER_HOUR = 3600
MAX_HOURS = 168  # one week of data to avoid runaway simulations on Vercel


class _FallbackBuildingParameters:
    """Minimal thermal parameters used when SciPy/Numpy are unavailable."""

    C_building = 5.0e6
    C_wall = 2.5e6
    C_air = 1.5e6
    U_wall_ext = 280.0
    U_wall_int = 420.0
    U_window = 120.0
    eta_window = 0.8
    S_window = 20.0
    building_orientation = 0.0
    surface_total = 150.0
    height = 2.7
    air_changes_per_hour = 1.0


def _sun_angle(seconds: float) -> float:
    """Approximate solar angle for a 24h cycle (degrees)."""

    hours = (seconds / SECONDS_PER_HOUR) % 24.0
    if 6 <= hours <= 18:
        return -90.0 + 15.0 * (hours - 6.0)
    return -90.0


class _FallbackBuildingModel:
    """Simple RC thermal model used when the detailed model cannot be imported."""

    def __init__(self) -> None:
        self.params = _FallbackBuildingParameters()
        self.state = {
            "T_int": 20.0,
            "T_wall": 18.0,
            "T_air": 19.0,
        }
        self.time = 0.0

    def update(
        self,
        dt: float,
        T_ext: float,
        I_solar: float,
        Q_heating: float,
        Q_occupancy: float,
    ) -> Dict[str, float]:
        params = self.params
        T_int = self.state["T_int"]
        T_wall = self.state["T_wall"]
        T_air = self.state["T_air"]

        steps = max(1, int(dt // 300))
        sub_dt = dt / steps

        rho_air = 1.2
        Cp_air = 1000.0
        V_building = params.surface_total * params.height
        h_conv_int = 8.0
        A_internal = params.surface_total * 3.0

        for _ in range(steps):
            solar_angle = _sun_angle(self.time)
            solar_factor = max(
                0.0,
                math.cos(math.radians(solar_angle + params.building_orientation)),
            )
            Q_solar = params.eta_window * params.S_window * I_solar * solar_factor

            Q_wall_ext = params.U_wall_ext * (T_wall - T_ext)
            Q_wall_int = params.U_wall_int * (T_int - T_wall)
            Q_window = params.U_window * (T_int - T_ext)

            Q_infiltration = (
                params.air_changes_per_hour
                * V_building
                * rho_air
                * Cp_air
                * (T_int - T_ext)
            ) / 3600.0

            Q_conv_internal = h_conv_int * A_internal * (T_air - T_int)

            effective_heating = Q_heating * 0.6
            effective_occupancy = Q_occupancy * 0.5

            dT_int_dt = (
                effective_heating
                + Q_solar
                + effective_occupancy
                + Q_conv_internal
                - Q_wall_int
                - Q_window
                - Q_infiltration
            ) / params.C_building
            dT_wall_dt = (Q_wall_ext - Q_wall_int) / params.C_wall
            dT_air_dt = -Q_conv_internal / params.C_air

            T_int = max(10.0, min(30.0, T_int + dT_int_dt * sub_dt))
            T_wall = max(5.0, min(30.0, T_wall + dT_wall_dt * sub_dt))
            T_air = max(10.0, min(30.0, T_air + dT_air_dt * sub_dt))

            self.time += sub_dt

        self.state.update({"T_int": T_int, "T_wall": T_wall, "T_air": T_air})
        return {
            "T_int": T_int,
            "T_wall": T_wall,
            "T_air": T_air,
            "time": self.time,
        }

    @staticmethod
    def get_energy_consumption(Q_heating: float, dt: float) -> float:
        return (Q_heating * dt) / 3.6e6

    def get_thermal_inertia_indicator(self) -> float:
        return abs(self.state["T_wall"] - self.state["T_int"])


def _create_model() -> Tuple[Any, str]:
    """Return the best available building model and its description."""

    if BuildingThermalModel is not None and BuildingParameters is not None:
        return BuildingThermalModel(BuildingParameters()), "detailed"
    return _FallbackBuildingModel(), "fallback"


def _exterior_temperature(hour: int) -> float:
    """Return a smooth exterior temperature profile (°C) for the simulation."""
    daily_cycle = math.sin(2 * math.pi * (hour % 24) / 24.0)
    return 6.0 + 7.0 * daily_cycle


def _solar_irradiance(hour: int) -> float:
    """Synthetic solar irradiance curve (W/m²) clamped to daylight hours."""
    daylight_factor = math.sin(math.pi * (hour % 24) / 12.0)
    return max(0.0, 450.0 * daylight_factor)


def run_simulation(hours: int, heating_power: float, occupancy_gain: float) -> Dict[str, object]:
    """Execute the thermal model and return structured results."""
    model, model_source = _create_model()
    timeline: List[Dict[str, float]] = []
    total_energy = 0.0

    for hour in range(hours):
        state = model.update(
            dt=SECONDS_PER_HOUR,
            T_ext=_exterior_temperature(hour),
            I_solar=_solar_irradiance(hour),
            Q_heating=heating_power,
            Q_occupancy=occupancy_gain,
        )
        energy = model.get_energy_consumption(
            heating_power, SECONDS_PER_HOUR
        )

        total_energy += energy

        timeline.append(
            {
                "hour": hour + 1,
                "indoor_c": round(float(state["T_int"]), 2),
                "wall_c": round(float(state["T_wall"]), 2),
                "air_c": round(float(state["T_air"]), 2),
                "energy_kwh": round(float(energy), 4),
            }
        )

    final_temperatures: Dict[str, float] = {}
    if timeline:
        last_state = timeline[-1]
        final_temperatures = {
            "indoor_c": last_state["indoor_c"],
            "wall_c": last_state["wall_c"],
            "air_c": last_state["air_c"],
        }

    summary: Dict[str, object] = {
        "hours_simulated": hours,
        "heating_power_w": round(float(heating_power), 2),
        "occupancy_gain_w": round(float(occupancy_gain), 2),
        "total_energy_kwh": round(float(total_energy), 4),
        "thermal_inertia_indicator": round(
            float(model.get_thermal_inertia_indicator()), 4
        ),
        "final_temperatures_c": final_temperatures,
        "model_variant": model_source,
    }
    if model_source == "fallback" and MODEL_IMPORT_ERROR is not None:
        summary["model_warning"] = str(MODEL_IMPORT_ERROR)

    response: Dict[str, object] = {
        "summary": summary,
        "timeline": timeline,
    }
    return response


class handler(BaseHTTPRequestHandler):
    """Vercel-compatible request handler exposing the simulation endpoint."""

    def _set_common_headers(self) -> None:
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")

    def do_OPTIONS(self) -> None:  # noqa: N802 (method name required by BaseHTTPRequestHandler)
        self.send_response(204)
        self._set_common_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 (method name required by BaseHTTPRequestHandler)
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        try:
            hours = max(1, min(MAX_HOURS, int(params.get("hours", ["24"])[0])))
            heating_power = max(0.0, float(params.get("heating_power", ["2000"])[0]))
            occupancy_gain = max(0.0, float(params.get("occupancy_gain", ["150"])[0]))
        except ValueError:
            self.send_response(400)
            self._set_common_headers()
            self.end_headers()
            payload = json.dumps({"error": "Invalid query parameters"})
            self.wfile.write(payload.encode("utf-8"))
            return

        result = run_simulation(hours, heating_power, occupancy_gain)
        payload = json.dumps(result)

        self.send_response(200)
        self._set_common_headers()
        self.end_headers()
        self.wfile.write(payload.encode("utf-8"))

    def log_message(self, format: str, *args: Tuple[object, ...]) -> None:  # noqa: A003
        """Silence default request logging to keep serverless logs clean."""
        return

if __name__ == "__main__":
    sample = run_simulation(6, 2000, 150)
    print(json.dumps(sample["summary"], indent=2))