"""Serverless entry point for Vercel to expose a lightweight simulation API."""
from __future__ import annotations

import json
import math
from http.server import BaseHTTPRequestHandler
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

from user_models import BuildingParameters, BuildingThermalModel

SECONDS_PER_HOUR = 3600
MAX_HOURS = 168  # one week of data to avoid runaway simulations on Vercel


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
    model = BuildingThermalModel(BuildingParameters())
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
        energy = model.get_energy_consumption(heating_power, SECONDS_PER_HOUR)
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

    response: Dict[str, object] = {
        "summary": {
            "hours_simulated": hours,
            "heating_power_w": round(float(heating_power), 2),
            "occupancy_gain_w": round(float(occupancy_gain), 2),
            "total_energy_kwh": round(float(total_energy), 4),
            "thermal_inertia_indicator": round(float(model.get_thermal_inertia_indicator()), 4),
            "final_temperatures_c": final_temperatures,
        },
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
