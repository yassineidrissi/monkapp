# Monkapp Thermal Simulation Toolkit

This repository now contains a compact and well documented building thermal
simulation stack.  The previous dump of experimentation scripts has been cleaned
up and replaced by a reusable Python package with unit-tested components.

## Features

- **Deterministic building model** (`monkapp.building.BuildingThermalModel`) with
a 2R2C representation, solar gains and infiltration losses.
- **Probabilistic occupancy model** (`monkapp.occupancy.OccupancyModel`) able to
  generate hourly presence probabilities for weekdays and weekends.
- **Adaptive heating controller** (`monkapp.controller.AdaptivePreheatingController`)
  implementing a PI loop and pre-heating logic driven by the occupancy
  probabilities.
- **Reinforcement learning tools** (`monkapp.rl`) featuring an LSTM based PPO
  agent and a lightweight thermal control environment.
- **Synthetic weather generator** (`monkapp.weather.SyntheticWeatherProvider`)
  used to run realistic simulations offline.
- **Convenience orchestration utilities** (`monkapp.simulation`) to simulate the
  closed loop behaviour and compute energy/comfort metrics.
- **CLI demo** (`training_evaluation.py`) that runs a short simulation and prints
  a human readable summary.

## Quick start

Create a virtual environment, install the dependencies and run the demo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pytest
python training_evaluation.py --hours 168 --step 0.5
```

The script prints energy consumption, final indoor temperature and the ratio of
comfortable hours.

To try the LSTM-PPO agent that learns a control policy directly, run:

```bash
python training_evaluation.py --train-ppo --hours 72 --step 0.5 --ppo-iterations 6
```

This launches a short on-policy optimisation and reports the mean episodic
return together with the final loss value.

## Library usage

```python
from monkapp import (
    AdaptivePreheatingController,
    BuildingThermalModel,
    OccupancyModel,
    SimulationConfig,
    SyntheticWeatherProvider,
    run_closed_loop_simulation,
)

occupancy = OccupancyModel.from_periods(
    weekday_periods=[(6.0, 9.0, 0.9), (17.0, 23.0, 0.95)],
    weekend_periods=[(8.0, 23.0, 0.8)],
)
controller = AdaptivePreheatingController()
building = BuildingThermalModel()
weather = SyntheticWeatherProvider()
config = SimulationConfig(duration_hours=72, time_step_hours=0.25)

result = run_closed_loop_simulation(building, controller, occupancy, weather, config)
print(result.energy_consumption_kwh)
```

## Tests

Pytest test cases are located in the `tests/` directory and can be executed
with:

```bash
pytest
```

Running the test suite validates the thermal model, controller logic and the
closed-loop simulation interface.
