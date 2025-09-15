from monkapp import (
    AdaptivePreheatingController,
    BuildingThermalModel,
    OccupancyModel,
    SimulationConfig,
    SyntheticWeatherProvider,
    run_closed_loop_simulation,
)


def test_run_closed_loop_simulation():
    occupancy = OccupancyModel.from_periods(
        weekday_periods=[(6.0, 9.0, 0.9), (17.0, 23.0, 0.9)],
        weekend_periods=[(8.0, 23.0, 0.7)],
        base_probability=0.1,
    )
    controller = AdaptivePreheatingController()
    building = BuildingThermalModel()
    weather = SyntheticWeatherProvider()
    config = SimulationConfig(duration_hours=48, time_step_hours=0.5)

    result = run_closed_loop_simulation(building, controller, occupancy, weather, config)

    expected_steps = int(config.duration_hours / config.time_step_hours)
    assert len(result.indoor_temperatures) == expected_steps
    assert result.energy_consumption_kwh >= 0
    assert 0 <= result.comfort_violations_hours <= config.duration_hours
    mean_temperature = sum(result.indoor_temperatures) / len(result.indoor_temperatures)
    assert mean_temperature > 15.0
