from monkapp.controller import AdaptivePreheatingController, ControllerConfig
from monkapp.occupancy import OccupancyModel
from monkapp.building import BuildingThermalModel, ThermalState


def _occupancy_model():
    return OccupancyModel.from_periods(
        weekday_periods=[(7.0, 9.0, 1.0)],
        base_probability=0.0,
    )


def test_target_temperature_preheats():
    controller = AdaptivePreheatingController()
    occupancy = _occupancy_model()

    setback = controller.config.setback_temperature
    comfort = controller.config.comfort_temperature

    target_far = controller.compute_target(occupancy, day_index=0, hour=3.0)
    target_preheat = controller.compute_target(occupancy, day_index=0, hour=6.0)
    target_present = controller.compute_target(occupancy, day_index=0, hour=8.0)

    assert abs(target_far - setback) < 1e-6
    assert setback < target_preheat < comfort
    assert abs(target_present - comfort) < 1e-3


def test_hvac_power_is_clamped():
    controller = AdaptivePreheatingController(ControllerConfig(hvac_power_max=3000.0))
    occupancy = _occupancy_model()
    model = BuildingThermalModel(initial_state=ThermalState(indoor=15.0, envelope=15.0))

    target = controller.compute_target(occupancy, day_index=0, hour=7.5)
    power = controller.compute_hvac_power(model.state, target, dt_hours=0.5)
    assert 0 <= power <= 3000.0
