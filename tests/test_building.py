from monkapp.building import BuildingParameters, BuildingThermalModel, ThermalState


def test_building_cools_towards_outdoor_temperature():
    params = BuildingParameters()
    model = BuildingThermalModel(params, ThermalState(indoor=22.0, envelope=20.0))

    for _ in range(96):
        model.step(0.25, outdoor_temp=0.0, solar_irradiance=0.0, hvac_power=0.0)

    state = model.state
    assert state.indoor < 22.0
    assert state.indoor > -20.0


def test_hvac_power_raises_temperature():
    params = BuildingParameters()
    model = BuildingThermalModel(params, ThermalState(indoor=18.0, envelope=17.0))

    for _ in range(24):
        model.step(0.25, outdoor_temp=5.0, solar_irradiance=0.0, hvac_power=5000.0)

    assert model.state.indoor > 19.5


def test_solar_gain_effect():
    params = BuildingParameters()
    model = BuildingThermalModel(params, ThermalState(indoor=19.0, envelope=18.5))

    model.step(0.5, outdoor_temp=5.0, solar_irradiance=800.0, hvac_power=0.0)
    assert model.state.indoor > 19.0
