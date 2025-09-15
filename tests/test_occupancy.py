from monkapp.occupancy import OccupancyModel


def test_from_periods_builds_expected_profile():
    model = OccupancyModel.from_periods(
        weekday_periods=[(8.0, 18.0, 0.8)],
        weekend_periods=[(10.0, 22.0, 0.6)],
        base_probability=0.1,
    )

    assert model.probability(day_index=1, hour=9.0) > 0.6
    assert model.probability(day_index=6, hour=9.0) < 0.4


def test_expected_probability_increases_before_presence():
    model = OccupancyModel.from_periods(
        weekday_periods=[(7.0, 9.0, 1.0)],
        base_probability=0.0,
    )

    immediate = model.probability(0, 6.0)
    anticipated = model.expected_probability(0, 6.0, horizon_hours=2.0)
    assert anticipated > immediate
