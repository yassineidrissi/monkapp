import math
import random

from monkapp import (
    BuildingThermalModel,
    LSTMPPOAgent,
    OccupancyModel,
    PPOConfig,
    PPOEnvConfig,
    SyntheticWeatherProvider,
    ThermalComfortEnv,
)


def test_lstm_ppo_agent_performs_updates():
    occupancy = OccupancyModel.from_periods(
        weekday_periods=[(6.0, 9.0, 0.85), (17.0, 23.0, 0.9)],
        weekend_periods=[(8.0, 22.0, 0.75)],
        base_probability=0.1,
    )
    weather = SyntheticWeatherProvider(rng=random.Random(42))
    model = BuildingThermalModel()
    env = ThermalComfortEnv(
        model,
        occupancy,
        weather,
        PPOEnvConfig(time_step_hours=0.5, episode_hours=12.0, occupant_count=1.0),
    )

    config = PPOConfig(
        learning_rate=0.01,
        rollout_length=12,
        train_epochs=2,
        hidden_size=16,
    )
    agent = LSTMPPOAgent(
        observation_size=env.observation_size,
        action_scale=env.action_scale,
        config=config,
        seed=0,
    )

    def flatten(matrix):
        return [value for row in matrix for value in row]

    initial_weights = flatten(agent._network.actor_weight)
    stats = agent.train(env, iterations=4)

    assert stats["loss"], "the agent should record optimisation losses"
    assert stats["episode_returns"], "the agent should record episodic returns"
    assert all(math.isfinite(value) for value in stats["loss"])

    updated_weights = flatten(agent._network.actor_weight)
    assert any(abs(a - b) > 1e-6 for a, b in zip(initial_weights, updated_weights)), "policy weights must change after training"

    obs = env.reset()
    action, _ = agent.act(obs)
    assert 0.0 <= action <= env.action_scale
