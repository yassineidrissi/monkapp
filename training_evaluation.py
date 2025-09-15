"""Command line entry point to run a quick thermal simulation."""
from __future__ import annotations

import argparse
import random
from statistics import fmean

from monkapp import (
    AdaptivePreheatingController,
    BuildingThermalModel,
    LSTMPPOAgent,
    OccupancyModel,
    PPOConfig,
    PPOEnvConfig,
    SimulationConfig,
    SyntheticWeatherProvider,
    ThermalComfortEnv,
    run_closed_loop_simulation,
)


def _default_occupancy() -> OccupancyModel:
    return OccupancyModel.from_periods(
        weekday_periods=[(6.0, 9.0, 0.9), (17.0, 23.0, 0.95)],
        weekend_periods=[(8.0, 23.0, 0.8)],
        base_probability=0.1,
        presence_value=0.9,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=72,
                        help="Number of hours to simulate (default: 72)")
    parser.add_argument("--step", type=float, default=0.25,
                        help="Simulation time step in hours (default: 0.25)")
    parser.add_argument("--occupants", type=float, default=1.0,
                        help="Expected number of occupants")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Seed used for the synthetic weather generator")
    parser.add_argument("--train-ppo", action="store_true",
                        help="Train an LSTM-PPO agent instead of running the PI controller")
    parser.add_argument("--ppo-iterations", type=int, default=8,
                        help="Number of PPO update iterations to run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    occupancy = _default_occupancy()
    weather = SyntheticWeatherProvider(rng=random.Random(args.seed))

    if args.train_ppo:
        env_model = BuildingThermalModel()
        env = ThermalComfortEnv(
            env_model,
            occupancy,
            weather,
            PPOEnvConfig(
                time_step_hours=args.step,
                episode_hours=float(args.hours),
                occupant_count=args.occupants,
            ),
        )
        ppo = LSTMPPOAgent(
            observation_size=env.observation_size,
            action_scale=env.action_scale,
            config=PPOConfig(rollout_length=32, train_epochs=2, learning_rate=0.01),
            seed=args.seed,
        )
        stats = ppo.train(env, iterations=max(1, args.ppo_iterations))
        mean_return = fmean(stats["episode_returns"]) if stats["episode_returns"] else 0.0
        last_loss = stats["loss"][-1] if stats["loss"] else float("nan")
        print("=== PPO training summary ===")
        print(f"Iterations executed: {len(stats['loss'])}")
        print(f"Last optimisation loss: {last_loss:.3f}")
        print(f"Mean episode return: {mean_return:.3f}")
        hvac_power, _ = ppo.act(env.reset())
        print(f"Initial deterministic action: {hvac_power:.1f} W")
        return

    controller = AdaptivePreheatingController()
    model = BuildingThermalModel()

    config = SimulationConfig(
        duration_hours=args.hours,
        time_step_hours=args.step,
        occupant_count=args.occupants,
    )

    result = run_closed_loop_simulation(model, controller, occupancy, weather, config)

    comfort_ratio = 1.0 - (result.comfort_violations_hours / max(1e-6, args.hours))

    print("=== Simulation summary ===")
    print(f"Duration: {args.hours} hours (step={args.step} h)")
    print(f"Energy consumption: {result.energy_consumption_kwh:.2f} kWh")
    print(f"Final indoor temperature: {result.indoor_temperatures[-1]:.1f}°C")
    print(f"Average indoor temperature: {fmean(result.indoor_temperatures):.1f}°C")
    print(f"Comfort ratio: {comfort_ratio:.1%}")


if __name__ == "__main__":
    main()
