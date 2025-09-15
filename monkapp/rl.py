"""Reinforcement learning utilities for monkapp."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import math
import random

from .building import BuildingThermalModel
from .occupancy import OccupancyModel
from .weather import SyntheticWeatherProvider

__all__ = [
    "PPOConfig",
    "PPOEnvConfig",
    "ThermalComfortEnv",
    "LSTMPPOAgent",
]


# ---------------------------------------------------------------------------
# Small linear algebra helpers implemented with plain Python lists.


def _zeros(length: int) -> List[float]:
    return [0.0 for _ in range(length)]


def _zeros_matrix(rows: int, cols: int) -> List[List[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _clone_vector(vector: Sequence[float]) -> List[float]:
    return [float(v) for v in vector]


def _clone_matrix(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(v) for v in row] for row in matrix]


def _vec_add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def _vec_sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def _vec_mul(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x * y for x, y in zip(a, b)]


def _vec_scale(a: Sequence[float], scalar: float) -> List[float]:
    return [x * scalar for x in a]


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    return [sum(row[i] * vector[i] for i in range(len(vector))) for row in matrix]


def _outer(a: Sequence[float], b: Sequence[float]) -> List[List[float]]:
    return [[x * y for y in b] for x in a]


def _vec_sigmoid(a: Sequence[float]) -> List[float]:
    return [1.0 / (1.0 + math.exp(-x)) for x in a]


def _vec_tanh(a: Sequence[float]) -> List[float]:
    return [math.tanh(x) for x in a]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sum_squares(values: Sequence[float]) -> float:
    return sum(v * v for v in values)


def _matrix_sum_squares(matrix: Sequence[Sequence[float]]) -> float:
    return sum(_sum_squares(row) for row in matrix)


def _add_matrix(dest: List[List[float]], increment: Sequence[Sequence[float]]) -> None:
    for row_dest, row_inc in zip(dest, increment):
        for i in range(len(row_dest)):
            row_dest[i] += row_inc[i]


def _add_vector(dest: List[float], increment: Sequence[float]) -> None:
    for i in range(len(dest)):
        dest[i] += increment[i]


# ---------------------------------------------------------------------------
# Environment definitions.


@dataclass
class PPOEnvConfig:
    """Configuration of the thermal control environment used for PPO training."""

    time_step_hours: float = 0.25
    episode_hours: float = 24.0
    occupant_heat_gain: float = 120.0
    occupant_count: float = 1.0
    comfort_temperature: float = 21.0
    comfort_weight: float = 1.0
    energy_weight: float = 2e-4
    start_hour: float = 0.0
    start_day_index: int = 0
    hvac_power_limit: float | None = None


class ThermalComfortEnv:
    """Simple environment coupling the thermal model, occupancy and weather."""

    def __init__(
        self,
        model: BuildingThermalModel,
        occupancy: OccupancyModel,
        weather: SyntheticWeatherProvider,
        config: PPOEnvConfig | None = None,
    ) -> None:
        self.model = model
        self.occupancy = occupancy
        self.weather = weather
        self.config = config or PPOEnvConfig()

        self._max_power = self.config.hvac_power_limit or self.model.params.max_hvac_power
        self._max_steps = int(self.config.episode_hours / self.config.time_step_hours)
        if self._max_steps <= 0:
            raise ValueError("episode_hours must be positive")

        self._initial_rng_state = self.weather.rng.getstate()
        self._step_count = 0
        self._absolute_hour = self.config.start_hour
        self._current_occ = 0.0
        self._current_weather = self.weather.conditions_at(self._absolute_hour)
        self._prev_hvac = 0.0

    @property
    def observation_size(self) -> int:
        return 7

    @property
    def action_scale(self) -> float:
        return float(self._max_power)

    def reset(self) -> List[float]:
        self.model.reset()
        self._step_count = 0
        self._absolute_hour = self.config.start_hour
        self.weather.rng.setstate(self._initial_rng_state)
        self._current_weather = self.weather.conditions_at(self._absolute_hour)
        day_index = self.config.start_day_index + int(self._absolute_hour // 24)
        hour = self._absolute_hour % 24.0
        self._current_occ = self.occupancy.probability(day_index, hour)
        self._prev_hvac = 0.0
        return self._observation()

    def step(self, hvac_power: float) -> Tuple[List[float], float, bool, dict]:
        if self._step_count >= self._max_steps:
            raise RuntimeError("Episode already finished, call reset() before stepping again")

        hvac_power = float(max(0.0, min(self._max_power, hvac_power)))
        cfg = self.config

        current_occ = self._current_occ
        weather = self._current_weather
        internal_gain = current_occ * cfg.occupant_heat_gain * cfg.occupant_count
        state = self.model.step(
            cfg.time_step_hours,
            weather.outdoor_temp,
            weather.solar_irradiance,
            hvac_power,
            internal_gain,
        )

        comfort_error = abs(state.indoor - cfg.comfort_temperature)
        comfort_penalty = cfg.comfort_weight * comfort_error * (0.5 + current_occ)
        energy_penalty = cfg.energy_weight * (hvac_power / self._max_power)
        reward = -(comfort_penalty + energy_penalty)

        self._prev_hvac = hvac_power
        self._step_count += 1
        self._absolute_hour += cfg.time_step_hours

        next_day = cfg.start_day_index + int(self._absolute_hour // 24)
        next_hour = self._absolute_hour % 24.0
        self._current_occ = self.occupancy.probability(next_day, next_hour)
        self._current_weather = self.weather.conditions_at(self._absolute_hour)

        done = self._step_count >= self._max_steps
        return self._observation(), reward, done, {
            "occupancy": current_occ,
            "hvac_power": hvac_power,
        }

    def _observation(self) -> List[float]:
        hour = self._absolute_hour % 24.0
        state = self.model.state
        return [
            state.indoor,
            state.envelope,
            self._current_weather.outdoor_temp,
            self._current_occ,
            math.sin(2.0 * math.pi * hour / 24.0),
            math.cos(2.0 * math.pi * hour / 24.0),
            self._prev_hvac / self._max_power,
        ]


# ---------------------------------------------------------------------------
# PPO agent


@dataclass
class PPOConfig:
    learning_rate: float = 5e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    rollout_length: int = 64
    train_epochs: int = 4
    hidden_size: int = 32
    max_grad_norm: float = 1.0
    action_size: int = 1


class _LSTMActorCritic:
    def __init__(self, input_size: int, hidden_size: int, action_size: int, rng: random.Random) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self._rng = rng

        scale = 1.0 / math.sqrt(max(1.0, input_size + hidden_size))
        self.W_f = self._rand_matrix(hidden_size, input_size, scale)
        self.U_f = self._rand_matrix(hidden_size, hidden_size, scale)
        self.b_f = _zeros(hidden_size)

        self.W_i = self._rand_matrix(hidden_size, input_size, scale)
        self.U_i = self._rand_matrix(hidden_size, hidden_size, scale)
        self.b_i = _zeros(hidden_size)

        self.W_o = self._rand_matrix(hidden_size, input_size, scale)
        self.U_o = self._rand_matrix(hidden_size, hidden_size, scale)
        self.b_o = _zeros(hidden_size)

        self.W_c = self._rand_matrix(hidden_size, input_size, scale)
        self.U_c = self._rand_matrix(hidden_size, hidden_size, scale)
        self.b_c = _zeros(hidden_size)

        self.actor_weight = self._rand_matrix(action_size, hidden_size, scale)
        self.actor_bias = _zeros(action_size)
        self.value_weight = self._rand_vector(hidden_size, scale)
        self.value_bias = 0.0
        self.log_std = _zeros(action_size)

    def _rand_matrix(self, rows: int, cols: int, scale: float) -> List[List[float]]:
        return [[self._rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]

    def _rand_vector(self, length: int, scale: float) -> List[float]:
        return [self._rng.gauss(0.0, scale) for _ in range(length)]

    def init_state(self) -> Tuple[List[float], List[float]]:
        return _zeros(self.hidden_size), _zeros(self.hidden_size)

    def forward_step(
        self,
        x: Sequence[float],
        h_prev: Sequence[float],
        c_prev: Sequence[float],
    ) -> Tuple[List[float], float, List[float], List[float], List[float], Dict[str, List[float]]]:
        x_vec = _clone_vector(x)
        h_prev_vec = _clone_vector(h_prev)
        c_prev_vec = _clone_vector(c_prev)

        f = _vec_sigmoid(_vec_add(_vec_add(_mat_vec(self.W_f, x_vec), _mat_vec(self.U_f, h_prev_vec)), self.b_f))
        i = _vec_sigmoid(_vec_add(_vec_add(_mat_vec(self.W_i, x_vec), _mat_vec(self.U_i, h_prev_vec)), self.b_i))
        o = _vec_sigmoid(_vec_add(_vec_add(_mat_vec(self.W_o, x_vec), _mat_vec(self.U_o, h_prev_vec)), self.b_o))
        g = _vec_tanh(_vec_add(_vec_add(_mat_vec(self.W_c, x_vec), _mat_vec(self.U_c, h_prev_vec)), self.b_c))
        c = _vec_add(_vec_mul(f, c_prev_vec), _vec_mul(i, g))
        h = _vec_mul(o, _vec_tanh(c))

        mean = [ _dot(row, h) + self.actor_bias[idx] for idx, row in enumerate(self.actor_weight) ]
        value = _dot(self.value_weight, h) + self.value_bias

        cache = {
            "x": x_vec,
            "h_prev": h_prev_vec,
            "c_prev": c_prev_vec,
            "f": f,
            "i": i,
            "o": o,
            "g": g,
            "c": c,
            "h": h,
        }
        return mean, value, _clone_vector(self.log_std), h, c, cache

    def evaluate_value(self, x: Sequence[float], h_prev: Sequence[float], c_prev: Sequence[float]) -> float:
        _, value, _, _, _, _ = self.forward_step(x, h_prev, c_prev)
        return value

    def compute_gradients(
        self,
        states: Sequence[Sequence[float]],
        actions: Sequence[Sequence[float]],
        old_log_probs: Sequence[float],
        returns: Sequence[float],
        advantages: Sequence[float],
        init_hidden: Sequence[float],
        init_cell: Sequence[float],
        config: PPOConfig,
    ) -> Tuple[float, dict]:
        grads = {
            "W_f": _zeros_matrix(self.hidden_size, self.input_size),
            "U_f": _zeros_matrix(self.hidden_size, self.hidden_size),
            "b_f": _zeros(self.hidden_size),
            "W_i": _zeros_matrix(self.hidden_size, self.input_size),
            "U_i": _zeros_matrix(self.hidden_size, self.hidden_size),
            "b_i": _zeros(self.hidden_size),
            "W_o": _zeros_matrix(self.hidden_size, self.input_size),
            "U_o": _zeros_matrix(self.hidden_size, self.hidden_size),
            "b_o": _zeros(self.hidden_size),
            "W_c": _zeros_matrix(self.hidden_size, self.input_size),
            "U_c": _zeros_matrix(self.hidden_size, self.hidden_size),
            "b_c": _zeros(self.hidden_size),
            "actor_weight": _zeros_matrix(self.action_size, self.hidden_size),
            "actor_bias": _zeros(self.action_size),
            "value_weight": _zeros(self.hidden_size),
            "value_bias": 0.0,
            "log_std": _zeros(self.action_size),
        }

        caches: List[Dict[str, List[float]]] = []
        h = _clone_vector(init_hidden)
        c = _clone_vector(init_cell)
        means: List[List[float]] = []
        values: List[float] = []
        log_probs: List[float] = []
        std = [math.exp(v) for v in self.log_std]

        for state, action in zip(states, actions):
            mean, value, _, h, c, cache = self.forward_step(state, h, c)
            caches.append(cache)
            means.append(mean)
            values.append(value)
            diff = _vec_sub(action, mean)
            log_prob = -0.5 * sum((diff[i] / std[i]) ** 2 + 2.0 * self.log_std[i] + math.log(2.0 * math.pi) for i in range(self.action_size))
            log_probs.append(log_prob)

        total_loss = 0.0
        d_hidden: List[List[float]] = []

        for t in range(len(states)):
            advantage = advantages[t]
            ret = returns[t]
            value = values[t]
            mean = means[t]
            old_log_prob = old_log_probs[t]
            log_prob = log_probs[t]

            ratio = math.exp(log_prob - old_log_prob)
            clipped = max(min(ratio, 1.0 + config.clip_ratio), 1.0 - config.clip_ratio)
            loss1 = ratio * advantage
            loss2 = clipped * advantage
            if loss1 <= loss2:
                policy_loss = -loss1
                dlogprob = -advantage * ratio
            else:
                policy_loss = -loss2
                dlogprob = 0.0

            value_error = value - ret
            value_loss = 0.5 * value_error * value_error
            entropy = sum(0.5 * math.log(2.0 * math.pi * math.e) + ls for ls in self.log_std)
            total_loss += policy_loss + config.value_loss_coeff * value_loss - config.entropy_coeff * entropy

            d_value = config.value_loss_coeff * value_error
            diff = _vec_sub(actions[t], mean)
            d_mean = [dlogprob * diff[i] / (std[i] ** 2) for i in range(self.action_size)]
            d_log_std = [dlogprob * (-1.0 + (diff[i] ** 2) / (std[i] ** 2)) - config.entropy_coeff for i in range(self.action_size)]

            _add_matrix(grads["actor_weight"], _outer(d_mean, caches[t]["h"]))
            _add_vector(grads["actor_bias"], d_mean)
            _add_vector(grads["log_std"], d_log_std)
            _add_vector(grads["value_weight"], [d_value * v for v in caches[t]["h"]])
            grads["value_bias"] += d_value

            dh_from_actor = [sum(self.actor_weight[row][col] * d_mean[row] for row in range(self.action_size)) for col in range(self.hidden_size)]
            dh_from_value = [self.value_weight[col] * d_value for col in range(self.hidden_size)]
            d_hidden.append([dh_from_actor[col] + dh_from_value[col] for col in range(self.hidden_size)])

        lstm_grads = self._backward_lstm(caches, d_hidden)
        for key in ("W_f", "U_f", "b_f", "W_i", "U_i", "b_i", "W_o", "U_o", "b_o", "W_c", "U_c", "b_c"):
            _add_matrix(grads[key], lstm_grads[key]) if key.startswith("W") or key.startswith("U") else _add_vector(grads[key], lstm_grads[key])

        loss = total_loss / max(1, len(states))
        return loss, grads

    def _backward_lstm(self, caches: Sequence[Dict[str, List[float]]], d_hidden: Sequence[List[float]]) -> Dict[str, List[List[float]] | List[float]]:
        grads = {
            "W_f": _zeros_matrix(self.hidden_size, self.input_size),
            "U_f": _zeros_matrix(self.hidden_size, self.hidden_size),
            "b_f": _zeros(self.hidden_size),
            "W_i": _zeros_matrix(self.hidden_size, self.input_size),
            "U_i": _zeros_matrix(self.hidden_size, self.hidden_size),
            "b_i": _zeros(self.hidden_size),
            "W_o": _zeros_matrix(self.hidden_size, self.input_size),
            "U_o": _zeros_matrix(self.hidden_size, self.hidden_size),
            "b_o": _zeros(self.hidden_size),
            "W_c": _zeros_matrix(self.hidden_size, self.input_size),
            "U_c": _zeros_matrix(self.hidden_size, self.hidden_size),
            "b_c": _zeros(self.hidden_size),
        }

        dh_next = _zeros(self.hidden_size)
        dc_next = _zeros(self.hidden_size)

        for cache, dh in zip(reversed(caches), reversed(d_hidden)):
            dh_total = [dh[i] + dh_next[i] for i in range(self.hidden_size)]
            grads_step, dh_next, dc_next = self._backward_step(cache, dh_total, dc_next)
            for key in ("W_f", "U_f", "b_f", "W_i", "U_i", "b_i", "W_o", "U_o", "b_o", "W_c", "U_c", "b_c"):
                if key.startswith("W") or key.startswith("U"):
                    _add_matrix(grads[key], grads_step[key])
                else:
                    _add_vector(grads[key], grads_step[key])

        return grads

    def _backward_step(
        self,
        cache: Dict[str, List[float]],
        dh: Sequence[float],
        dc: Sequence[float],
    ) -> Tuple[Dict[str, List[List[float]] | List[float]], List[float], List[float]]:
        x = cache["x"]
        h_prev = cache["h_prev"]
        c_prev = cache["c_prev"]
        f = cache["f"]
        i = cache["i"]
        o = cache["o"]
        g = cache["g"]
        c = cache["c"]

        tanh_c = _vec_tanh(c)
        do = [dh[idx] * tanh_c[idx] for idx in range(self.hidden_size)]
        do_pre = [do[idx] * o[idx] * (1.0 - o[idx]) for idx in range(self.hidden_size)]

        dc_total = [dc[idx] + dh[idx] * o[idx] * (1.0 - tanh_c[idx] ** 2) for idx in range(self.hidden_size)]
        df = [dc_total[idx] * c_prev[idx] for idx in range(self.hidden_size)]
        df_pre = [df[idx] * f[idx] * (1.0 - f[idx]) for idx in range(self.hidden_size)]

        di = [dc_total[idx] * g[idx] for idx in range(self.hidden_size)]
        di_pre = [di[idx] * i[idx] * (1.0 - i[idx]) for idx in range(self.hidden_size)]

        dg = [dc_total[idx] * i[idx] for idx in range(self.hidden_size)]
        dg_pre = [dg[idx] * (1.0 - g[idx] ** 2) for idx in range(self.hidden_size)]

        dc_prev = [dc_total[idx] * f[idx] for idx in range(self.hidden_size)]

        grads = {
            "W_f": _outer(df_pre, x),
            "U_f": _outer(df_pre, h_prev),
            "b_f": df_pre,
            "W_i": _outer(di_pre, x),
            "U_i": _outer(di_pre, h_prev),
            "b_i": di_pre,
            "W_o": _outer(do_pre, x),
            "U_o": _outer(do_pre, h_prev),
            "b_o": do_pre,
            "W_c": _outer(dg_pre, x),
            "U_c": _outer(dg_pre, h_prev),
            "b_c": dg_pre,
        }

        dh_prev = _zeros(self.hidden_size)
        for col in range(self.hidden_size):
            dh_prev[col] = sum(
                self.U_f[row][col] * df_pre[row]
                + self.U_i[row][col] * di_pre[row]
                + self.U_o[row][col] * do_pre[row]
                + self.U_c[row][col] * dg_pre[row]
                for row in range(self.hidden_size)
            )

        return grads, dh_prev, dc_prev

    def apply_gradients(self, grads: dict, learning_rate: float, max_norm: float) -> None:
        squared_norm = 0.0
        squared_norm += _matrix_sum_squares(grads["W_f"]) + _matrix_sum_squares(grads["U_f"]) + _sum_squares(grads["b_f"])
        squared_norm += _matrix_sum_squares(grads["W_i"]) + _matrix_sum_squares(grads["U_i"]) + _sum_squares(grads["b_i"])
        squared_norm += _matrix_sum_squares(grads["W_o"]) + _matrix_sum_squares(grads["U_o"]) + _sum_squares(grads["b_o"])
        squared_norm += _matrix_sum_squares(grads["W_c"]) + _matrix_sum_squares(grads["U_c"]) + _sum_squares(grads["b_c"])
        squared_norm += _matrix_sum_squares(grads["actor_weight"]) + _sum_squares(grads["actor_bias"])
        squared_norm += _sum_squares(grads["value_weight"]) + grads["value_bias"] ** 2
        squared_norm += _sum_squares(grads["log_std"])

        norm = math.sqrt(max(squared_norm, 1e-12))
        scale = 1.0
        if norm > max_norm:
            scale = max_norm / norm

        def update_matrix(param: List[List[float]], grad: Sequence[Sequence[float]]) -> None:
            for row_idx in range(len(param)):
                for col_idx in range(len(param[row_idx])):
                    param[row_idx][col_idx] -= learning_rate * scale * grad[row_idx][col_idx]

        def update_vector(param: List[float], grad: Sequence[float]) -> None:
            for idx in range(len(param)):
                param[idx] -= learning_rate * scale * grad[idx]

        update_matrix(self.W_f, grads["W_f"])
        update_matrix(self.U_f, grads["U_f"])
        update_vector(self.b_f, grads["b_f"])
        update_matrix(self.W_i, grads["W_i"])
        update_matrix(self.U_i, grads["U_i"])
        update_vector(self.b_i, grads["b_i"])
        update_matrix(self.W_o, grads["W_o"])
        update_matrix(self.U_o, grads["U_o"])
        update_vector(self.b_o, grads["b_o"])
        update_matrix(self.W_c, grads["W_c"])
        update_matrix(self.U_c, grads["U_c"])
        update_vector(self.b_c, grads["b_c"])
        update_matrix(self.actor_weight, grads["actor_weight"])
        update_vector(self.actor_bias, grads["actor_bias"])
        update_vector(self.value_weight, grads["value_weight"])
        self.value_bias -= learning_rate * scale * grads["value_bias"]
        update_vector(self.log_std, grads["log_std"])


@dataclass
class _RolloutStep:
    state: List[float]
    action: List[float]
    log_prob: float
    value: float
    reward: float
    done: bool
    hidden: List[float]
    cell: List[float]


@dataclass
class _RolloutSegment:
    states: List[List[float]]
    actions: List[List[float]]
    old_log_probs: List[float]
    returns: List[float]
    advantages: List[float]
    init_hidden: List[float]
    init_cell: List[float]


class LSTMPPOAgent:
    """On-policy PPO agent with an LSTM backbone."""

    def __init__(
        self,
        observation_size: int,
        action_scale: float,
        config: PPOConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.config = config or PPOConfig()
        self._rng = random.Random(seed)
        self._network = _LSTMActorCritic(
            observation_size,
            self.config.hidden_size,
            self.config.action_size,
            random.Random(self._rng.random()),
        )
        self._action_scale = float(action_scale)
        self._current_obs: List[float] | None = None
        self._current_hidden = self._network.init_state()
        self._episode_returns: List[float] = []

    def _ensure_state(self, env: ThermalComfortEnv) -> None:
        if self._current_obs is None:
            self._current_obs = env.reset()
            self._current_hidden = self._network.init_state()

    def _sample_action(self, mean: Sequence[float], log_std: Sequence[float]) -> Tuple[List[float], float, float]:
        std = [math.exp(value) for value in log_std]
        action = [mean[idx] + std[idx] * self._rng.gauss(0.0, 1.0) for idx in range(len(mean))]
        log_prob = -0.5 * sum(
            ((action[idx] - mean[idx]) / std[idx]) ** 2 + 2.0 * log_std[idx] + math.log(2.0 * math.pi)
            for idx in range(len(mean))
        )
        hvac_fraction = _sigmoid(action[0]) if action else 0.0
        hvac_power = hvac_fraction * self._action_scale
        return [float(a) for a in action], log_prob, hvac_power

    def _finalise_segment(self, steps: List[_RolloutStep], last_value: float) -> _RolloutSegment:
        states = [step.state[:] for step in steps]
        actions = [step.action[:] for step in steps]
        old_log_probs = [step.log_prob for step in steps]
        values = [step.value for step in steps]
        rewards = [step.reward for step in steps]
        dones = [step.done for step in steps]

        returns = [0.0 for _ in steps]
        advantages = [0.0 for _ in steps]
        next_value = last_value
        gae = 0.0
        for idx in reversed(range(len(steps))):
            non_terminal = 0.0 if dones[idx] else 1.0
            delta = rewards[idx] + self.config.gamma * next_value * non_terminal - values[idx]
            gae = delta + self.config.gamma * self.config.gae_lambda * non_terminal * gae
            advantages[idx] = gae
            returns[idx] = gae + values[idx]
            next_value = values[idx]

        return _RolloutSegment(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            returns=returns,
            advantages=advantages,
            init_hidden=steps[0].hidden[:],
            init_cell=steps[0].cell[:],
        )

    def _collect_rollout(self, env: ThermalComfortEnv) -> Tuple[List[_RolloutSegment], List[float]]:
        self._ensure_state(env)
        obs = self._current_obs[:]
        h, c = self._current_hidden
        segments: List[_RolloutSegment] = []
        episode_returns: List[float] = []
        steps: List[_RolloutStep] = []
        partial_return = 0.0

        for _ in range(self.config.rollout_length):
            mean, value, log_std, next_h, next_c, _ = self._network.forward_step(obs, h, c)
            action, log_prob, hvac_power = self._sample_action(mean, log_std)
            next_obs, reward, done, _ = env.step(hvac_power)

            steps.append(
                _RolloutStep(
                    state=obs[:],
                    action=action[:],
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                    done=done,
                    hidden=h[:],
                    cell=c[:],
                )
            )
            partial_return += reward

            if done:
                segments.append(self._finalise_segment(steps, last_value=0.0))
                episode_returns.append(partial_return)
                steps = []
                partial_return = 0.0
                obs = env.reset()
                h, c = self._network.init_state()
            else:
                obs = next_obs
                h, c = next_h, next_c

        self._current_obs = obs[:]
        self._current_hidden = (h[:], c[:])

        if steps:
            last_value = self._network.evaluate_value(obs, h, c)
            segments.append(self._finalise_segment(steps, last_value))
            episode_returns.append(partial_return)

        return segments, episode_returns

    def _normalise_advantages(self, segments: Sequence[_RolloutSegment]) -> None:
        all_values: List[float] = [value for segment in segments for value in segment.advantages]
        if not all_values:
            return
        mean = sum(all_values) / len(all_values)
        variance = sum((value - mean) ** 2 for value in all_values) / len(all_values)
        std = math.sqrt(variance)
        if std < 1e-8:
            std = 1.0
        for segment in segments:
            segment.advantages = [(value - mean) / std for value in segment.advantages]

    def _update(self, segments: Sequence[_RolloutSegment]) -> float:
        self._normalise_advantages(segments)
        total_loss = 0.0
        batch_count = 0
        for _ in range(self.config.train_epochs):
            for segment in segments:
                loss, grads = self._network.compute_gradients(
                    segment.states,
                    segment.actions,
                    segment.old_log_probs,
                    segment.returns,
                    segment.advantages,
                    segment.init_hidden,
                    segment.init_cell,
                    self.config,
                )
                self._network.apply_gradients(grads, self.config.learning_rate, self.config.max_grad_norm)
                total_loss += loss
                batch_count += 1
        return total_loss / max(1, batch_count)

    def train(self, env: ThermalComfortEnv, iterations: int) -> dict:
        losses: List[float] = []
        episodic_returns: List[float] = []
        for _ in range(iterations):
            segments, returns = self._collect_rollout(env)
            if not segments:
                continue
            loss = self._update(segments)
            losses.append(loss)
            episodic_returns.extend(returns)
        self._episode_returns.extend(episodic_returns)
        return {"loss": losses, "episode_returns": episodic_returns}

    def act(self, observation: Sequence[float], hidden: Tuple[Sequence[float], Sequence[float]] | None = None) -> Tuple[float, Tuple[List[float], List[float]]]:
        if hidden is None:
            hidden = self._network.init_state()
        h, c = hidden
        mean, _, _, next_h, next_c, _ = self._network.forward_step(observation, h, c)
        hvac_power = _sigmoid(mean[0]) * self._action_scale if mean else 0.0
        return hvac_power, (next_h, next_c)
