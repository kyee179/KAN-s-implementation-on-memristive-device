"""Behavioral CMOS Gilbert voltage multiplier model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ArrayLike = float | np.ndarray


@dataclass(frozen=True)
class GilbertMultiplierParameters:
    """First-order parameters for the CMOS Gilbert voltage multiplier.

    Defaults follow the Renduchintala et al. 45 nm CMOS study: 1 V supply,
    approximately +/-400 mV differential input sweep, 10 GHz 3 dB bandwidth,
    and 440 uW power dissipation. The product gain is left explicit because it
    depends on the circuit biasing and load used for a specific implementation.
    """

    supply_voltage: float = 1.0
    input_linear_range: float = 0.4
    product_gain: float = 1.0
    bandwidth_hz: float = 10e9
    power_w: float = 440e-6
    voltage_gain_db: float = 5.5
    soft_clip: bool = True


class GilbertMultiplier:
    """Approximate analog product block with finite range and bandwidth."""

    def __init__(self, parameters: GilbertMultiplierParameters | None = None) -> None:
        self.parameters = parameters or GilbertMultiplierParameters()

    def multiply(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        p = self.parameters
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if p.soft_clip:
            x_eff = p.input_linear_range * np.tanh(x_arr / p.input_linear_range)
            y_eff = p.input_linear_range * np.tanh(y_arr / p.input_linear_range)
        else:
            x_eff = np.clip(x_arr, -p.input_linear_range, p.input_linear_range)
            y_eff = np.clip(y_arr, -p.input_linear_range, p.input_linear_range)
        output = p.product_gain * x_eff * y_eff
        return np.clip(output, -0.5 * p.supply_voltage, 0.5 * p.supply_voltage)

    def cascade_power(self, x: ArrayLike, power: int) -> np.ndarray:
        """Generate x**power by repeated Gilbert multiplications."""

        if power < 1:
            raise ValueError("power must be positive")
        result = np.asarray(x, dtype=float)
        for _ in range(power - 1):
            result = self.multiply(result, x)
        return result

    def lowpass_response(self, samples: ArrayLike, dt: float) -> np.ndarray:
        """Apply a first-order low-pass response using the 3 dB bandwidth."""

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        p = self.parameters
        values = np.asarray(samples, dtype=float)
        tau = 1.0 / (2.0 * np.pi * p.bandwidth_hz)
        alpha = dt / (tau + dt)
        output = np.empty_like(values, dtype=float)
        previous = 0.0
        for index, value in np.ndenumerate(values):
            previous = previous + alpha * (float(value) - previous)
            output[index] = previous
        return output

    def energy_for_operations(self, num_operations: int, clock_hz: float | None = None) -> float:
        """Estimate dynamic wall-clock energy from power and operation rate."""

        if num_operations < 0:
            raise ValueError("num_operations must be non-negative")
        rate = self.parameters.bandwidth_hz if clock_hz is None else clock_hz
        if rate <= 0.0:
            raise ValueError("clock_hz must be positive")
        return self.parameters.power_w * num_operations / rate