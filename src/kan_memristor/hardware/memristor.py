"""Behavioral memristive-device models for hardware-aware KAN simulation.

The dynamic device follows the compact memdiode structure used by Aguirre,
Sune, and Miranda: a hyperbolic-sine conduction law whose amplitude depends on
an internal memory state, plus a SET/RESET state-balance equation. The weight
mapper follows the simulation logic used by Mehonic et al.: continuous weights
are represented by finite conductance states and can include variability or
faults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ArrayLike = float | np.ndarray


@dataclass(frozen=True)
class DynamicMemdiodeParameters:
    """Parameters for a compact bipolar resistive-switching memdiode.

    Defaults are close to the LTSpice example values reported by Aguirre et al.
    They are not a fitted device deck; use them as a starting point for network
    experiments before replacing them with calibrated measurements.
    """

    i_off: float = 1e-7
    i_on: float = 1e-2
    alpha_off: float = 2.0
    alpha_on: float = 2.0
    r_off: float = 10.0
    r_on: float = 10.0
    r_internal: float = 50.0
    r_pristine_parallel: float = 1e10
    eta_set: float = 50.0
    eta_reset: float = 100.0
    v_set: float = 1.4
    v_reset: float = -0.4
    v_snapback: float = 0.4
    snapback_current: float = 2e-4
    snapforward_gamma: float = 1.0
    min_tau_s: float = 1e-12
    max_tau_s: float = 1e12


class DynamicMemdiode:
    """Compact dynamic memdiode model with state lambda in [0, 1]."""

    def __init__(self, parameters: DynamicMemdiodeParameters | None = None, initial_state: float = 0.0) -> None:
        self.parameters = parameters or DynamicMemdiodeParameters()
        self.state = float(np.clip(initial_state, 0.0, 1.0))

    def _interpolate(self, off: float, on: float, state: ArrayLike | None = None) -> np.ndarray:
        lam = self.state if state is None else state
        lam_array = np.asarray(lam, dtype=float)
        return off + (on - off) * np.clip(lam_array, 0.0, 1.0)

    def i0(self, state: ArrayLike | None = None) -> np.ndarray:
        p = self.parameters
        return self._interpolate(p.i_off, p.i_on, state)

    def alpha(self, state: ArrayLike | None = None) -> np.ndarray:
        p = self.parameters
        return self._interpolate(p.alpha_off, p.alpha_on, state)

    def series_resistance(self, state: ArrayLike | None = None) -> np.ndarray:
        p = self.parameters
        return p.r_internal + self._interpolate(p.r_off, p.r_on, state)

    def current(self, voltage: ArrayLike, state: ArrayLike | None = None, iterations: int = 40) -> np.ndarray:
        """Solve I = I0(lambda) * sinh(alpha(lambda) * (V - R(lambda)*I))."""

        p = self.parameters
        v = np.asarray(voltage, dtype=float)
        i0 = self.i0(state)
        alpha = self.alpha(state)
        resistance = self.series_resistance(state)
        current = np.zeros(np.broadcast(v, i0, alpha, resistance).shape, dtype=float)
        v = np.broadcast_to(v, current.shape)
        i0 = np.broadcast_to(i0, current.shape)
        alpha = np.broadcast_to(alpha, current.shape)
        resistance = np.broadcast_to(resistance, current.shape)

        for _ in range(iterations):
            argument = np.clip(alpha * (v - resistance * current), -60.0, 60.0)
            sinh_argument = np.sinh(argument)
            cosh_argument = np.cosh(argument)
            residual = current - i0 * sinh_argument
            derivative = 1.0 + i0 * alpha * resistance * cosh_argument
            current -= residual / np.maximum(derivative, 1e-30)

        return current + v / p.r_pristine_parallel

    def state_derivative(self, voltage: ArrayLike, state: ArrayLike | None = None) -> np.ndarray:
        """Return d(lambda)/dt from the SET/RESET balance equation."""

        p = self.parameters
        lam = self.state if state is None else state
        lam = np.asarray(lam, dtype=float)
        v = np.asarray(voltage, dtype=float)
        current = self.current(v, lam)
        internal_voltage = v - self.series_resistance(lam) * current
        set_threshold = np.where(np.abs(current) > p.snapback_current, p.v_snapback, p.v_set)
        tau_set = np.exp(np.clip(-p.eta_set * (internal_voltage - set_threshold), -60.0, 60.0))
        tau_reset = np.exp(
            np.clip(p.eta_reset * np.power(np.clip(lam, 0.0, 1.0), p.snapforward_gamma) * (internal_voltage - p.v_reset), -60.0, 60.0)
        )
        tau_set = np.clip(tau_set, p.min_tau_s, p.max_tau_s)
        tau_reset = np.clip(tau_reset, p.min_tau_s, p.max_tau_s)
        return np.where(v >= 0.0, (1.0 - lam) / tau_set, -lam / tau_reset)

    def step(self, voltage: float, dt: float) -> float:
        """Advance the internal memory state by one explicit Euler step."""

        derivative = float(self.state_derivative(voltage, self.state))
        self.state = float(np.clip(self.state + dt * derivative, 0.0, 1.0))
        return self.state

    def conductance(self, read_voltage: float = 0.2, state: ArrayLike | None = None) -> np.ndarray:
        """Return read conductance G = I/V at a small read voltage."""

        if read_voltage == 0.0:
            raise ValueError("read_voltage must be non-zero")
        return self.current(read_voltage, state) / read_voltage


@dataclass(frozen=True)
class RRAMWeightMapper:
    """Map software weights onto finite RRAM conductance states."""

    r_lrs: float = 1e4
    r_hrs: float = 1e6
    n_states: int = 16
    read_voltage: float = 0.5
    variability_std: float = 0.0
    stuck_lrs_probability: float = 0.0
    stuck_hrs_probability: float = 0.0
    seed: int = 1

    @property
    def g_min(self) -> float:
        return 1.0 / self.r_hrs

    @property
    def g_max(self) -> float:
        return 1.0 / self.r_lrs

    @property
    def resistance_ratio(self) -> float:
        return self.r_hrs / self.r_lrs

    def conductance_levels(self) -> np.ndarray:
        if self.n_states < 2:
            raise ValueError("n_states must be at least 2")
        return np.linspace(self.g_min, self.g_max, self.n_states)

    def quantize_conductance(self, conductance: ArrayLike) -> np.ndarray:
        levels = self.conductance_levels()
        g = np.asarray(conductance, dtype=float)
        indices = np.abs(g[..., None] - levels).argmin(axis=-1)
        return levels[indices]

    def signed_weight_to_differential_pair(
        self,
        weight: ArrayLike,
        max_abs_weight: float | None = None,
        apply_nonidealities: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Represent signed weights as a positive/negative conductance pair."""

        w = np.asarray(weight, dtype=float)
        if max_abs_weight is None:
            max_abs_weight = float(np.max(np.abs(w))) if w.size else 1.0
        max_abs_weight = max(max_abs_weight, 1e-12)
        normalized = np.clip(w / max_abs_weight, -1.0, 1.0)
        span = self.g_max - self.g_min
        g_pos = self.g_min + np.maximum(normalized, 0.0) * span
        g_neg = self.g_min + np.maximum(-normalized, 0.0) * span
        g_pos = self.quantize_conductance(g_pos)
        g_neg = self.quantize_conductance(g_neg)
        if apply_nonidealities:
            g_pos = self.apply_nonidealities(g_pos)
            g_neg = self.apply_nonidealities(g_neg)
        return g_pos, g_neg, max_abs_weight

    def differential_pair_to_weight(self, g_pos: ArrayLike, g_neg: ArrayLike, max_abs_weight: float) -> np.ndarray:
        span = self.g_max - self.g_min
        return (np.asarray(g_pos, dtype=float) - np.asarray(g_neg, dtype=float)) / span * max_abs_weight

    def apply_nonidealities(self, conductance: ArrayLike) -> np.ndarray:
        """Apply simple variability and stuck-device effects."""

        rng = np.random.default_rng(self.seed)
        g = np.asarray(conductance, dtype=float).copy()
        if self.variability_std > 0.0:
            g *= rng.lognormal(mean=0.0, sigma=self.variability_std, size=g.shape)
        if self.stuck_lrs_probability > 0.0:
            mask = rng.random(g.shape) < self.stuck_lrs_probability
            g = np.where(mask, self.g_max, g)
        if self.stuck_hrs_probability > 0.0:
            mask = rng.random(g.shape) < self.stuck_hrs_probability
            g = np.where(mask, self.g_min, g)
        return np.clip(g, self.g_min, self.g_max)

    def voltage_nonlinear_current(self, conductance: ArrayLike, voltage: ArrayLike, nonlinearity: float = 0.0) -> np.ndarray:
        """Approximate RRAM I/V nonlinearity around the read conductance."""

        g = np.asarray(conductance, dtype=float)
        v = np.asarray(voltage, dtype=float)
        scale = 1.0 + nonlinearity * (v / self.read_voltage) ** 2
        return g * v * scale

    def conductance_linearity(self, conductance: ArrayLike, voltage: float, nonlinearity: float = 0.0) -> np.ndarray:
        """Return G(V) / G(0.5 V), following the conductance-linearity idea."""

        current_v = self.voltage_nonlinear_current(conductance, voltage, nonlinearity)
        current_half = self.voltage_nonlinear_current(conductance, 0.5 * voltage, nonlinearity)
        g_v = current_v / voltage
        g_half = current_half / (0.5 * voltage)
        return g_v / g_half