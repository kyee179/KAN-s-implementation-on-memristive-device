"""Energy estimates for physical KAN and digital MLP comparisons.

The estimates are intentionally transparent and parameterized. They combine the
paper-based hardware assumptions already used in this project:

* RRAM read energy from V^2 * G * t_read.
* Gilbert multiplier energy from P / f using the Renduchintala et al. default
  multiplier parameters.
* DynamicMemdiode pulse energy from |V * I(V, state)| * pulse_width.
* Digital MLP energy from a configurable energy-per-MAC assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch

from kan_memristor.hardware.gilbert_multiplier import GilbertMultiplier
from kan_memristor.hardware.memristor import DynamicMemdiode
from kan_memristor.hardware.physical_kan import DynamicPulseConfig, PhysicalOddPolynomialKAN


@dataclass(frozen=True)
class InferenceEnergyBreakdown:
    rram_read_j_per_sample: float
    gilbert_j_per_sample: float
    peripheral_j_per_sample: float
    total_j_per_sample: float
    memristor_count: int
    gilbert_multiplier_count: int
    read_time_s: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ProgrammingEnergyBreakdown:
    set_pulse_j: float
    reset_pulse_j: float
    set_pulses: int
    reset_pulses: int
    total_set_j: float
    total_reset_j: float
    total_programming_j: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class MLPEnergyBreakdown:
    mac_count: int
    energy_per_mac_j: float
    inference_j_per_sample: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def count_mlp_macs(widths: list[int]) -> int:
    """Count dense multiply-accumulate operations for one MLP inference."""

    if len(widths) < 2:
        raise ValueError("widths must contain at least input and output sizes")
    return sum(in_features * out_features for in_features, out_features in zip(widths[:-1], widths[1:]))


def estimate_physical_kan_inference_energy(
    model: PhysicalOddPolynomialKAN,
    x: torch.Tensor,
    read_time_s: float = 1e-9,
    peripheral_j_per_sample: float = 0.0,
) -> InferenceEnergyBreakdown:
    """Estimate average physical KAN inference energy per sample."""

    if read_time_s <= 0.0:
        raise ValueError("read_time_s must be positive")
    if peripheral_j_per_sample < 0.0:
        raise ValueError("peripheral_j_per_sample must be non-negative")

    batch_size = max(int(x.shape[0]), 1)
    total_rram_energy = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    current = x
    for index, layer in enumerate(model.layers):
        voltage_rows = layer.voltage_rows(current)
        conductance_sum = layer.g_pos.detach() + layer.g_neg.detach()
        layer_energy = torch.einsum("bik,oik->b", voltage_rows.pow(2), conductance_sum)
        total_rram_energy = total_rram_energy + layer_energy.sum() * read_time_s
        current = layer(current)
        if index < len(model.layers) - 1:
            from kan_memristor.models import apply_inter_layer_normalization

            current = apply_inter_layer_normalization(
                current,
                mode=layer.forward_config.inter_layer_normalization,
                gain=layer.forward_config.normalization_gain,
            )

    rram_j_per_sample = float((total_rram_energy / batch_size).detach().cpu().item())
    gilbert_count = model.count_gilbert_multipliers()
    gilbert_params = model.layers[0].forward_config.gilbert_parameters
    gilbert_j_per_sample = GilbertMultiplier(gilbert_params).energy_for_operations(gilbert_count)
    total = rram_j_per_sample + gilbert_j_per_sample + peripheral_j_per_sample
    return InferenceEnergyBreakdown(
        rram_read_j_per_sample=rram_j_per_sample,
        gilbert_j_per_sample=gilbert_j_per_sample,
        peripheral_j_per_sample=peripheral_j_per_sample,
        total_j_per_sample=total,
        memristor_count=model.count_memristors(),
        gilbert_multiplier_count=gilbert_count,
        read_time_s=read_time_s,
    )


def estimate_dynamic_pulse_energy(
    pulse_config: DynamicPulseConfig,
    set_pulses: int,
    reset_pulses: int,
    states: np.ndarray | None = None,
) -> ProgrammingEnergyBreakdown:
    """Estimate SET/RESET programming energy from the DynamicMemdiode model."""

    if set_pulses < 0 or reset_pulses < 0:
        raise ValueError("pulse counts must be non-negative")
    if states is None:
        states = np.linspace(0.0, 1.0, 101)
    states = np.asarray(states, dtype=float)
    device = DynamicMemdiode(pulse_config.memdiode_parameters)
    set_current = np.abs(device.current(pulse_config.set_voltage, state=states))
    reset_current = np.abs(device.current(pulse_config.reset_voltage, state=states))
    set_pulse_j = float(np.mean(np.abs(pulse_config.set_voltage) * set_current * pulse_config.pulse_width_s))
    reset_pulse_j = float(np.mean(np.abs(pulse_config.reset_voltage) * reset_current * pulse_config.pulse_width_s))
    total_set = set_pulse_j * set_pulses
    total_reset = reset_pulse_j * reset_pulses
    return ProgrammingEnergyBreakdown(
        set_pulse_j=set_pulse_j,
        reset_pulse_j=reset_pulse_j,
        set_pulses=set_pulses,
        reset_pulses=reset_pulses,
        total_set_j=total_set,
        total_reset_j=total_reset,
        total_programming_j=total_set + total_reset,
    )


def estimate_mlp_inference_energy(widths: list[int], energy_per_mac_j: float = 4.6e-12) -> MLPEnergyBreakdown:
    """Estimate digital MLP inference energy from a configurable MAC cost."""

    if energy_per_mac_j <= 0.0:
        raise ValueError("energy_per_mac_j must be positive")
    macs = count_mlp_macs(widths)
    return MLPEnergyBreakdown(
        mac_count=macs,
        energy_per_mac_j=energy_per_mac_j,
        inference_j_per_sample=macs * energy_per_mac_j,
    )
