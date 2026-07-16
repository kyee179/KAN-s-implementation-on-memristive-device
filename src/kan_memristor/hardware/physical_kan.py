"""Memristive odd-polynomial KAN layers and pulse-based training.

The model mirrors the hardware correspondence used in the project notes:

* Gilbert multiplier stages generate bounded voltage rows proportional to
  x, x^3, and x^5.
* Differential RRAM pairs store signed conductance contributions for every
  edge and every polynomial row.
* Node currents sum naturally, then a fixed current-to-voltage gain converts
  the summed current into the next layer signal.
* Backpropagation is used only to compute the desired direction of change;
  conductance is changed by discrete SET/RESET-like pulses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from kan_memristor.models import OddPolynomialKAN, OddPolynomialKANLayer
from kan_memristor.hardware.memristor import RRAMWeightMapper


@dataclass(frozen=True)
class HardwareKANScaling:
    """Fixed circuit scaling for a physical KAN layer.

    input_scale_v is the Mehonic-style fixed factor k that maps dimensionless
    layer inputs to safe device voltages. The multiplier chain is assumed to
    rescale x^3 and x^5 rows back into the same voltage window, so each power
    row is represented as k*x^p. current_to_voltage_gain is the fixed
    transimpedance-like gain that maps summed crossbar current into the next
    layer voltage/software signal.
    """

    input_scale_v: float = 0.2
    current_to_voltage_gain: float | None = None
    conductance_utilization: float = 0.85

    def __post_init__(self) -> None:
        if self.input_scale_v <= 0.0:
            raise ValueError("input_scale_v must be positive")
        if self.current_to_voltage_gain is not None and self.current_to_voltage_gain <= 0.0:
            raise ValueError("current_to_voltage_gain must be positive")
        if not 0.0 < self.conductance_utilization <= 1.0:
            raise ValueError("conductance_utilization must be in (0, 1]")


class PhysicalOddPolynomialKANLayer(nn.Module):
    """Odd-polynomial KAN layer represented by differential RRAM pairs."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        powers: tuple[int, ...] = (1, 3, 5),
        mapper: RRAMWeightMapper | None = None,
        scaling: HardwareKANScaling | None = None,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not powers:
            raise ValueError("powers must not be empty")
        if any(power <= 0 or power % 2 == 0 for power in powers):
            raise ValueError("powers must be positive odd integers")
        self.in_features = in_features
        self.out_features = out_features
        self.mapper = mapper or RRAMWeightMapper()
        self.scaling = scaling or HardwareKANScaling()
        self.register_buffer("powers", torch.tensor(powers, dtype=torch.float32))
        self.g_pos = nn.Parameter(torch.full((out_features, in_features, len(powers)), self.mapper.g_min))
        self.g_neg = nn.Parameter(torch.full((out_features, in_features, len(powers)), self.mapper.g_min))
        if bias is None:
            bias = torch.zeros(out_features, dtype=torch.float32)
        self.bias = nn.Parameter(bias.detach().clone().float())
        gain = self.scaling.current_to_voltage_gain or 1.0
        self.register_buffer("current_to_voltage_gain", torch.tensor(float(gain), dtype=torch.float32))

    @property
    def conductance_span(self) -> float:
        return self.mapper.g_max - self.mapper.g_min

    @classmethod
    def from_software_layer(
        cls,
        layer: OddPolynomialKANLayer,
        mapper: RRAMWeightMapper | None = None,
        scaling: HardwareKANScaling | None = None,
    ) -> "PhysicalOddPolynomialKANLayer":
        """Map a trained software odd-polynomial layer onto RRAM pairs."""

        mapper = mapper or RRAMWeightMapper()
        scaling = scaling or HardwareKANScaling()
        coefficients = layer.coefficients.detach().cpu().float()
        max_abs = float(coefficients.abs().max().item()) if coefficients.numel() else 1.0
        span = mapper.g_max - mapper.g_min
        gain = scaling.current_to_voltage_gain
        if gain is None:
            gain = max(max_abs / (scaling.conductance_utilization * span * scaling.input_scale_v), 1.0)
            scaling = HardwareKANScaling(
                input_scale_v=scaling.input_scale_v,
                current_to_voltage_gain=gain,
                conductance_utilization=scaling.conductance_utilization,
            )
        physical = cls(
            layer.in_features,
            layer.out_features,
            powers=tuple(int(power.item()) for power in layer.powers),
            mapper=mapper,
            scaling=scaling,
            bias=layer.bias.detach().cpu(),
        )
        physical.current_to_voltage_gain.fill_(float(gain))
        desired_delta_g = coefficients / (float(gain) * scaling.input_scale_v)
        desired_delta_g = desired_delta_g.clamp(min=-span, max=span)
        g_pos = torch.full_like(desired_delta_g, mapper.g_min) + desired_delta_g.clamp_min(0.0)
        g_neg = torch.full_like(desired_delta_g, mapper.g_min) + (-desired_delta_g).clamp_min(0.0)
        with torch.no_grad():
            physical.g_pos.copy_(physical.quantize(g_pos))
            physical.g_neg.copy_(physical.quantize(g_neg))
        return physical

    def quantize(self, conductance: torch.Tensor) -> torch.Tensor:
        levels = torch.linspace(
            self.mapper.g_min,
            self.mapper.g_max,
            self.mapper.n_states,
            dtype=conductance.dtype,
            device=conductance.device,
        )
        indices = torch.abs(conductance.unsqueeze(-1) - levels).argmin(dim=-1)
        return levels[indices]

    def effective_coefficients(self) -> torch.Tensor:
        """Return dimensionless coefficients implied by the physical layer."""

        return self.current_to_voltage_gain * self.scaling.input_scale_v * (self.g_pos - self.g_neg)

    def edge_currents(self, x: torch.Tensor) -> torch.Tensor:
        voltage_rows = self.scaling.input_scale_v * x.unsqueeze(-1).pow(self.powers.to(x.device, x.dtype))
        return torch.einsum("bik,oik->bo", voltage_rows, self.g_pos - self.g_neg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        currents = self.edge_currents(x)
        return self.current_to_voltage_gain.to(x.device, x.dtype) * currents + self.bias.to(x.device, x.dtype)

    def clamp_and_quantize_(self) -> None:
        with torch.no_grad():
            self.g_pos.clamp_(self.mapper.g_min, self.mapper.g_max)
            self.g_neg.clamp_(self.mapper.g_min, self.mapper.g_max)
            self.g_pos.copy_(self.quantize(self.g_pos))
            self.g_neg.copy_(self.quantize(self.g_neg))


class PhysicalOddPolynomialKAN(nn.Module):
    """Stack of memristive odd-polynomial KAN layers."""

    def __init__(self, layers: Iterable[PhysicalOddPolynomialKANLayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(list(layers))
        if not self.layers:
            raise ValueError("PhysicalOddPolynomialKAN requires at least one layer")

    @classmethod
    def from_software_model(
        cls,
        model: OddPolynomialKAN,
        mapper: RRAMWeightMapper | None = None,
        input_scale_v: float = 0.2,
        current_to_voltage_gain: float | None = None,
    ) -> "PhysicalOddPolynomialKAN":
        layers: list[PhysicalOddPolynomialKANLayer] = []
        for software_layer in model.layers:
            scaling = HardwareKANScaling(
                input_scale_v=input_scale_v,
                current_to_voltage_gain=current_to_voltage_gain,
            )
            layers.append(PhysicalOddPolynomialKANLayer.from_software_layer(software_layer, mapper=mapper, scaling=scaling))
        return cls(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def conductance_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for layer in self.layers:
            params.extend([layer.g_pos, layer.g_neg])
        return params

    def clamp_and_quantize_(self) -> None:
        for layer in self.layers:
            layer.clamp_and_quantize_()

    def count_memristors(self) -> int:
        return sum(layer.g_pos.numel() + layer.g_neg.numel() for layer in self.layers)


@dataclass(frozen=True)
class PulseUpdateConfig:
    """SET/RESET pulse optimizer settings."""

    max_pulses_per_update: int = 1
    gradient_deadzone_quantile: float = 0.5
    bias_learning_rate: float = 1e-3
    conductance_learning_rate: float = 1e-10
    quantize_after_update: bool = True

    def __post_init__(self) -> None:
        if self.max_pulses_per_update < 1:
            raise ValueError("max_pulses_per_update must be at least 1")
        if not 0.0 <= self.gradient_deadzone_quantile < 1.0:
            raise ValueError("gradient_deadzone_quantile must be in [0, 1)")
        if self.conductance_learning_rate <= 0.0:
            raise ValueError("conductance_learning_rate must be positive")


class MemristivePulseOptimizer:
    """Manual optimizer that converts gradients into discrete conductance pulses."""

    def __init__(self, model: PhysicalOddPolynomialKAN, config: PulseUpdateConfig | None = None) -> None:
        self.model = model
        self.config = config or PulseUpdateConfig()
        self.total_set_pulses = 0
        self.total_reset_pulses = 0
        self._residuals: dict[int, torch.Tensor] = {}

    def zero_grad(self) -> None:
        for parameter in self.model.parameters():
            parameter.grad = None

    def step(self) -> dict[str, int]:
        step_set = 0
        step_reset = 0
        with torch.no_grad():
            for layer in self.model.layers:
                g_step = (layer.mapper.g_max - layer.mapper.g_min) / (layer.mapper.n_states - 1)
                for conductance in (layer.g_pos, layer.g_neg):
                    if conductance.grad is None:
                        continue
                    grad = conductance.grad
                    abs_grad = grad.abs()
                    max_grad = float(abs_grad.max().item()) if abs_grad.numel() else 0.0
                    if max_grad <= 0.0:
                        continue
                    if self.config.gradient_deadzone_quantile > 0.0:
                        threshold = torch.quantile(abs_grad.flatten(), self.config.gradient_deadzone_quantile)
                    else:
                        threshold = torch.tensor(0.0, dtype=abs_grad.dtype, device=abs_grad.device)
                    desired = -self.config.conductance_learning_rate * grad
                    desired = torch.where(abs_grad >= threshold, desired, torch.zeros_like(desired))
                    residual = self._residuals.setdefault(id(conductance), torch.zeros_like(conductance))
                    residual.add_(desired)
                    pulses = torch.floor(residual.abs() / g_step).to(torch.int64)
                    pulses = pulses.clamp(min=0, max=self.config.max_pulses_per_update)
                    delta = torch.sign(residual) * pulses.to(conductance.dtype) * g_step
                    residual.sub_(delta)
                    conductance.add_(delta)
                    step_set += int((delta > 0.0).sum().item())
                    step_reset += int((delta < 0.0).sum().item())
                if layer.bias.grad is not None and self.config.bias_learning_rate > 0.0:
                    layer.bias.add_(-self.config.bias_learning_rate * layer.bias.grad)
                if self.config.quantize_after_update:
                    layer.clamp_and_quantize_()
                else:
                    layer.g_pos.clamp_(layer.mapper.g_min, layer.mapper.g_max)
                    layer.g_neg.clamp_(layer.mapper.g_min, layer.mapper.g_max)
        self.total_set_pulses += step_set
        self.total_reset_pulses += step_reset
        self.zero_grad()
        return {"set_pulses": step_set, "reset_pulses": step_reset}


