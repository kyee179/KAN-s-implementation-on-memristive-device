"""Physical generalized-bell basis KAN layers.

The layer models a Dorzhigulov-James GBF current cell followed by a TIA
and a differential RRAM crossbar column. The GBF shape parameters are fixed;
only the crossbar coefficients are learned or mapped from software.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import torch
from torch import nn

from kan_memristor.hardware.memristor import RRAMWeightMapper
from kan_memristor.models import GeneralizedBellKAN, GeneralizedBellKANLayer, apply_inter_layer_normalization


@dataclass(frozen=True)
class GBFCellConfig:
    """Compact GBF-cell and TIA assumptions."""

    output_current_peak_a: float = 1e-6
    tia_transresistance_ohm: float = 1e6
    cell_power_w: float = 4.1e-6
    tia_power_w: float = 1e-6
    cell_area_um2: float = 10.0

    def __post_init__(self) -> None:
        if self.output_current_peak_a <= 0.0:
            raise ValueError("output_current_peak_a must be positive")
        if self.tia_transresistance_ohm <= 0.0:
            raise ValueError("tia_transresistance_ohm must be positive")
        if self.cell_power_w < 0.0:
            raise ValueError("cell_power_w must be non-negative")
        if self.tia_power_w < 0.0:
            raise ValueError("tia_power_w must be non-negative")
        if self.cell_area_um2 < 0.0:
            raise ValueError("cell_area_um2 must be non-negative")

    @property
    def row_voltage_scale_v(self) -> float:
        return self.output_current_peak_a * self.tia_transresistance_ohm


@dataclass(frozen=True)
class GBFCrossbarScaling:
    """Scaling from GBF/TIA row voltages to output node voltage."""

    current_to_voltage_gain: float | None = None
    conductance_utilization: float = 0.85

    def __post_init__(self) -> None:
        if self.current_to_voltage_gain is not None and self.current_to_voltage_gain <= 0.0:
            raise ValueError("current_to_voltage_gain must be positive")
        if not 0.0 < self.conductance_utilization <= 1.0:
            raise ValueError("conductance_utilization must be in (0, 1]")


@dataclass(frozen=True)
class GBFForwardConfig:
    """Forward-path options for the GBF/TIA/crossbar layer."""

    inter_layer_normalization: str = "tanh"
    normalization_gain: float = 1.0
    adc_clip_voltage: float | None = None

    def __post_init__(self) -> None:
        if self.adc_clip_voltage is not None and self.adc_clip_voltage <= 0.0:
            raise ValueError("adc_clip_voltage must be positive")
        apply_inter_layer_normalization(
            torch.zeros(1),
            mode=self.inter_layer_normalization,
            gain=self.normalization_gain,
        )


class PhysicalGeneralizedBellKANLayer(nn.Module):
    """Fixed GBF-cell bank feeding a differential RRAM crossbar."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_basis: int = 9,
        centers: torch.Tensor | None = None,
        widths: torch.Tensor | None = None,
        slopes: torch.Tensor | None = None,
        mapper: RRAMWeightMapper | None = None,
        cell_config: GBFCellConfig | None = None,
        scaling: GBFCrossbarScaling | None = None,
        forward_config: GBFForwardConfig | None = None,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if num_basis < 2:
            raise ValueError("num_basis must be at least 2")
        self.in_features = in_features
        self.out_features = out_features
        self.num_basis = num_basis
        self.mapper = mapper or RRAMWeightMapper()
        self.cell_config = cell_config or GBFCellConfig()
        self.scaling = scaling or GBFCrossbarScaling()
        self.forward_config = forward_config or GBFForwardConfig()
        if centers is None:
            centers = torch.linspace(-1.0, 1.0, num_basis)
        if widths is None:
            widths = torch.full((num_basis,), 1.5 * 2.0 / (num_basis - 1))
        if slopes is None:
            slopes = torch.full((num_basis,), 2.0)
        if centers.shape != (num_basis,) or widths.shape != (num_basis,) or slopes.shape != (num_basis,):
            raise ValueError("centers, widths, and slopes must have shape (num_basis,)")
        if torch.any(widths <= 0.0) or torch.any(slopes <= 0.0):
            raise ValueError("widths and slopes must be positive")
        self.register_buffer("centers", centers.detach().clone().float())
        self.register_buffer("widths", widths.detach().clone().float())
        self.register_buffer("slopes", slopes.detach().clone().float())
        self.g_pos = nn.Parameter(torch.full((out_features, in_features, num_basis), self.mapper.g_min))
        self.g_neg = nn.Parameter(torch.full((out_features, in_features, num_basis), self.mapper.g_min))
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
        layer: GeneralizedBellKANLayer,
        mapper: RRAMWeightMapper | None = None,
        cell_config: GBFCellConfig | None = None,
        scaling: GBFCrossbarScaling | None = None,
        forward_config: GBFForwardConfig | None = None,
    ) -> "PhysicalGeneralizedBellKANLayer":
        mapper = mapper or RRAMWeightMapper()
        cell_config = cell_config or GBFCellConfig()
        scaling = scaling or GBFCrossbarScaling()
        coefficients = layer.coefficients.detach().cpu().float()
        max_abs = float(coefficients.abs().max().item()) if coefficients.numel() else 1.0
        span = mapper.g_max - mapper.g_min
        row_scale = cell_config.row_voltage_scale_v
        gain = scaling.current_to_voltage_gain
        if gain is None:
            gain = max(max_abs / (scaling.conductance_utilization * span * row_scale), 1.0)
            scaling = replace(scaling, current_to_voltage_gain=gain)
        physical = cls(
            layer.in_features,
            layer.out_features,
            num_basis=layer.num_basis,
            centers=layer.centers.detach().cpu(),
            widths=layer.widths.detach().cpu(),
            slopes=layer.slopes.detach().cpu(),
            mapper=mapper,
            cell_config=cell_config,
            scaling=scaling,
            forward_config=forward_config,
            bias=layer.bias.detach().cpu(),
        )
        physical.current_to_voltage_gain.fill_(float(gain))
        desired_delta_g = coefficients / (float(gain) * row_scale)
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

    def gbf_currents(self, x: torch.Tensor) -> torch.Tensor:
        centers = self.centers.to(dtype=x.dtype, device=x.device)
        widths = self.widths.to(dtype=x.dtype, device=x.device)
        slopes = self.slopes.to(dtype=x.dtype, device=x.device)
        normalized = (x.unsqueeze(-1) - centers).abs() / widths.clamp_min(torch.finfo(x.dtype).eps)
        basis = 1.0 / (1.0 + normalized.pow(2.0 * slopes))
        return self.cell_config.output_current_peak_a * basis

    def tia_voltages(self, x: torch.Tensor) -> torch.Tensor:
        return self.cell_config.tia_transresistance_ohm * self.gbf_currents(x)

    def edge_currents(self, x: torch.Tensor) -> torch.Tensor:
        row_voltages = self.tia_voltages(x)
        return torch.einsum("bik,oik->bo", row_voltages, self.g_pos - self.g_neg)

    def effective_coefficients(self) -> torch.Tensor:
        return self.current_to_voltage_gain * self.cell_config.row_voltage_scale_v * (self.g_pos - self.g_neg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.current_to_voltage_gain.to(x.device, x.dtype) * self.edge_currents(x)
        output = output + self.bias.to(x.device, x.dtype)
        clip = self.forward_config.adc_clip_voltage
        if clip is not None:
            output = output.clamp(-clip, clip)
        return output

    def count_gbf_cells(self) -> int:
        return self.in_features * self.num_basis

    def count_tias(self) -> int:
        return self.count_gbf_cells()

    def count_memristors(self) -> int:
        return self.g_pos.numel() + self.g_neg.numel()


class PhysicalGeneralizedBellKAN(nn.Module):
    """Stack of GBF-cell/TIA/crossbar KAN layers."""

    def __init__(self, layers: Iterable[PhysicalGeneralizedBellKANLayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(list(layers))
        if not self.layers:
            raise ValueError("PhysicalGeneralizedBellKAN requires at least one layer")

    @classmethod
    def from_software_model(
        cls,
        model: GeneralizedBellKAN,
        mapper: RRAMWeightMapper | None = None,
        cell_config: GBFCellConfig | None = None,
        current_to_voltage_gain: float | None = None,
        forward_config: GBFForwardConfig | None = None,
    ) -> "PhysicalGeneralizedBellKAN":
        layers: list[PhysicalGeneralizedBellKANLayer] = []
        for software_layer in model.layers:
            scaling = GBFCrossbarScaling(current_to_voltage_gain=current_to_voltage_gain)
            layers.append(
                PhysicalGeneralizedBellKANLayer.from_software_layer(
                    software_layer,
                    mapper=mapper,
                    cell_config=cell_config,
                    scaling=scaling,
                    forward_config=forward_config,
                )
            )
        return cls(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = apply_inter_layer_normalization(
                    x,
                    mode=layer.forward_config.inter_layer_normalization,
                    gain=layer.forward_config.normalization_gain,
                )
        return x

    def count_gbf_cells(self) -> int:
        return sum(layer.count_gbf_cells() for layer in self.layers)

    def count_tias(self) -> int:
        return sum(layer.count_tias() for layer in self.layers)

    def count_memristors(self) -> int:
        return sum(layer.count_memristors() for layer in self.layers)

    def estimate_frontend_power_w(self) -> float:
        if not self.layers:
            return 0.0
        return sum(
            layer.count_gbf_cells() * (layer.cell_config.cell_power_w + layer.cell_config.tia_power_w)
            for layer in self.layers
        )
