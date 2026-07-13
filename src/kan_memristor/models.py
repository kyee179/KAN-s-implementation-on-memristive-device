"""Small transparent KAN and MLP models for baseline experiments."""

from __future__ import annotations

import torch
from torch import nn


class RBFKANLayer(nn.Module):
    """KAN layer with one learned univariate RBF expansion on every edge."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_basis: int = 11,
        grid_range: tuple[float, float] = (-1.5, 1.5),
    ) -> None:
        super().__init__()
        centers = torch.linspace(grid_range[0], grid_range[1], num_basis)
        spacing = (grid_range[1] - grid_range[0]) / max(num_basis - 1, 1)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / (spacing**2 + 1e-8)
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, num_basis))
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.08)
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = torch.exp(-self.gamma * (x.unsqueeze(-1) - self.centers) ** 2)
        nonlinear = torch.einsum("bik,oik->bo", basis, self.spline_weight)
        linear = torch.einsum("bi,oi->bo", x, self.base_weight)
        return nonlinear + linear + self.bias


class RBFKAN(nn.Module):
    """A compact KAN composed of edge-function layers."""

    def __init__(self, widths: list[int], num_basis: int = 11) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths must contain at least input and output sizes")
        self.layers = nn.ModuleList(
            [RBFKANLayer(widths[i], widths[i + 1], num_basis=num_basis) for i in range(len(widths) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def make_mlp(widths: list[int]) -> nn.Sequential:
    """Create a SiLU MLP baseline with the same input/output dimensions."""

    modules: list[nn.Module] = []
    for in_features, out_features in zip(widths[:-2], widths[1:-1]):
        modules.append(nn.Linear(in_features, out_features))
        modules.append(nn.SiLU())
    modules.append(nn.Linear(widths[-2], widths[-1]))
    return nn.Sequential(*modules)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)