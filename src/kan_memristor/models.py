"""Small transparent KAN and MLP models for baseline experiments."""

from __future__ import annotations

import torch
from torch import nn


class BSplineKANLayer(nn.Module):
    """KAN layer with one learned univariate B-spline on every edge."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_basis: int = 11,
        degree: int = 3,
        grid_range: tuple[float, float] = (-1.5, 1.5),
    ) -> None:
        super().__init__()
        if degree < 0:
            raise ValueError("degree must be non-negative")
        if num_basis <= degree:
            raise ValueError("num_basis must be greater than degree")
        self.in_features = in_features
        self.out_features = out_features
        self.num_basis = num_basis
        self.degree = degree
        self.grid_range = grid_range
        self.register_buffer("knots", self._make_open_uniform_knots(num_basis, degree, grid_range))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, num_basis))
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    @staticmethod
    def _make_open_uniform_knots(
        num_basis: int,
        degree: int,
        grid_range: tuple[float, float],
    ) -> torch.Tensor:
        low, high = grid_range
        interior_count = num_basis - degree - 1
        if interior_count > 0:
            interior = torch.linspace(low, high, interior_count + 2)[1:-1]
            return torch.cat(
                [
                    torch.full((degree + 1,), low),
                    interior,
                    torch.full((degree + 1,), high),
                ]
            )
        return torch.cat([torch.full((degree + 1,), low), torch.full((degree + 1,), high)])

    def reset_parameters(self) -> None:
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.05)
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.zeros_(self.bias)

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        low, high = self.grid_range
        eps = torch.finfo(x.dtype).eps
        x = x.clamp(min=low + eps, max=high - eps).unsqueeze(-1)
        knots = self.knots.to(dtype=x.dtype, device=x.device)
        basis = ((x >= knots[:-1]) & (x < knots[1:])).to(dtype=x.dtype)

        for order in range(1, self.degree + 1):
            new_len = basis.shape[-1] - 1
            left_den = knots[order : order + new_len] - knots[:new_len]
            right_den = knots[order + 1 : order + 1 + new_len] - knots[1 : 1 + new_len]
            left = torch.zeros_like(basis[..., :new_len])
            right = torch.zeros_like(basis[..., :new_len])
            left_mask = left_den > 0
            right_mask = right_den > 0
            if left_mask.any():
                left[..., left_mask] = (
                    (x - knots[:new_len][left_mask]) / left_den[left_mask]
                ) * basis[..., :new_len][..., left_mask]
            if right_mask.any():
                right[..., right_mask] = (
                    (knots[order + 1 : order + 1 + new_len][right_mask] - x) / right_den[right_mask]
                ) * basis[..., 1 : new_len + 1][..., right_mask]
            basis = left + right
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = self._basis(x)
        nonlinear = torch.einsum("bik,oik->bo", basis, self.spline_weight)
        linear = torch.einsum("bi,oi->bo", x, self.base_weight)
        return nonlinear + linear + self.bias


class BSplineKAN(nn.Module):
    """A compact KAN composed of B-spline edge-function layers."""

    def __init__(self, widths: list[int], num_basis: int = 11, degree: int = 3) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths must contain at least input and output sizes")
        self.layers = nn.ModuleList(
            [
                BSplineKANLayer(
                    widths[i],
                    widths[i + 1],
                    num_basis=num_basis,
                    degree=degree,
                )
                for i in range(len(widths) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class OddPolynomialKANLayer(nn.Module):
    """KAN layer whose edge functions are c1*x + c3*x^3 + c5*x^5."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        powers: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        super().__init__()
        if not powers:
            raise ValueError("powers must not be empty")
        if any(power <= 0 or power % 2 == 0 for power in powers):
            raise ValueError("powers must be positive odd integers")
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("powers", torch.tensor(powers, dtype=torch.float32))
        self.coefficients = nn.Parameter(torch.empty(out_features, in_features, len(powers)))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.coefficients.zero_()
            self.coefficients[..., 0].normal_(mean=0.0, std=0.35)
            if self.coefficients.shape[-1] > 1:
                self.coefficients[..., 1:].normal_(mean=0.0, std=0.03)
        nn.init.zeros_(self.bias)

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        powers = self.powers.to(dtype=x.dtype, device=x.device)
        return x.unsqueeze(-1).pow(powers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = self._basis(x)
        edge_values = torch.einsum("bik,oik->bo", basis, self.coefficients)
        return edge_values + self.bias


class OddPolynomialKAN(nn.Module):
    """A compact KAN using odd-polynomial edge functions only."""

    def __init__(self, widths: list[int], powers: tuple[int, ...] = (1, 3, 5)) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths must contain at least input and output sizes")
        self.layers = nn.ModuleList(
            [OddPolynomialKANLayer(widths[i], widths[i + 1], powers=powers) for i in range(len(widths) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# Backwards-compatible aliases for older local notebooks or scripts.
KANLayer = BSplineKANLayer
KAN = BSplineKAN


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