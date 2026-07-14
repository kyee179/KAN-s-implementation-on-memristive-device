import torch

from kan_memristor.models import (
    BSplineKAN,
    BSplineKANLayer,
    OddPolynomialKAN,
    OddPolynomialKANLayer,
    count_parameters,
    make_mlp,
)


def test_b_spline_layer_basis_shape_and_partition():
    layer = BSplineKANLayer(2, 4, num_basis=6, degree=3)
    basis = layer._basis(torch.zeros(3, 2))
    assert basis.shape == (3, 2, 6)
    assert torch.allclose(basis.sum(dim=-1), torch.ones(3, 2), atol=1e-5)


def test_b_spline_kan_forward_shape():
    model = BSplineKAN([2, 4, 1], num_basis=6)
    output = model(torch.zeros(3, 2))
    assert output.shape == (3, 1)
    assert count_parameters(model) > 0


def test_odd_polynomial_layer_basis_is_odd():
    layer = OddPolynomialKANLayer(2, 4)
    x = torch.tensor([[0.25, -0.5], [1.0, -1.0]])
    assert torch.allclose(layer._basis(-x), -layer._basis(x))


def test_odd_polynomial_kan_forward_shape():
    model = OddPolynomialKAN([2, 4, 1])
    output = model(torch.zeros(3, 2))
    assert output.shape == (3, 1)
    assert count_parameters(model) > 0


def test_mlp_forward_shape():
    model = make_mlp([2, 8, 1])
    output = model(torch.zeros(3, 2))
    assert output.shape == (3, 1)