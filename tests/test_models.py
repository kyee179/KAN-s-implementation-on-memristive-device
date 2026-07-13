import torch

from kan_memristor.models import RBFKAN, count_parameters, make_mlp


def test_rbf_kan_forward_shape():
    model = RBFKAN([2, 4, 1], num_basis=5)
    output = model(torch.zeros(3, 2))
    assert output.shape == (3, 1)
    assert count_parameters(model) > 0


def test_mlp_forward_shape():
    model = make_mlp([2, 8, 1])
    output = model(torch.zeros(3, 2))
    assert output.shape == (3, 1)