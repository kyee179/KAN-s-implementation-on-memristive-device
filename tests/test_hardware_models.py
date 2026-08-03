import numpy as np
import torch

from kan_memristor.hardware.gilbert_multiplier import GilbertMultiplier
from kan_memristor.hardware.memristor import DynamicMemdiode, RRAMWeightMapper
from kan_memristor.hardware.odd_polynomial_edge import HardwareOddPolynomialEdge
from kan_memristor.hardware.gilbert_multiplier import GilbertMultiplierParameters
from kan_memristor.hardware.physical_kan import (
    DynamicPulseConfig,
    HardwareKANScaling,
    MemristivePulseOptimizer,
    PhysicalForwardConfig,
    PhysicalOddPolynomialKAN,
    PhysicalOddPolynomialKANLayer,
    PulseUpdateConfig,
)
from kan_memristor.models import OddPolynomialKAN, OddPolynomialKANLayer


def test_memdiode_current_increases_with_state():
    device = DynamicMemdiode()
    low_state_current = device.current(0.2, state=0.0)
    high_state_current = device.current(0.2, state=1.0)
    assert high_state_current > low_state_current
    assert abs(float(device.current(0.0, state=0.5))) < 1e-15


def test_memdiode_state_moves_under_set_and_reset_pulses():
    device = DynamicMemdiode(initial_state=0.2)
    for _ in range(20):
        device.step(1.8, dt=1e-9)
    after_set = device.state
    for _ in range(20):
        device.step(-1.0, dt=1e-9)
    assert after_set > 0.2
    assert device.state < after_set


def test_rram_signed_weight_mapping_is_monotonic():
    mapper = RRAMWeightMapper(n_states=8, r_lrs=1e4, r_hrs=1e6)
    weights = np.array([-1.0, -0.2, 0.0, 0.8])
    g_pos, g_neg, scale = mapper.signed_weight_to_differential_pair(weights, apply_nonidealities=False)
    recovered = mapper.differential_pair_to_weight(g_pos, g_neg, scale)
    assert mapper.resistance_ratio == 100.0
    assert recovered[0] < recovered[1] <= recovered[2] < recovered[3]


def test_gilbert_multiplier_product_and_energy():
    multiplier = GilbertMultiplier()
    product = multiplier.multiply(np.array([0.1, -0.1]), 0.2)
    assert product[0] > 0.0
    assert product[1] < 0.0
    assert multiplier.energy_for_operations(10) > 0.0


def test_hardware_odd_polynomial_edge_is_odd_symmetric():
    edge = HardwareOddPolynomialEdge((1.0, 0.5, -0.25))
    x = np.array([-0.5, 0.0, 0.5])
    y = edge(x)
    assert np.allclose(y[0], -y[2], atol=1e-8)
    assert abs(y[1]) < 1e-12


def test_physical_kan_from_software_preserves_shape_and_counts_memristors():
    software = OddPolynomialKAN([2, 3, 1])
    mapper = RRAMWeightMapper(n_states=16, r_lrs=1e4, r_hrs=1e6)
    physical = PhysicalOddPolynomialKAN.from_software_model(software, mapper=mapper, input_scale_v=0.2)
    x = torch.randn(5, 2)
    y = physical(x)
    expected_memristors = 2 * ((3 * 2 * 3) + (1 * 3 * 3))
    assert y.shape == (5, 1)
    assert physical.count_memristors() == expected_memristors
    assert all(layer.current_to_voltage_gain.item() > 0.0 for layer in physical.layers)


def test_physical_layer_approximates_software_edge_after_mapping():
    software = OddPolynomialKANLayer(1, 1)
    with torch.no_grad():
        software.coefficients[:] = torch.tensor([[[0.25, -0.10, 0.05]]])
        software.bias.zero_()
    physical = PhysicalOddPolynomialKAN.from_software_model(
        OddPolynomialKAN([1, 1]),
        mapper=RRAMWeightMapper(n_states=128, r_lrs=1e4, r_hrs=1e6),
        input_scale_v=0.2,
    )
    physical.layers[0] = physical.layers[0].from_software_layer(
        software,
        mapper=RRAMWeightMapper(n_states=128, r_lrs=1e4, r_hrs=1e6),
    )
    x = torch.linspace(-0.8, 0.8, 9).unsqueeze(1)
    assert torch.max(torch.abs(software(x) - physical(x))).item() < 0.02


def test_memristive_pulse_optimizer_updates_conductance_states():
    software = OddPolynomialKAN([1, 1])
    mapper = RRAMWeightMapper(n_states=16, r_lrs=1e4, r_hrs=1e6)
    physical = PhysicalOddPolynomialKAN.from_software_model(software, mapper=mapper, input_scale_v=0.2)
    optimizer = MemristivePulseOptimizer(
        physical,
        PulseUpdateConfig(max_pulses_per_update=1, gradient_deadzone_quantile=0.0, conductance_learning_rate=1e-5),
    )
    x = torch.tensor([[-0.5], [0.5]])
    target = torch.tensor([[0.5], [-0.5]])
    loss = torch.nn.functional.mse_loss(physical(x), target)
    loss.backward()
    before = physical.layers[0].g_pos.detach().clone()
    pulse_counts = optimizer.step()
    after = physical.layers[0].g_pos.detach()
    assert pulse_counts["set_pulses"] + pulse_counts["reset_pulses"] > 0
    assert not torch.equal(before, after) or not torch.equal(physical.layers[0].g_neg.detach(), before)
    assert torch.all(after >= mapper.g_min)
    assert torch.all(after <= mapper.g_max)



def test_physical_layer_uses_gilbert_multiplier_for_power_rows():
    ideal_layer = PhysicalOddPolynomialKANLayer(1, 1)
    gilbert_layer = PhysicalOddPolynomialKANLayer(
        1,
        1,
        forward_config=PhysicalForwardConfig(
            use_gilbert_multiplier=True,
            gilbert_parameters=GilbertMultiplierParameters(product_gain=0.5, input_linear_range=0.2),
        ),
    )
    x = torch.tensor([[0.8]])
    ideal_rows = ideal_layer.voltage_rows(x)
    gilbert_rows = gilbert_layer.voltage_rows(x)
    assert ideal_rows.shape == gilbert_rows.shape == (1, 1, 3)
    assert torch.isclose(ideal_rows[..., 0], gilbert_rows[..., 0]).all()
    assert not torch.isclose(ideal_rows[..., 1], gilbert_rows[..., 1]).all()


def test_dynamic_memdiode_pulse_optimizer_updates_state_and_conductance():
    software = OddPolynomialKAN([1, 1])
    mapper = RRAMWeightMapper(n_states=32, r_lrs=1e4, r_hrs=1e6)
    physical = PhysicalOddPolynomialKAN.from_software_model(
        software,
        mapper=mapper,
        input_scale_v=0.2,
        dynamic_pulse_config=DynamicPulseConfig(enabled=True, set_voltage=1.8, reset_voltage=-1.0, pulse_width_s=1e-9),
    )
    optimizer = MemristivePulseOptimizer(
        physical,
        PulseUpdateConfig(
            max_pulses_per_update=2,
            gradient_deadzone_quantile=0.0,
            conductance_learning_rate=1e-5,
            dynamic_pulse_config=DynamicPulseConfig(enabled=True, set_voltage=1.8, reset_voltage=-1.0, pulse_width_s=1e-9),
            quantize_after_update=False,
        ),
    )
    x = torch.tensor([[-0.5], [0.5]])
    target = torch.tensor([[0.5], [-0.5]])
    before_state = torch.cat([physical.layers[0].state_pos.flatten(), physical.layers[0].state_neg.flatten()]).detach().clone()
    before_g = torch.cat([physical.layers[0].g_pos.flatten(), physical.layers[0].g_neg.flatten()]).detach().clone()
    loss = torch.nn.functional.mse_loss(physical(x), target)
    loss.backward()
    pulse_counts = optimizer.step()
    after_state = torch.cat([physical.layers[0].state_pos.flatten(), physical.layers[0].state_neg.flatten()]).detach()
    after_g = torch.cat([physical.layers[0].g_pos.flatten(), physical.layers[0].g_neg.flatten()]).detach()
    assert pulse_counts["set_pulses"] + pulse_counts["reset_pulses"] > 0
    assert not torch.equal(before_state, after_state)
    assert not torch.equal(before_g, after_g)






def test_physical_kan_inter_layer_tanh_changes_second_layer_drive():
    mapper = RRAMWeightMapper(n_states=16, r_lrs=1e4, r_hrs=1e6)
    scaling = HardwareKANScaling(input_scale_v=0.2, current_to_voltage_gain=1e6)
    layer1 = PhysicalOddPolynomialKANLayer(1, 1, mapper=mapper, scaling=scaling)
    layer2 = PhysicalOddPolynomialKANLayer(1, 1, mapper=mapper, scaling=scaling)
    for layer in (layer1, layer2):
        with torch.no_grad():
            layer.g_pos.fill_(mapper.g_max)
            layer.g_neg.fill_(mapper.g_min)
    no_norm = PhysicalOddPolynomialKAN([layer1, layer2])

    norm_config = PhysicalForwardConfig(inter_layer_normalization="tanh", normalization_gain=1.0)
    norm_layer1 = PhysicalOddPolynomialKANLayer(1, 1, mapper=mapper, scaling=scaling, forward_config=norm_config)
    norm_layer2 = PhysicalOddPolynomialKANLayer(1, 1, mapper=mapper, scaling=scaling, forward_config=norm_config)
    for layer in (norm_layer1, norm_layer2):
        with torch.no_grad():
            layer.g_pos.fill_(mapper.g_max)
            layer.g_neg.fill_(mapper.g_min)
    with_norm = PhysicalOddPolynomialKAN([norm_layer1, norm_layer2])

    x = torch.ones(1, 1)
    assert torch.abs(no_norm(x) - with_norm(x)).item() > 1.0


def test_physical_layer_generates_even_power_voltage_rows():
    layer = PhysicalOddPolynomialKANLayer(
        1,
        1,
        powers=(1, 2, 3, 4),
        forward_config=PhysicalForwardConfig(use_gilbert_multiplier=False),
    )
    rows = layer.voltage_rows(torch.tensor([[-0.5]]))
    assert rows.shape == (1, 1, 4)
    assert rows[..., 0].item() < 0.0
    assert rows[..., 1].item() > 0.0
    assert rows[..., 2].item() < 0.0
    assert rows[..., 3].item() > 0.0
