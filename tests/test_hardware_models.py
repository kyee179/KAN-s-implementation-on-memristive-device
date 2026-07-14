import numpy as np

from kan_memristor.hardware.gilbert_multiplier import GilbertMultiplier
from kan_memristor.hardware.memristor import DynamicMemdiode, RRAMWeightMapper
from kan_memristor.hardware.odd_polynomial_edge import HardwareOddPolynomialEdge


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