import torch

from kan_memristor.hardware.energy import (
    count_mlp_macs,
    estimate_dynamic_pulse_energy,
    estimate_mlp_inference_energy,
    estimate_physical_kan_inference_energy,
)
from kan_memristor.hardware.memristor import RRAMWeightMapper
from kan_memristor.hardware.physical_kan import DynamicPulseConfig, PhysicalForwardConfig, PhysicalOddPolynomialKAN
from kan_memristor.models import OddPolynomialKAN


def test_count_mlp_macs_matches_dense_layers():
    assert count_mlp_macs([2, 64, 64, 1]) == 4288


def test_mlp_energy_scales_with_mac_count():
    small = estimate_mlp_inference_energy([2, 4, 1], energy_per_mac_j=1e-12)
    large = estimate_mlp_inference_energy([2, 8, 1], energy_per_mac_j=1e-12)
    assert small.inference_j_per_sample == 12e-12
    assert large.inference_j_per_sample > small.inference_j_per_sample


def test_physical_kan_inference_energy_is_positive():
    software = OddPolynomialKAN([2, 3, 1], inter_layer_normalization="tanh", normalization_gain=2.0)
    physical = PhysicalOddPolynomialKAN.from_software_model(
        software,
        mapper=RRAMWeightMapper(n_states=16),
        input_scale_v=0.2,
        forward_config=PhysicalForwardConfig(use_gilbert_multiplier=True, inter_layer_normalization="tanh", normalization_gain=2.0),
    )
    energy = estimate_physical_kan_inference_energy(physical, torch.ones(4, 2), read_time_s=1e-9)
    assert energy.total_j_per_sample > 0.0
    assert energy.rram_read_j_per_sample > 0.0
    assert energy.gilbert_j_per_sample > 0.0
    assert energy.memristor_count == physical.count_memristors()


def test_dynamic_pulse_energy_uses_set_and_reset_counts():
    energy = estimate_dynamic_pulse_energy(DynamicPulseConfig(), set_pulses=3, reset_pulses=5)
    assert energy.set_pulse_j > 0.0
    assert energy.reset_pulse_j > 0.0
    assert energy.total_programming_j == energy.total_set_j + energy.total_reset_j
    assert energy.set_pulses == 3
    assert energy.reset_pulses == 5
