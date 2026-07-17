"""Behavioral hardware models for physical KAN studies."""

from kan_memristor.hardware.gilbert_multiplier import GilbertMultiplier, GilbertMultiplierParameters
from kan_memristor.hardware.memristor import DynamicMemdiode, DynamicMemdiodeParameters, RRAMWeightMapper
from kan_memristor.hardware.odd_polynomial_edge import HardwareOddPolynomialEdge
from kan_memristor.hardware.physical_kan import (
    DynamicPulseConfig,
    HardwareKANScaling,
    MemristivePulseOptimizer,
    PhysicalForwardConfig,
    PhysicalOddPolynomialKAN,
    PhysicalOddPolynomialKANLayer,
    PulseUpdateConfig,
)

__all__ = [
    "DynamicMemdiode",
    "DynamicMemdiodeParameters",
    "DynamicPulseConfig",
    "GilbertMultiplier",
    "GilbertMultiplierParameters",
    "HardwareKANScaling",
    "HardwareOddPolynomialEdge",
    "MemristivePulseOptimizer",
    "PhysicalForwardConfig",
    "PhysicalOddPolynomialKAN",
    "PhysicalOddPolynomialKANLayer",
    "PulseUpdateConfig",
    "RRAMWeightMapper",
]
