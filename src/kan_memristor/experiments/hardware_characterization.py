"""Characterize hardware blocks used by the physical KAN simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kan_memristor.hardware.gilbert_multiplier import GilbertMultiplier
from kan_memristor.hardware.memristor import DynamicMemdiode, RRAMWeightMapper
from kan_memristor.hardware.odd_polynomial_edge import HardwareOddPolynomialEdge


def characterize_memdiode(output_dir: Path) -> dict[str, float]:
    device = DynamicMemdiode()
    voltages = np.linspace(-1.5, 1.5, 401)
    states = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    currents = np.stack([device.current(voltages, state=state) for state in states])

    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for state, current in zip(states, currents):
        axis.semilogy(voltages[voltages >= 0], np.maximum(current[voltages >= 0], 1e-15), label=f"lambda={state:.2f}")
    axis.set_xlabel("voltage (V)")
    axis.set_ylabel("current magnitude (A)")
    axis.set_title("Memdiode positive-bias I-V")
    axis.legend()
    fig.savefig(output_dir / "memdiode_iv.png", dpi=160)
    plt.close(fig)

    pulse_device = DynamicMemdiode(initial_state=0.1)
    pulse_schedule = [(1.8, 1e-9)] * 120 + [(-1.4, 1e-9)] * 120
    state_trace = []
    for voltage, dt in pulse_schedule:
        state_trace.append(pulse_device.step(float(voltage), dt=dt))
    state_trace_array = np.asarray(state_trace)

    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.plot(state_trace_array)
    axis.set_xlabel("pulse index")
    axis.set_ylabel("lambda")
    axis.set_title("Memdiode SET/RESET state evolution")
    fig.savefig(output_dir / "memdiode_state_trace.png", dpi=160)
    plt.close(fig)

    return {
        "current_at_0p2v_hrs_a": float(device.current(0.2, state=0.0)),
        "current_at_0p2v_lrs_a": float(device.current(0.2, state=1.0)),
        "conductance_ratio_lrs_over_hrs": float(device.conductance(0.2, state=1.0) / device.conductance(0.2, state=0.0)),
        "state_after_set_reset_pulses": float(state_trace_array[-1]),
        "state_peak_during_pulses": float(state_trace_array.max()),
    }


def characterize_gilbert(output_dir: Path) -> dict[str, float]:
    multiplier = GilbertMultiplier()
    grid = np.linspace(-0.4, 0.4, 101)
    vx, vy = np.meshgrid(grid, grid, indexing="ij")
    predicted = multiplier.multiply(vx, vy)
    ideal = vx * vy
    error = predicted - ideal

    fig, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    image = axis.imshow(error, extent=[-0.4, 0.4, -0.4, 0.4], origin="lower", aspect="auto", cmap="coolwarm")
    axis.set_xlabel("Vx (V)")
    axis.set_ylabel("Vy (V)")
    axis.set_title("Gilbert product error")
    fig.colorbar(image, ax=axis)
    fig.savefig(output_dir / "gilbert_product_error.png", dpi=160)
    plt.close(fig)

    return {
        "product_rmse_v": float(np.sqrt(np.mean(error**2))),
        "product_max_abs_error_v": float(np.max(np.abs(error))),
        "bandwidth_hz": float(multiplier.parameters.bandwidth_hz),
        "power_w": float(multiplier.parameters.power_w),
        "energy_per_multiply_at_bandwidth_j": float(multiplier.energy_for_operations(1)),
    }


def characterize_rram_mapper() -> dict[str, float]:
    mapper = RRAMWeightMapper(n_states=16, r_lrs=1e4, r_hrs=1e6)
    weights = np.linspace(-1.0, 1.0, 101)
    g_pos, g_neg, scale = mapper.signed_weight_to_differential_pair(weights, apply_nonidealities=False)
    recovered = mapper.differential_pair_to_weight(g_pos, g_neg, scale)
    return {
        "resistance_ratio_hrs_over_lrs": float(mapper.resistance_ratio),
        "num_conductance_states": float(mapper.n_states),
        "max_abs_quantization_error": float(np.max(np.abs(recovered - weights))),
        "mean_abs_quantization_error": float(np.mean(np.abs(recovered - weights))),
    }


def characterize_hardware_edge(output_dir: Path) -> dict[str, float]:
    edge = HardwareOddPolynomialEdge((1.0, 0.5, -0.25))
    x = np.linspace(-1.0, 1.0, 401)
    y = edge(x)
    odd_error = y + edge(-x)

    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.plot(x, y)
    axis.set_xlabel("normalized input")
    axis.set_ylabel("edge output (V-weight units)")
    axis.set_title("Hardware odd-polynomial edge")
    fig.savefig(output_dir / "hardware_odd_polynomial_edge.png", dpi=160)
    plt.close(fig)

    return {
        "max_abs_odd_symmetry_error": float(np.max(np.abs(odd_error))),
        "output_span": float(y.max() - y.min()),
    }


def run(output_dir: Path) -> dict[str, dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "memdiode": characterize_memdiode(output_dir),
        "gilbert_multiplier": characterize_gilbert(output_dir),
        "rram_weight_mapper": characterize_rram_mapper(),
        "hardware_odd_polynomial_edge": characterize_hardware_edge(output_dir),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/hardware_characterization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(Path(args.output_dir))


if __name__ == "__main__":
    main()