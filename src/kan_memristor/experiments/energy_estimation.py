"""Estimate physical KAN energy and compare it with a digital MLP baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from kan_memristor.datasets import SupervisedDataset, load_dataset
from kan_memristor.experiments.baseline_kan import default_widths
from kan_memristor.experiments.hardware_train import (
    HardwareTrainConfig,
    _evaluate,
    _loss_for_task,
    _make_dynamic_pulse_config,
    _make_forward_config,
    _pulse_train,
    _train_software_model,
)
from kan_memristor.hardware.energy import (
    estimate_dynamic_pulse_energy,
    estimate_mlp_inference_energy,
    estimate_physical_kan_inference_energy,
)
from kan_memristor.hardware.memristor import RRAMWeightMapper
from kan_memristor.hardware.physical_kan import PhysicalOddPolynomialKAN


@dataclass(frozen=True)
class EnergyExperimentResult:
    dataset: str
    task: str
    physical_test_loss: float
    physical_test_mse: float | None
    physical_test_accuracy: float | None
    kan_energy: dict
    mlp_energy: dict
    programming_energy: dict
    energy_ratio_kan_vs_mlp: float
    config: dict


def _config_for_dataset(dataset_name: str, args: argparse.Namespace) -> HardwareTrainConfig:
    return HardwareTrainConfig(
        dataset=dataset_name,
        widths=default_widths(dataset_name, "odd_poly_kan"),
        powers=tuple(args.powers),
        k=args.k,
        n_states=args.n_states,
        r_lrs=args.r_lrs,
        r_hrs=args.r_hrs,
        pretrain_epochs=args.pretrain_epochs,
        pulse_epochs=args.pulse_epochs,
        batch_size=args.batch_size,
        pretrain_learning_rate=args.pretrain_learning_rate,
        bias_learning_rate=args.bias_learning_rate,
        conductance_learning_rate=args.conductance_learning_rate,
        max_pulses_per_update=args.max_pulses_per_update,
        gradient_deadzone_quantile=args.gradient_deadzone_quantile,
        seed=args.seed,
        complete_physical=True,
        rram_iv_nonlinearity=args.rram_iv_nonlinearity,
        variability_std=args.variability_std,
        stuck_lrs_probability=args.stuck_lrs_probability,
        stuck_hrs_probability=args.stuck_hrs_probability,
        gilbert_product_gain=args.gilbert_product_gain,
        gilbert_input_linear_range=args.gilbert_input_linear_range,
        dynamic_set_voltage=args.dynamic_set_voltage,
        dynamic_reset_voltage=args.dynamic_reset_voltage,
        dynamic_pulse_width_s=args.dynamic_pulse_width_s,
        inter_layer_normalization=args.inter_layer_normalization,
        normalization_gain=args.normalization_gain,
    )


def run_dataset(dataset_name: str, args: argparse.Namespace) -> EnergyExperimentResult:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = load_dataset(dataset_name, n_train=args.n_train, n_test=args.n_test, seed=args.seed)
    config = _config_for_dataset(dataset_name, args)
    software_model, _ = _train_software_model(dataset, config)
    mapper = RRAMWeightMapper(
        r_lrs=args.r_lrs,
        r_hrs=args.r_hrs,
        n_states=args.n_states,
        variability_std=args.variability_std,
        stuck_lrs_probability=args.stuck_lrs_probability,
        stuck_hrs_probability=args.stuck_hrs_probability,
        seed=args.seed,
    )
    shim = argparse.Namespace(
        **vars(args),
        complete_physical=True,
        use_gilbert_multiplier=False,
        dynamic_memdiode_pulses=True,
    )
    physical_model = PhysicalOddPolynomialKAN.from_software_model(
        software_model,
        mapper=mapper,
        input_scale_v=args.k,
        current_to_voltage_gain=args.current_to_voltage_gain,
        forward_config=_make_forward_config(shim, args.k),
        dynamic_pulse_config=_make_dynamic_pulse_config(shim),
    )

    set_pulses = args.set_pulses
    reset_pulses = args.reset_pulses
    if args.pulse_epochs > 0:
        _, set_pulses, reset_pulses = _pulse_train(physical_model, dataset, config)

    loss_fn = _loss_for_task(dataset.task)
    test_loss, test_mse, test_accuracy, _ = _evaluate(physical_model, dataset, loss_fn, split="test")
    x_energy = torch.from_numpy(dataset.x_test[: min(args.energy_samples, len(dataset.x_test))])
    kan_energy = estimate_physical_kan_inference_energy(
        physical_model,
        x_energy,
        read_time_s=args.read_time_s,
        peripheral_j_per_sample=args.peripheral_energy_j,
    )
    mlp_widths = default_widths(dataset_name, "mlp")
    mlp_energy = estimate_mlp_inference_energy(mlp_widths, energy_per_mac_j=args.energy_per_mac_j)
    programming_energy = estimate_dynamic_pulse_energy(
        _make_dynamic_pulse_config(shim),
        set_pulses=set_pulses,
        reset_pulses=reset_pulses,
    )
    return EnergyExperimentResult(
        dataset=dataset.name,
        task=dataset.task,
        physical_test_loss=test_loss,
        physical_test_mse=test_mse,
        physical_test_accuracy=test_accuracy,
        kan_energy=kan_energy.to_dict(),
        mlp_energy=mlp_energy.to_dict(),
        programming_energy=programming_energy.to_dict(),
        energy_ratio_kan_vs_mlp=kan_energy.total_j_per_sample / mlp_energy.inference_j_per_sample,
        config=asdict(config) | {
            "read_time_s": args.read_time_s,
            "energy_per_mac_j": args.energy_per_mac_j,
            "peripheral_energy_j": args.peripheral_energy_j,
            "energy_samples": args.energy_samples,
            "mlp_widths": mlp_widths,
        },
    )


def run_suite(args: argparse.Namespace) -> list[EnergyExperimentResult]:
    results = [run_dataset(dataset_name, args) for dataset_name in args.datasets]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(result) for result in results]
    (output_dir / "energy_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for result in payload:
        print(json.dumps(result, indent=2))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["complicated_function", "taglietti_yinyang"])
    parser.add_argument("--k", type=float, default=0.2)
    parser.add_argument("--powers", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--energy-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pretrain-epochs", type=int, default=150)
    parser.add_argument("--pulse-epochs", type=int, default=0)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-3)
    parser.add_argument("--bias-learning-rate", type=float, default=5e-4)
    parser.add_argument("--conductance-learning-rate", type=float, default=1e-11)
    parser.add_argument("--max-pulses-per-update", type=int, default=1)
    parser.add_argument("--gradient-deadzone-quantile", type=float, default=0.9)
    parser.add_argument("--n-states", type=int, default=64)
    parser.add_argument("--r-lrs", type=float, default=1e4)
    parser.add_argument("--r-hrs", type=float, default=1e6)
    parser.add_argument("--current-to-voltage-gain", type=float, default=None)
    parser.add_argument("--gilbert-product-gain", type=float, default=None)
    parser.add_argument("--gilbert-input-linear-range", type=float, default=0.4)
    parser.add_argument("--dynamic-set-voltage", type=float, default=1.8)
    parser.add_argument("--dynamic-reset-voltage", type=float, default=-1.0)
    parser.add_argument("--dynamic-pulse-width-s", type=float, default=1e-9)
    parser.add_argument("--rram-iv-nonlinearity", type=float, default=0.0)
    parser.add_argument("--variability-std", type=float, default=0.0)
    parser.add_argument("--stuck-lrs-probability", type=float, default=0.0)
    parser.add_argument("--stuck-hrs-probability", type=float, default=0.0)
    parser.add_argument("--adc-clip-voltage", type=float, default=None)
    parser.add_argument("--inter-layer-normalization", choices=["none", "tanh", "clip"], default="tanh")
    parser.add_argument("--normalization-gain", type=float, default=2.0)
    parser.add_argument("--read-time-s", type=float, default=1e-9)
    parser.add_argument("--energy-per-mac-j", type=float, default=4.6e-12)
    parser.add_argument("--peripheral-energy-j", type=float, default=0.0)
    parser.add_argument("--set-pulses", type=int, default=0)
    parser.add_argument("--reset-pulses", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/energy_estimation")
    return parser.parse_args()


def main() -> None:
    run_suite(parse_args())


if __name__ == "__main__":
    main()
