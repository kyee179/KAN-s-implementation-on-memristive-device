"""Train and evaluate memristive odd-polynomial KAN variants."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kan_memristor.datasets import SupervisedDataset, load_dataset
from kan_memristor.experiments.baseline_kan import default_widths
from kan_memristor.hardware.gilbert_multiplier import GilbertMultiplierParameters
from kan_memristor.hardware.memristor import DynamicMemdiodeParameters, RRAMWeightMapper
from kan_memristor.hardware.physical_kan import (
    DynamicPulseConfig,
    MemristivePulseOptimizer,
    PhysicalForwardConfig,
    PhysicalOddPolynomialKAN,
    PulseUpdateConfig,
)
from kan_memristor.models import OddPolynomialKAN, count_parameters


@dataclass(frozen=True)
class HardwareTrainConfig:
    dataset: str
    widths: list[int]
    k: float
    n_states: int
    r_lrs: float
    r_hrs: float
    pretrain_epochs: int
    pulse_epochs: int
    batch_size: int
    pretrain_learning_rate: float
    bias_learning_rate: float
    conductance_learning_rate: float
    max_pulses_per_update: int
    gradient_deadzone_quantile: float
    seed: int
    complete_physical: bool
    rram_iv_nonlinearity: float
    variability_std: float
    stuck_lrs_probability: float
    stuck_hrs_probability: float
    gilbert_product_gain: float | None
    gilbert_input_linear_range: float
    dynamic_set_voltage: float
    dynamic_reset_voltage: float
    dynamic_pulse_width_s: float


@dataclass(frozen=True)
class HardwareStageResult:
    dataset: str
    stage: str
    task: str
    k: float
    train_loss: float
    test_loss: float
    test_mse: float | None
    test_accuracy: float | None
    parameter_count: int
    memristor_count: int | None
    gilbert_multiplier_count: int | None
    set_pulses: int
    reset_pulses: int
    current_to_voltage_gains: list[float] | None
    config: dict


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loss_for_task(task: str) -> nn.Module:
    if task == "regression":
        return nn.MSELoss()
    if task == "classification":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown task: {task}")


def _evaluate(
    model: nn.Module,
    dataset: SupervisedDataset,
    loss_fn: nn.Module,
    split: str = "test",
) -> tuple[float, float | None, float | None, np.ndarray]:
    model.eval()
    if split == "train":
        x_np = dataset.x_train
        y_np = dataset.y_train
    else:
        x_np = dataset.x_test
        y_np = dataset.y_test
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    with torch.no_grad():
        logits = model(x)
        loss = float(loss_fn(logits, y).item())
        predictions = logits.detach().cpu().numpy()
    if dataset.task == "regression":
        mse = float(np.mean((predictions - y_np) ** 2))
        return loss, mse, None, predictions
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(predictions, -60.0, 60.0)))
    classes = (probabilities >= 0.5).astype(np.float32)
    accuracy = float(np.mean(classes == y_np))
    return loss, None, accuracy, probabilities


def _train_software_model(dataset: SupervisedDataset, config: HardwareTrainConfig) -> tuple[OddPolynomialKAN, float]:
    _set_seed(config.seed)
    model = OddPolynomialKAN(config.widths)
    loss_fn = _loss_for_task(dataset.task)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.pretrain_learning_rate, weight_decay=1e-4)
    train_data = TensorDataset(torch.from_numpy(dataset.x_train), torch.from_numpy(dataset.y_train))
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, generator=generator)
    last_loss = 0.0
    model.train()
    for _ in range(config.pretrain_epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
    return model, last_loss


def _make_forward_config(args: argparse.Namespace, k: float) -> PhysicalForwardConfig:
    product_gain = args.gilbert_product_gain
    if product_gain is None and (args.complete_physical or args.use_gilbert_multiplier):
        product_gain = 1.0 / k
    if product_gain is None:
        product_gain = 1.0
    return PhysicalForwardConfig(
        use_gilbert_multiplier=args.complete_physical or args.use_gilbert_multiplier,
        gilbert_parameters=GilbertMultiplierParameters(
            input_linear_range=args.gilbert_input_linear_range,
            product_gain=product_gain,
        ),
        rram_iv_nonlinearity=args.rram_iv_nonlinearity,
        adc_clip_voltage=args.adc_clip_voltage,
    )


def _make_dynamic_pulse_config(args: argparse.Namespace) -> DynamicPulseConfig:
    return DynamicPulseConfig(
        enabled=args.complete_physical or args.dynamic_memdiode_pulses,
        set_voltage=args.dynamic_set_voltage,
        reset_voltage=args.dynamic_reset_voltage,
        pulse_width_s=args.dynamic_pulse_width_s,
        memdiode_parameters=DynamicMemdiodeParameters(),
    )


def _pulse_train(
    model: PhysicalOddPolynomialKAN,
    dataset: SupervisedDataset,
    config: HardwareTrainConfig,
) -> tuple[float, int, int]:
    loss_fn = _loss_for_task(dataset.task)
    train_data = TensorDataset(torch.from_numpy(dataset.x_train), torch.from_numpy(dataset.y_train))
    generator = torch.Generator().manual_seed(config.seed + 17)
    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, generator=generator)
    pulse_optimizer = MemristivePulseOptimizer(
        model,
        PulseUpdateConfig(
            max_pulses_per_update=config.max_pulses_per_update,
            gradient_deadzone_quantile=config.gradient_deadzone_quantile,
            bias_learning_rate=config.bias_learning_rate,
            conductance_learning_rate=config.conductance_learning_rate,
            dynamic_pulse_config=DynamicPulseConfig(
                enabled=config.complete_physical,
                set_voltage=config.dynamic_set_voltage,
                reset_voltage=config.dynamic_reset_voltage,
                pulse_width_s=config.dynamic_pulse_width_s,
            ),
        ),
    )
    last_loss = 0.0
    model.train()
    for _ in range(config.pulse_epochs):
        for x_batch, y_batch in loader:
            pulse_optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            pulse_optimizer.step()
            last_loss = float(loss.item())
    return last_loss, pulse_optimizer.total_set_pulses, pulse_optimizer.total_reset_pulses


def _stage_result(
    dataset: SupervisedDataset,
    stage: str,
    model: nn.Module,
    config: HardwareTrainConfig,
    train_loss: float,
    set_pulses: int = 0,
    reset_pulses: int = 0,
) -> tuple[HardwareStageResult, np.ndarray]:
    loss_fn = _loss_for_task(dataset.task)
    test_loss, test_mse, test_accuracy, predictions = _evaluate(model, dataset, loss_fn, split="test")
    if isinstance(model, PhysicalOddPolynomialKAN):
        memristor_count: int | None = model.count_memristors()
        gilbert_count: int | None = model.count_gilbert_multipliers()
        gains: list[float] | None = [float(layer.current_to_voltage_gain.item()) for layer in model.layers]
    else:
        memristor_count = None
        gilbert_count = None
        gains = None
    result = HardwareStageResult(
        dataset=dataset.name,
        stage=stage,
        task=dataset.task,
        k=config.k,
        train_loss=train_loss,
        test_loss=test_loss,
        test_mse=test_mse,
        test_accuracy=test_accuracy,
        parameter_count=count_parameters(model),
        memristor_count=memristor_count,
        gilbert_multiplier_count=gilbert_count,
        set_pulses=set_pulses,
        reset_pulses=reset_pulses,
        current_to_voltage_gains=gains,
        config=asdict(config),
    )
    return result, predictions


def _plot_predictions(dataset: SupervisedDataset, predictions: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    if dataset.task == "regression":
        target = dataset.y_test[:, 0]
        pred = predictions[:, 0]
        sc0 = axes[0].scatter(dataset.x_test[:, 0], dataset.x_test[:, 1], c=target, s=8, cmap="viridis")
        sc1 = axes[1].scatter(dataset.x_test[:, 0], dataset.x_test[:, 1], c=pred, s=8, cmap="viridis")
        axes[0].set_title("target")
        axes[1].set_title("hardware prediction")
        fig.colorbar(sc0, ax=axes[0], fraction=0.046)
        fig.colorbar(sc1, ax=axes[1], fraction=0.046)
    else:
        target = dataset.y_test[:, 0]
        pred = predictions[:, 0]
        axes[0].scatter(dataset.x_test[:, 0], dataset.x_test[:, 1], c=target, s=8, cmap="coolwarm", vmin=0, vmax=1)
        axes[1].scatter(dataset.x_test[:, 0], dataset.x_test[:, 1], c=pred, s=8, cmap="coolwarm", vmin=0, vmax=1)
        axes[0].set_title("target class")
        axes[1].set_title("hardware probability")
    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_dataset(dataset_name: str, k: float, args: argparse.Namespace) -> list[HardwareStageResult]:
    dataset = load_dataset(dataset_name, n_train=args.n_train, n_test=args.n_test, seed=args.seed)
    config = HardwareTrainConfig(
        dataset=dataset_name,
        widths=default_widths(dataset_name, "odd_poly_kan"),
        k=k,
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
        complete_physical=args.complete_physical,
        rram_iv_nonlinearity=args.rram_iv_nonlinearity,
        variability_std=args.variability_std,
        stuck_lrs_probability=args.stuck_lrs_probability,
        stuck_hrs_probability=args.stuck_hrs_probability,
        gilbert_product_gain=args.gilbert_product_gain,
        gilbert_input_linear_range=args.gilbert_input_linear_range,
        dynamic_set_voltage=args.dynamic_set_voltage,
        dynamic_reset_voltage=args.dynamic_reset_voltage,
        dynamic_pulse_width_s=args.dynamic_pulse_width_s,
    )
    mapper = RRAMWeightMapper(
        r_lrs=args.r_lrs,
        r_hrs=args.r_hrs,
        n_states=args.n_states,
        variability_std=args.variability_std,
        stuck_lrs_probability=args.stuck_lrs_probability,
        stuck_hrs_probability=args.stuck_hrs_probability,
        seed=args.seed,
    )
    forward_config = _make_forward_config(args, k)
    dynamic_pulse_config = _make_dynamic_pulse_config(args)
    output_dir = Path(args.output_dir)
    results: list[HardwareStageResult] = []
    mode_tag = "complete_physical" if args.complete_physical else "mapped"
    checkpoint_tag = f"{dataset_name}_k{k:g}_{mode_tag}"

    software_model, pretrain_loss = _train_software_model(dataset, config)
    pretrain_result, _ = _stage_result(dataset, "ideal_pretrain", software_model, config, pretrain_loss)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": software_model.state_dict(), "result": asdict(pretrain_result)},
        output_dir / f"{checkpoint_tag}_ideal_pretrain.pt",
    )
    results.append(pretrain_result)
    print(json.dumps(asdict(pretrain_result), indent=2))

    physical_model = PhysicalOddPolynomialKAN.from_software_model(
        software_model,
        mapper=mapper,
        input_scale_v=k,
        current_to_voltage_gain=args.current_to_voltage_gain,
        forward_config=forward_config,
        dynamic_pulse_config=dynamic_pulse_config,
    )
    mapped_train_loss, _, _, _ = _evaluate(physical_model, dataset, _loss_for_task(dataset.task), split="train")
    mapped_stage = "complete_physical_mapped" if args.complete_physical else "mapped_quantized"
    mapped_result, mapped_predictions = _stage_result(dataset, mapped_stage, physical_model, config, mapped_train_loss)
    results.append(mapped_result)
    print(json.dumps(asdict(mapped_result), indent=2))
    _plot_predictions(
        dataset,
        mapped_predictions,
        output_dir / f"{checkpoint_tag}_{mapped_stage}.png",
        f"{dataset_name} / {mapped_stage} / k={k:g}",
    )

    pulse_loss, set_pulses, reset_pulses = _pulse_train(physical_model, dataset, config)
    pulse_stage = "complete_physical_pulse_trained" if args.complete_physical else "pulse_trained"
    pulse_result, pulse_predictions = _stage_result(
        dataset,
        pulse_stage,
        physical_model,
        config,
        pulse_loss,
        set_pulses=set_pulses,
        reset_pulses=reset_pulses,
    )
    results.append(pulse_result)
    print(json.dumps(asdict(pulse_result), indent=2))
    _plot_predictions(
        dataset,
        pulse_predictions,
        output_dir / f"{checkpoint_tag}_{pulse_stage}.png",
        f"{dataset_name} / {pulse_stage} / k={k:g}",
    )
    torch.save(
        {"model_state": physical_model.state_dict(), "result": asdict(pulse_result)},
        output_dir / f"{checkpoint_tag}_{pulse_stage}.pt",
    )
    return results


def run_suite(args: argparse.Namespace) -> list[HardwareStageResult]:
    all_results: list[HardwareStageResult] = []
    for dataset_name in args.datasets:
        for k in args.k_values:
            all_results.extend(run_dataset(dataset_name, k, args))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps([asdict(result) for result in all_results], indent=2),
        encoding="utf-8",
    )
    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["complicated_function", "taglietti_yinyang"])
    parser.add_argument("--k-values", nargs="+", type=float, default=[0.2])
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pretrain-epochs", type=int, default=120)
    parser.add_argument("--pulse-epochs", type=int, default=20)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-3)
    parser.add_argument("--bias-learning-rate", type=float, default=5e-4)
    parser.add_argument("--conductance-learning-rate", type=float, default=1e-10)
    parser.add_argument("--max-pulses-per-update", type=int, default=1)
    parser.add_argument("--gradient-deadzone-quantile", type=float, default=0.65)
    parser.add_argument("--n-states", type=int, default=32)
    parser.add_argument("--r-lrs", type=float, default=1e4)
    parser.add_argument("--r-hrs", type=float, default=1e6)
    parser.add_argument("--current-to-voltage-gain", type=float, default=None)
    parser.add_argument("--complete-physical", action="store_true")
    parser.add_argument("--use-gilbert-multiplier", action="store_true")
    parser.add_argument("--gilbert-product-gain", type=float, default=None)
    parser.add_argument("--gilbert-input-linear-range", type=float, default=0.4)
    parser.add_argument("--dynamic-memdiode-pulses", action="store_true")
    parser.add_argument("--dynamic-set-voltage", type=float, default=1.8)
    parser.add_argument("--dynamic-reset-voltage", type=float, default=-1.0)
    parser.add_argument("--dynamic-pulse-width-s", type=float, default=1e-9)
    parser.add_argument("--rram-iv-nonlinearity", type=float, default=0.0)
    parser.add_argument("--variability-std", type=float, default=0.0)
    parser.add_argument("--stuck-lrs-probability", type=float, default=0.0)
    parser.add_argument("--stuck-hrs-probability", type=float, default=0.0)
    parser.add_argument("--adc-clip-voltage", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/hardware_training")
    return parser.parse_args()


def main() -> None:
    run_suite(parse_args())


if __name__ == "__main__":
    main()


