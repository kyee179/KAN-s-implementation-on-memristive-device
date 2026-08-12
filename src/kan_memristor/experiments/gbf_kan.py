"""Train and map a fixed generalized-bell basis KAN.

This experiment replaces each KAN edge function with a fixed bank of
generalized bell functions. The GBF shape parameters are not trainable; only
the crossbar coefficients and node biases are learned.
"""

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
from kan_memristor.hardware.gbf_kan import (
    GBFCellConfig,
    GBFForwardConfig,
    PhysicalGeneralizedBellKAN,
)
from kan_memristor.hardware.memristor import RRAMWeightMapper
from kan_memristor.models import GeneralizedBellKAN, count_parameters


@dataclass(frozen=True)
class GBFTrainConfig:
    dataset: str
    widths: list[int]
    num_basis: int
    basis_width: float | None
    slope: float
    grid_min: float
    grid_max: float
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    n_states: int
    r_lrs: float
    r_hrs: float
    output_current_peak_a: float
    tia_transresistance_ohm: float
    gbf_cell_power_w: float
    tia_power_w: float
    read_time_s: float
    tanh_energy_j_per_activation: float
    clip_energy_j_per_activation: float
    bias_energy_j_per_output: float
    inter_layer_normalization: str
    normalization_gain: float


@dataclass(frozen=True)
class GBFStageResult:
    dataset: str
    stage: str
    task: str
    train_loss: float
    test_loss: float
    test_mse: float | None
    test_accuracy: float | None
    parameter_count: int
    gbf_cell_count: int | None
    tia_count: int | None
    memristor_count: int | None
    frontend_j_per_sample: float | None
    crossbar_read_j_per_sample: float | None
    normalization_j_per_sample: float | None
    bias_j_per_sample: float | None
    total_inference_j_per_sample: float | None
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
    if task == "multiclass_classification":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown task: {task}")


def _evaluate(
    model: nn.Module,
    dataset: SupervisedDataset,
    loss_fn: nn.Module,
    split: str,
) -> tuple[float, float | None, float | None, np.ndarray]:
    model.eval()
    x_np = dataset.x_train if split == "train" else dataset.x_test
    y_np = dataset.y_train if split == "train" else dataset.y_test
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    with torch.no_grad():
        logits = model(x)
        loss = float(loss_fn(logits, y).item())
        predictions = logits.detach().cpu().numpy()
    if dataset.task == "regression":
        mse = float(np.mean((predictions - y_np) ** 2))
        return loss, mse, None, predictions
    if dataset.task == "multiclass_classification":
        classes = np.argmax(predictions, axis=1)
        accuracy = float(np.mean(classes == y_np))
        return loss, None, accuracy, predictions
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(predictions, -60.0, 60.0)))
    classes = (probabilities >= 0.5).astype(np.float32)
    accuracy = float(np.mean(classes == y_np))
    return loss, None, accuracy, probabilities


def _train_software_model(dataset: SupervisedDataset, config: GBFTrainConfig) -> tuple[GeneralizedBellKAN, float]:
    _set_seed(config.seed)
    model = GeneralizedBellKAN(
        config.widths,
        num_basis=config.num_basis,
        grid_range=(config.grid_min, config.grid_max),
        basis_width=config.basis_width,
        slope=config.slope,
        inter_layer_normalization=config.inter_layer_normalization,
        normalization_gain=config.normalization_gain,
    )
    loss_fn = _loss_for_task(dataset.task)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    train_data = TensorDataset(torch.from_numpy(dataset.x_train), torch.from_numpy(dataset.y_train))
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, generator=generator)
    last_loss = 0.0
    model.train()
    for _ in range(config.epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
    return model, last_loss


def _estimate_gbf_inference_energy(
    model: PhysicalGeneralizedBellKAN,
    x: torch.Tensor,
    config: GBFTrainConfig,
) -> tuple[float, float, float, float, float]:
    read_time_s = config.read_time_s
    if read_time_s <= 0.0:
        raise ValueError("read_time_s must be positive")
    if config.tanh_energy_j_per_activation < 0.0:
        raise ValueError("tanh_energy_j_per_activation must be non-negative")
    if config.clip_energy_j_per_activation < 0.0:
        raise ValueError("clip_energy_j_per_activation must be non-negative")
    if config.bias_energy_j_per_output < 0.0:
        raise ValueError("bias_energy_j_per_output must be non-negative")
    batch_size = max(int(x.shape[0]), 1)
    total_crossbar = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    normalization_j = 0.0
    bias_j = 0.0
    current = x
    for index, layer in enumerate(model.layers):
        rows = layer.tia_voltages(current)
        conductance_sum = layer.g_pos.detach() + layer.g_neg.detach()
        layer_energy = torch.einsum("bik,oik->b", rows.pow(2), conductance_sum)
        total_crossbar = total_crossbar + layer_energy.sum() * read_time_s
        bias_j += layer.out_features * config.bias_energy_j_per_output
        current = layer(current)
        if index < len(model.layers) - 1:
            from kan_memristor.models import apply_inter_layer_normalization

            if layer.forward_config.inter_layer_normalization == "tanh":
                normalization_j += layer.out_features * config.tanh_energy_j_per_activation
            elif layer.forward_config.inter_layer_normalization == "clip":
                normalization_j += layer.out_features * config.clip_energy_j_per_activation
            current = apply_inter_layer_normalization(
                current,
                mode=layer.forward_config.inter_layer_normalization,
                gain=layer.forward_config.normalization_gain,
            )
    crossbar_j = float((total_crossbar / batch_size).detach().cpu().item())
    frontend_power = model.estimate_frontend_power_w()
    frontend_j = frontend_power * read_time_s
    return frontend_j, crossbar_j, normalization_j, bias_j, frontend_j + crossbar_j + normalization_j + bias_j


def _stage_result(
    dataset: SupervisedDataset,
    stage: str,
    model: nn.Module,
    config: GBFTrainConfig,
    train_loss: float,
) -> tuple[GBFStageResult, np.ndarray]:
    loss_fn = _loss_for_task(dataset.task)
    test_loss, test_mse, test_accuracy, predictions = _evaluate(model, dataset, loss_fn, split="test")
    if isinstance(model, PhysicalGeneralizedBellKAN):
        x = torch.from_numpy(dataset.x_test)
        frontend_j, crossbar_j, normalization_j, bias_j, total_j = _estimate_gbf_inference_energy(model, x, config)
        gbf_count: int | None = model.count_gbf_cells()
        tia_count: int | None = model.count_tias()
        memristor_count: int | None = model.count_memristors()
        gains: list[float] | None = [float(layer.current_to_voltage_gain.item()) for layer in model.layers]
    else:
        frontend_j = None
        crossbar_j = None
        normalization_j = None
        bias_j = None
        total_j = None
        gbf_count = None
        tia_count = None
        memristor_count = None
        gains = None
    result = GBFStageResult(
        dataset=dataset.name,
        stage=stage,
        task=dataset.task,
        train_loss=train_loss,
        test_loss=test_loss,
        test_mse=test_mse,
        test_accuracy=test_accuracy,
        parameter_count=count_parameters(model),
        gbf_cell_count=gbf_count,
        tia_count=tia_count,
        memristor_count=memristor_count,
        frontend_j_per_sample=frontend_j,
        crossbar_read_j_per_sample=crossbar_j,
        normalization_j_per_sample=normalization_j,
        bias_j_per_sample=bias_j,
        total_inference_j_per_sample=total_j,
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
        axes[1].set_title("prediction")
        fig.colorbar(sc0, ax=axes[0], fraction=0.046)
        fig.colorbar(sc1, ax=axes[1], fraction=0.046)
    elif dataset.task == "classification" and dataset.x_test.shape[1] == 2:
        target = dataset.y_test[:, 0]
        pred = predictions[:, 0]
        axes[0].scatter(dataset.x_test[:, 0], dataset.x_test[:, 1], c=target, s=8, cmap="coolwarm", vmin=0, vmax=1)
        axes[1].scatter(dataset.x_test[:, 0], dataset.x_test[:, 1], c=pred, s=8, cmap="coolwarm", vmin=0, vmax=1)
        axes[0].set_title("target class")
        axes[1].set_title("predicted probability")
    else:
        target = dataset.y_test.reshape(-1)
        pred = np.argmax(predictions, axis=1) if predictions.ndim > 1 and predictions.shape[1] > 1 else (predictions[:, 0] >= 0.0)
        labels = np.arange(int(max(target.max(initial=0), pred.max(initial=0))) + 1)
        target_counts = np.bincount(target.astype(int), minlength=len(labels))
        pred_counts = np.bincount(pred.astype(int), minlength=len(labels))
        axes[0].bar(labels, target_counts)
        axes[1].bar(labels, pred_counts)
        axes[0].set_title("target classes")
        axes[1].set_title("predicted classes")
    for axis in axes:
        if dataset.x_test.shape[1] == 2:
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_dataset(dataset_name: str, args: argparse.Namespace) -> list[GBFStageResult]:
    dataset = load_dataset(dataset_name, n_train=args.n_train, n_test=args.n_test, seed=args.seed)
    config = GBFTrainConfig(
        dataset=dataset_name,
        widths=default_widths(dataset_name, "odd_poly_kan"),
        num_basis=args.num_basis,
        basis_width=args.basis_width,
        slope=args.slope,
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        n_states=args.n_states,
        r_lrs=args.r_lrs,
        r_hrs=args.r_hrs,
        output_current_peak_a=args.output_current_peak_a,
        tia_transresistance_ohm=args.tia_transresistance_ohm,
        gbf_cell_power_w=args.gbf_cell_power_w,
        tia_power_w=args.tia_power_w,
        read_time_s=args.read_time_s,
        tanh_energy_j_per_activation=args.tanh_energy_j_per_activation,
        clip_energy_j_per_activation=args.clip_energy_j_per_activation,
        bias_energy_j_per_output=args.bias_energy_j_per_output,
        inter_layer_normalization=args.inter_layer_normalization,
        normalization_gain=args.normalization_gain,
    )
    mapper = RRAMWeightMapper(r_lrs=args.r_lrs, r_hrs=args.r_hrs, n_states=args.n_states, seed=args.seed)
    cell_config = GBFCellConfig(
        output_current_peak_a=args.output_current_peak_a,
        tia_transresistance_ohm=args.tia_transresistance_ohm,
        cell_power_w=args.gbf_cell_power_w,
        tia_power_w=args.tia_power_w,
    )
    forward_config = GBFForwardConfig(
        inter_layer_normalization=args.inter_layer_normalization,
        normalization_gain=args.normalization_gain,
        adc_clip_voltage=args.adc_clip_voltage,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    software_model, pretrain_loss = _train_software_model(dataset, config)
    software_result, software_predictions = _stage_result(dataset, "software_pretrain", software_model, config, pretrain_loss)
    print(json.dumps(asdict(software_result), indent=2))
    _plot_predictions(
        dataset,
        software_predictions,
        output_dir / f"{dataset_name}_gbf_software.png",
        f"{dataset_name} / GBF software",
    )
    torch.save(
        {"model_state": software_model.state_dict(), "result": asdict(software_result)},
        output_dir / f"{dataset_name}_gbf_software.pt",
    )
    physical_model = PhysicalGeneralizedBellKAN.from_software_model(
        software_model,
        mapper=mapper,
        cell_config=cell_config,
        current_to_voltage_gain=args.current_to_voltage_gain,
        forward_config=forward_config,
    )
    mapped_train_loss, _, _, _ = _evaluate(physical_model, dataset, _loss_for_task(dataset.task), split="train")
    physical_result, physical_predictions = _stage_result(dataset, "physical_mapped", physical_model, config, mapped_train_loss)
    print(json.dumps(asdict(physical_result), indent=2))
    _plot_predictions(
        dataset,
        physical_predictions,
        output_dir / f"{dataset_name}_gbf_physical_mapped.png",
        f"{dataset_name} / GBF physical mapped",
    )
    torch.save(
        {"model_state": physical_model.state_dict(), "result": asdict(physical_result)},
        output_dir / f"{dataset_name}_gbf_physical_mapped.pt",
    )
    return [software_result, physical_result]


def run_suite(args: argparse.Namespace) -> list[GBFStageResult]:
    results: list[GBFStageResult] = []
    for dataset_name in args.datasets:
        results.extend(run_dataset(dataset_name, args))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["complicated_function", "taglietti_yinyang"])
    parser.add_argument("--num-basis", type=int, default=9)
    parser.add_argument("--basis-width", type=float, default=None)
    parser.add_argument("--slope", type=float, default=2.0)
    parser.add_argument("--grid-min", type=float, default=-1.0)
    parser.add_argument("--grid-max", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-states", type=int, default=64)
    parser.add_argument("--r-lrs", type=float, default=1e4)
    parser.add_argument("--r-hrs", type=float, default=1e6)
    parser.add_argument("--output-current-peak-a", type=float, default=1e-6)
    parser.add_argument("--tia-transresistance-ohm", type=float, default=1e6)
    parser.add_argument("--gbf-cell-power-w", type=float, default=4.1e-6)
    parser.add_argument("--tia-power-w", type=float, default=1e-6)
    parser.add_argument("--read-time-s", type=float, default=1e-9)
    parser.add_argument("--tanh-energy-j-per-activation", type=float, default=0.0)
    parser.add_argument("--clip-energy-j-per-activation", type=float, default=0.0)
    parser.add_argument("--bias-energy-j-per-output", type=float, default=0.0)
    parser.add_argument("--current-to-voltage-gain", type=float, default=None)
    parser.add_argument("--adc-clip-voltage", type=float, default=None)
    parser.add_argument("--inter-layer-normalization", choices=["none", "tanh", "clip"], default="tanh")
    parser.add_argument("--normalization-gain", type=float, default=1.0)
    parser.add_argument("--output-dir", default="outputs/gbf_kan")
    return parser.parse_args()


def main() -> None:
    run_suite(parse_args())


if __name__ == "__main__":
    main()
