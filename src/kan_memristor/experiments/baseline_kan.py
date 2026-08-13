"""Run early KAN validation experiments."""

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
from kan_memristor.models import BSplineKAN, GeneralizedBellKAN, OddPolynomialKAN, count_parameters, make_mlp


@dataclass(frozen=True)
class TrainConfig:
    dataset: str
    model: str
    widths: list[int]
    num_basis: int
    spline_degree: int
    powers: tuple[int, ...]
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int


@dataclass(frozen=True)
class ExperimentResult:
    dataset: str
    task: str
    model: str
    train_loss: float
    test_loss: float
    test_mse: float | None
    test_accuracy: float | None
    parameter_count: int
    config: dict


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_model(config: TrainConfig) -> nn.Module:
    if config.model == "kan":
        return BSplineKAN(config.widths, num_basis=config.num_basis, degree=config.spline_degree)
    if config.model == "odd_poly_kan":
        return OddPolynomialKAN(config.widths, powers=config.powers)
    if config.model == "gbf_kan":
        return GeneralizedBellKAN(config.widths)
    if config.model == "mlp":
        return make_mlp(config.widths)
    raise ValueError(f"Unknown model: {config.model}")


def _loss_for_task(task: str) -> nn.Module:
    if task == "regression":
        return nn.MSELoss()
    if task == "classification":
        return nn.BCEWithLogitsLoss()
    if task == "multiclass_classification":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown task: {task}")


def _evaluate(model: nn.Module, dataset: SupervisedDataset, loss_fn: nn.Module) -> tuple[float, float | None, float | None, np.ndarray]:
    model.eval()
    x_test = torch.from_numpy(dataset.x_test)
    y_test = torch.from_numpy(dataset.y_test)
    with torch.no_grad():
        logits = model(x_test)
        loss = float(loss_fn(logits, y_test).item())
        predictions = logits.detach().cpu().numpy()
    if dataset.task == "regression":
        mse = float(np.mean((predictions - dataset.y_test) ** 2))
        return loss, mse, None, predictions
    if dataset.task == "multiclass_classification":
        classes = np.argmax(predictions, axis=1)
        accuracy = float(np.mean(classes == dataset.y_test))
        return loss, None, accuracy, predictions
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(predictions, -60.0, 60.0)))
    classes = (probabilities >= 0.5).astype(np.float32)
    accuracy = float(np.mean(classes == dataset.y_test))
    return loss, None, accuracy, probabilities


def train_once(dataset: SupervisedDataset, config: TrainConfig) -> tuple[ExperimentResult, np.ndarray]:
    _set_seed(config.seed)
    model = _make_model(config)
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
            prediction = model(x_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

    test_loss, test_mse, test_accuracy, predictions = _evaluate(model, dataset, loss_fn)
    result = ExperimentResult(
        dataset=dataset.name,
        task=dataset.task,
        model=config.model,
        train_loss=last_loss,
        test_loss=test_loss,
        test_mse=test_mse,
        test_accuracy=test_accuracy,
        parameter_count=count_parameters(model),
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


def default_widths(dataset: str, model: str) -> list[int]:
    if dataset == "complicated_function":
        if model in {"kan", "odd_poly_kan", "gbf_kan"}:
            return [2, 16, 16, 1]
        if model == "mlp":
            return [2, 64, 64, 1]
    if dataset == "taglietti_yinyang":
        if model in {"kan", "odd_poly_kan", "gbf_kan"}:
            return [2, 12, 1]
        if model == "mlp":
            return [2, 64, 64, 1]
    raise ValueError(f"Unknown dataset/model combination: {dataset}/{model}")


def run_suite(args: argparse.Namespace) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    output_dir = Path(args.output_dir)
    for dataset_name in args.datasets:
        dataset = load_dataset(dataset_name, n_train=args.n_train, n_test=args.n_test, seed=args.seed)
        for model_name in args.models:
            config = TrainConfig(
                dataset=dataset_name,
                model=model_name,
                widths=default_widths(dataset_name, model_name),
                num_basis=args.num_basis,
                spline_degree=args.spline_degree,
                powers=tuple(args.powers),
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
            )
            result, predictions = train_once(dataset, config)
            results.append(result)
            plot_name = f"{dataset_name}_{model_name}.png"
            _plot_predictions(dataset, predictions, output_dir / plot_name, f"{dataset_name} / {model_name}")
            print(json.dumps(asdict(result), indent=2))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["complicated_function", "taglietti_yinyang"])
    parser.add_argument("--models", nargs="+", default=["kan", "mlp"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--num-basis", type=int, default=13)
    parser.add_argument("--spline-degree", type=int, default=3)
    parser.add_argument("--powers", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/baseline_tests")
    return parser.parse_args()


def main() -> None:
    run_suite(parse_args())


if __name__ == "__main__":
    main()
