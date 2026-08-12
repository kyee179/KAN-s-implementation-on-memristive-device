"""Compare MLP, Gilbert-polynomial KAN, and GBF-KAN on MNIST-like datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kan_memristor.datasets import SupervisedDataset, load_dataset
from kan_memristor.hardware.gbf_kan import GBFCellConfig, GBFForwardConfig, PhysicalGeneralizedBellKAN
from kan_memristor.hardware.gilbert_multiplier import GilbertMultiplierParameters
from kan_memristor.hardware.memristor import RRAMWeightMapper
from kan_memristor.hardware.physical_kan import DynamicPulseConfig, PhysicalForwardConfig, PhysicalOddPolynomialKAN
from kan_memristor.models import GeneralizedBellKAN, OddPolynomialKAN, count_parameters, make_mlp


@dataclass(frozen=True)
class ImageBenchmarkResult:
    dataset: str
    model: str
    stage: str
    train_loss: float
    test_loss: float
    test_accuracy: float
    parameter_count: int
    memristor_count: int | None
    gilbert_multiplier_count: int | None
    gbf_cell_count: int | None
    tia_count: int | None
    config: dict


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def _train(
    model: nn.Module,
    dataset: SupervisedDataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> float:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_data = TensorDataset(torch.from_numpy(dataset.x_train), torch.from_numpy(dataset.y_train))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
    return last_loss


def _evaluate(model: nn.Module, dataset: SupervisedDataset, split: str = "test") -> tuple[float, float]:
    x_np = dataset.x_test if split == "test" else dataset.x_train
    y_np = dataset.y_test if split == "test" else dataset.y_train
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return float(loss_fn(logits, y).item()), _accuracy(logits, y)


def _result(
    dataset: SupervisedDataset,
    model_name: str,
    stage: str,
    model: nn.Module,
    train_loss: float,
    config: dict,
) -> ImageBenchmarkResult:
    test_loss, test_accuracy = _evaluate(model, dataset)
    memristor_count: int | None = None
    gilbert_count: int | None = None
    gbf_count: int | None = None
    tia_count: int | None = None
    if isinstance(model, PhysicalOddPolynomialKAN):
        memristor_count = model.count_memristors()
        gilbert_count = model.count_gilbert_multipliers()
    if isinstance(model, PhysicalGeneralizedBellKAN):
        memristor_count = model.count_memristors()
        gbf_count = model.count_gbf_cells()
        tia_count = model.count_tias()
    return ImageBenchmarkResult(
        dataset=dataset.name,
        model=model_name,
        stage=stage,
        train_loss=train_loss,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        parameter_count=count_parameters(model),
        memristor_count=memristor_count,
        gilbert_multiplier_count=gilbert_count,
        gbf_cell_count=gbf_count,
        tia_count=tia_count,
        config=config,
    )


def _widths(hidden_width: int) -> list[int]:
    return [784, hidden_width, 10]


def run_dataset(dataset_name: str, args: argparse.Namespace) -> list[ImageBenchmarkResult]:
    _set_seed(args.seed)
    dataset = load_dataset(dataset_name, n_train=args.n_train, n_test=args.n_test, seed=args.seed)
    gilbert_widths = _widths(args.kan_hidden_width)
    gbf_widths = _widths(args.gbf_hidden_width)
    mapper = RRAMWeightMapper(n_states=args.n_states, r_lrs=args.r_lrs, r_hrs=args.r_hrs, seed=args.seed)
    common_config = {
        "n_train": args.n_train,
        "n_test": args.n_test,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "n_states": args.n_states,
    }
    results: list[ImageBenchmarkResult] = []

    if "mlp" in args.models:
        _set_seed(args.seed)
        mlp_widths = [784, args.mlp_hidden_width, 10]
        mlp = make_mlp(mlp_widths)
        train_loss = _train(mlp, dataset, args.epochs, args.batch_size, args.learning_rate, args.seed)
        results.append(_result(dataset, "mlp", "software", mlp, train_loss, common_config | {"widths": mlp_widths}))

    if "gilbert_kan" in args.models:
        _set_seed(args.seed)
        software_poly = OddPolynomialKAN(
            gilbert_widths,
            powers=tuple(args.powers),
            inter_layer_normalization=args.inter_layer_normalization,
            normalization_gain=args.normalization_gain,
        )
        train_loss = _train(software_poly, dataset, args.epochs, args.batch_size, args.learning_rate, args.seed)
        results.append(
            _result(
                dataset,
                "gilbert_kan",
                "software_pretrain",
                software_poly,
                train_loss,
                common_config | {"widths": gilbert_widths, "powers": args.powers},
            )
        )
        forward_config = PhysicalForwardConfig(
            use_gilbert_multiplier=True,
            gilbert_parameters=GilbertMultiplierParameters(
                product_gain=1.0 / args.input_scale_v,
                input_linear_range=args.gilbert_input_linear_range,
                soft_clip=args.gilbert_soft_clip,
            ),
            inter_layer_normalization=args.inter_layer_normalization,
            normalization_gain=args.normalization_gain,
        )
        physical_poly = PhysicalOddPolynomialKAN.from_software_model(
            software_poly,
            mapper=mapper,
            input_scale_v=args.input_scale_v,
            forward_config=forward_config,
            dynamic_pulse_config=DynamicPulseConfig(enabled=False),
        )
        mapped_train_loss, _ = _evaluate(physical_poly, dataset, split="train")
        results.append(
            _result(
                dataset,
                "gilbert_kan",
                "physical_mapped",
                physical_poly,
                mapped_train_loss,
                common_config | {"widths": gilbert_widths, "powers": args.powers},
            )
        )

    if "gbf_kan" in args.models:
        _set_seed(args.seed)
        software_gbf = GeneralizedBellKAN(
            gbf_widths,
            num_basis=args.num_basis,
            basis_width=args.basis_width,
            slope=args.slope,
            inter_layer_normalization=args.inter_layer_normalization,
            normalization_gain=args.normalization_gain,
        )
        train_loss = _train(software_gbf, dataset, args.epochs, args.batch_size, args.learning_rate, args.seed)
        results.append(
            _result(
                dataset,
                "gbf_kan",
                "software_pretrain",
                software_gbf,
                train_loss,
                common_config | {"widths": gbf_widths, "num_basis": args.num_basis, "slope": args.slope},
            )
        )
        physical_gbf = PhysicalGeneralizedBellKAN.from_software_model(
            software_gbf,
            mapper=mapper,
            cell_config=GBFCellConfig(
                output_current_peak_a=args.output_current_peak_a,
                tia_transresistance_ohm=args.tia_transresistance_ohm,
            ),
            forward_config=GBFForwardConfig(
                inter_layer_normalization=args.inter_layer_normalization,
                normalization_gain=args.normalization_gain,
            ),
        )
        mapped_train_loss, _ = _evaluate(physical_gbf, dataset, split="train")
        results.append(
            _result(
                dataset,
                "gbf_kan",
                "physical_mapped",
                physical_gbf,
                mapped_train_loss,
                common_config | {"widths": gbf_widths, "num_basis": args.num_basis, "slope": args.slope},
            )
        )

    return results


def run_suite(args: argparse.Namespace) -> list[ImageBenchmarkResult]:
    results: list[ImageBenchmarkResult] = []
    for dataset_name in args.datasets:
        results.extend(run_dataset(dataset_name, args))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(result) for result in results]
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for result in payload:
        print(json.dumps(result, indent=2))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["mnist", "fashion_mnist"])
    parser.add_argument("--models", nargs="+", default=["mlp", "gilbert_kan", "gbf_kan"])
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--kan-hidden-width", type=int, default=None)
    parser.add_argument("--gbf-hidden-width", type=int, default=None)
    parser.add_argument("--mlp-hidden-width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--powers", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--num-basis", type=int, default=9)
    parser.add_argument("--basis-width", type=float, default=None)
    parser.add_argument("--slope", type=float, default=2.0)
    parser.add_argument("--input-scale-v", type=float, default=0.2)
    parser.add_argument("--gilbert-input-linear-range", type=float, default=0.4)
    parser.add_argument("--gilbert-soft-clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-states", type=int, default=64)
    parser.add_argument("--r-lrs", type=float, default=1e4)
    parser.add_argument("--r-hrs", type=float, default=1e6)
    parser.add_argument("--output-current-peak-a", type=float, default=1e-6)
    parser.add_argument("--tia-transresistance-ohm", type=float, default=1e6)
    parser.add_argument("--inter-layer-normalization", choices=["none", "tanh", "clip"], default="tanh")
    parser.add_argument("--normalization-gain", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/image_benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kan_hidden_width is None:
        args.kan_hidden_width = args.hidden_width
    if args.gbf_hidden_width is None:
        args.gbf_hidden_width = args.hidden_width
    run_suite(args)


if __name__ == "__main__":
    main()
