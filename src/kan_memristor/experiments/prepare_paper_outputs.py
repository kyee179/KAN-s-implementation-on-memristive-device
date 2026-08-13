"""Prepare paper-facing result tables and figures.

This utility does not rerun training. It organizes the most useful existing
results for a paper about KAN implementation on memristive devices, focusing on
the two hardware routes developed in this project:

- Gilbert-multiplier polynomial KAN;
- generalized-bell-function (GBF) KAN.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MLP_ENERGY_PJ = 19.7248e3


@dataclass(frozen=True)
class MainPaperResult:
    dataset: str
    task: str
    route: str
    stage: str
    mse: float | None
    accuracy_pct: float | None
    energy_pj_per_sample: float | None
    memristors: int | None
    nonlinear_blocks: str
    source_output: str
    paper_use: str


MAIN_RESULTS: tuple[MainPaperResult, ...] = (
    MainPaperResult(
        dataset="complicated_function",
        task="regression",
        route="MLP baseline",
        stage="software",
        mse=0.3103488088,
        accuracy_pct=None,
        energy_pj_per_sample=None,
        memristors=None,
        nonlinear_blocks="-",
        source_output="outputs/baseline_tests",
        paper_use="software baseline",
    ),
    MainPaperResult(
        dataset="complicated_function",
        task="regression",
        route="Software B-spline KAN",
        stage="software",
        mse=0.0006223622,
        accuracy_pct=None,
        energy_pj_per_sample=None,
        memristors=None,
        nonlinear_blocks="-",
        source_output="outputs/baseline_tests",
        paper_use="software upper bound",
    ),
    MainPaperResult(
        dataset="complicated_function",
        task="regression",
        route="Gilbert physical KAN",
        stage="physical mapped",
        mse=0.4111667573,
        accuracy_pct=None,
        energy_pj_per_sample=9.102433,
        memristors=1824,
        nonlinear_blocks="204 Gilbert multipliers",
        source_output="outputs/normalized_physical_training_gain2",
        paper_use="main hardware route",
    ),
    MainPaperResult(
        dataset="complicated_function",
        task="regression",
        route="GBF physical KAN",
        stage="physical mapped",
        mse=0.0078100427,
        accuracy_pct=None,
        energy_pj_per_sample=12.045215,
        memristors=5472,
        nonlinear_blocks="306 GBF cells + 306 TIAs",
        source_output="outputs/gbf_kan_fixed_basis",
        paper_use="main hardware route",
    ),
    MainPaperResult(
        dataset="taglietti_yinyang",
        task="classification",
        route="MLP baseline",
        stage="software",
        mse=None,
        accuracy_pct=94.580078,
        energy_pj_per_sample=None,
        memristors=None,
        nonlinear_blocks="-",
        source_output="outputs/baseline_tests",
        paper_use="software baseline",
    ),
    MainPaperResult(
        dataset="taglietti_yinyang",
        task="classification",
        route="Software B-spline KAN",
        stage="software",
        mse=None,
        accuracy_pct=99.267578,
        energy_pj_per_sample=None,
        memristors=None,
        nonlinear_blocks="-",
        source_output="outputs/baseline_tests",
        paper_use="software upper bound",
    ),
    MainPaperResult(
        dataset="taglietti_yinyang",
        task="classification",
        route="Gilbert physical KAN",
        stage="physical mapped",
        mse=None,
        accuracy_pct=94.335938,
        energy_pj_per_sample=3.715814,
        memristors=216,
        nonlinear_blocks="84 Gilbert multipliers",
        source_output="outputs/normalized_physical_training_gain2",
        paper_use="main hardware route",
    ),
    MainPaperResult(
        dataset="taglietti_yinyang",
        task="classification",
        route="GBF physical KAN",
        stage="physical mapped",
        mse=None,
        accuracy_pct=98.828125,
        energy_pj_per_sample=3.110243,
        memristors=648,
        nonlinear_blocks="126 GBF cells + 126 TIAs",
        source_output="outputs/gbf_kan_fixed_basis",
        paper_use="main hardware route",
    ),
)


POLYNOMIAL_SWEEP_ROWS: tuple[dict[str, str], ...] = (
    {
        "dataset": "complicated_function",
        "powers": "1,3,5",
        "ideal_result": "MSE 0.07749",
        "physical_result": "MSE 0.41117",
        "memristors": "1824",
        "gilbert_multipliers": "204",
        "energy_pj_per_sample": "9.10",
    },
    {
        "dataset": "complicated_function",
        "powers": "1..5",
        "ideal_result": "MSE 0.02506",
        "physical_result": "MSE 0.47372",
        "memristors": "3040",
        "gilbert_multipliers": "340",
        "energy_pj_per_sample": "15.11",
    },
    {
        "dataset": "complicated_function",
        "powers": "1..7",
        "ideal_result": "MSE 0.01065",
        "physical_result": "MSE 0.43608",
        "memristors": "4256",
        "gilbert_multipliers": "714",
        "energy_pj_per_sample": "31.57",
    },
    {
        "dataset": "taglietti_yinyang",
        "powers": "1,3,5",
        "ideal_result": "Accuracy 95.61%",
        "physical_result": "Accuracy 94.34%",
        "memristors": "216",
        "gilbert_multipliers": "84",
        "energy_pj_per_sample": "3.72",
    },
    {
        "dataset": "taglietti_yinyang",
        "powers": "1..5",
        "ideal_result": "Accuracy 95.80%",
        "physical_result": "Accuracy 94.78%",
        "memristors": "360",
        "gilbert_multipliers": "140",
        "energy_pj_per_sample": "6.19",
    },
    {
        "dataset": "taglietti_yinyang",
        "powers": "1..7",
        "ideal_result": "Accuracy 95.85%",
        "physical_result": "Accuracy 94.68%",
        "memristors": "504",
        "gilbert_multipliers": "294",
        "energy_pj_per_sample": "12.96",
    },
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _write_main_tables(output_dir: Path) -> None:
    table_dir = output_dir / "tables"
    rows = [asdict(result) for result in MAIN_RESULTS]
    _write_csv(table_dir / "main_accuracy_energy_summary.csv", rows)

    energy_rows: list[dict[str, object]] = []
    for result in MAIN_RESULTS:
        if result.energy_pj_per_sample is None:
            continue
        energy_rows.append(
            {
                "dataset": result.dataset,
                "route": result.route,
                "physical_energy_pj_per_sample": result.energy_pj_per_sample,
                "digital_mlp_energy_pj_per_sample": MLP_ENERGY_PJ,
                "physical_percent_of_mlp": 100.0 * result.energy_pj_per_sample / MLP_ENERGY_PJ,
                "source_output": result.source_output,
            }
        )
    _write_csv(table_dir / "energy_comparison.csv", energy_rows)
    _write_csv(table_dir / "supplementary_polynomial_sweep.csv", list(POLYNOMIAL_SWEEP_ROWS))

    with (table_dir / "main_accuracy_energy_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def _plot_main_comparison(output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    routes = ["MLP baseline", "Software B-spline KAN", "Gilbert physical KAN", "GBF physical KAN"]
    colors = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b"]

    regression = {
        result.route: result.mse
        for result in MAIN_RESULTS
        if result.dataset == "complicated_function"
    }
    classification = {
        result.route: result.accuracy_pct
        for result in MAIN_RESULTS
        if result.dataset == "taglietti_yinyang"
    }
    energy = {
        f"{result.route}\n{result.dataset.replace('_', ' ')}": result.energy_pj_per_sample
        for result in MAIN_RESULTS
        if result.energy_pj_per_sample is not None
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    axes[0].bar(routes, [regression[route] for route in routes], color=colors)
    axes[0].set_yscale("log")
    axes[0].set_title("Regression")
    axes[0].set_ylabel("Test MSE, log scale")
    axes[0].grid(axis="y", alpha=0.25)
    for index, route in enumerate(routes):
        value = regression[route]
        axes[0].text(index, value * 1.18, f"{value:.4g}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(routes, [classification[route] for route in routes], color=colors)
    axes[1].set_title("Classification")
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].set_ylim(90, 100)
    axes[1].grid(axis="y", alpha=0.25)
    for index, route in enumerate(routes):
        value = classification[route]
        axes[1].text(index, value + 0.12, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)

    energy_names = list(energy)
    energy_values = [energy[name] for name in energy_names]
    axes[2].bar(["Digital MLP"] + energy_names, [MLP_ENERGY_PJ] + energy_values, color=["#bab0ac"] + colors[2:] * 2)
    axes[2].set_yscale("log")
    axes[2].set_title("Physical Inference Energy")
    axes[2].set_ylabel("pJ/sample, log scale")
    axes[2].grid(axis="y", alpha=0.25)
    for index, value in enumerate([MLP_ENERGY_PJ] + energy_values):
        label = f"{value / 1000:.2f} nJ" if value >= 1000 else f"{value:.2f} pJ"
        axes[2].text(index, value * 1.18, label, ha="center", va="bottom", fontsize=8)

    for axis in axes:
        axis.tick_params(axis="x", labelrotation=25)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")

    fig.suptitle("Paper Candidate Results: Accuracy and Energy")
    fig.savefig(figure_dir / "main_accuracy_energy_comparison.svg")
    plt.close(fig)


def _copy_raw_metrics(repo_root: Path, output_dir: Path) -> None:
    raw_dir = output_dir / "raw_metrics"
    raw_sources = {
        "baseline_tests_metrics.json": repo_root / "outputs/baseline_tests/metrics.json",
        "gilbert_normalized_metrics.json": repo_root / "outputs/normalized_physical_training_gain2/metrics.json",
        "gilbert_energy_metrics.json": repo_root / "outputs/energy_estimation/energy_metrics.json",
        "gbf_fixed_basis_metrics.json": repo_root / "outputs/gbf_kan_fixed_basis/metrics.json",
        "polynomial_1_to_5_metrics.json": repo_root / "outputs/polynomial_1_to_5_physical_mapped/metrics.json",
        "full_polynomial_metrics.json": repo_root / "outputs/full_polynomial_physical_mapped/metrics.json",
    }
    for name, source in raw_sources.items():
        _copy_if_exists(source, raw_dir / name)

    figure_sources = {
        "preserved_results_comparison.svg": repo_root / "docs/assets/preserved_results_comparison.svg",
        "gilbert_product_error.png": repo_root / "outputs/hardware_characterization/gilbert_product_error.png",
        "memdiode_iv.png": repo_root / "outputs/hardware_characterization/memdiode_iv.png",
        "gbf_regression_physical.png": repo_root / "outputs/gbf_kan_fixed_basis/complicated_function_gbf_physical_mapped.png",
        "gbf_classification_physical.png": repo_root / "outputs/gbf_kan_fixed_basis/taglietti_yinyang_gbf_physical_mapped.png",
        "gilbert_regression_physical.png": repo_root
        / "outputs/normalized_physical_training_gain2/complicated_function_k0.2_complete_physical_mapped.png",
        "gilbert_classification_physical.png": repo_root
        / "outputs/normalized_physical_training_gain2/taglietti_yinyang_k0.2_complete_physical_mapped.png",
    }
    for name, source in figure_sources.items():
        _copy_if_exists(source, output_dir / "figures" / name)


def _write_readme(output_dir: Path) -> None:
    readme = """# Paper-Ready Outputs

This folder collects the useful results for a paper on implementing KANs on memristive devices using two hardware routes:

1. Gilbert-multiplier polynomial KAN;
2. generalized-bell-function (GBF) KAN.

The image benchmark and smoke-test runs are not part of the paper story. They are archived under `outputs/_archive/` if they are ever needed again.

## Main Files

- `tables/main_accuracy_energy_summary.csv`: one table for the main accuracy and energy claims.
- `tables/main_accuracy_energy_summary.json`: JSON copy of the same table.
- `tables/energy_comparison.csv`: physical inference energy versus the digital MLP estimate.
- `tables/supplementary_polynomial_sweep.csv`: optional supplementary table for `x`, `x^3`, `x^5` versus expanded polynomial bases.
- `figures/main_accuracy_energy_comparison.svg`: compact paper-candidate visual comparing accuracy and energy.
- `raw_metrics/`: copied raw JSON metrics from the useful experiment folders.

## Recommended Paper Use

Use these as the main result set:

- software B-spline KAN as the ideal software KAN upper reference;
- MLP as the software baseline;
- normalized complete-physical Gilbert KAN as the polynomial hardware route;
- mapped GBF-KAN as the localized-basis hardware route.

The strongest hardware result is the GBF route on the original tasks: it reaches MSE `0.00781` on the complicated regression task and `98.83%` accuracy on the Taglietti-inspired yin-yang classification task after physical mapping. The Gilbert route is still important because it directly demonstrates polynomial edge functions generated by physical multipliers and RRAM weights; its main limitation is regression sensitivity after multiplier-chain and conductance mapping.

Energy values are lower-bound compute-array estimates. They include the modeled RRAM read and nonlinear frontend/multiplier costs, but not a complete chip-level ADC/DAC/controller budget.

## Useful Raw Folders

- `outputs/baseline_tests`
- `outputs/normalized_physical_training_gain2`
- `outputs/gbf_kan_fixed_basis`
- `outputs/energy_estimation`
- `outputs/full_polynomial_physical_mapped`
- `outputs/polynomial_1_to_5_physical_mapped`

## Less Useful For The Main Paper

- `outputs/_archive/removed_image_benchmarks`: image benchmarks from the removed scope.
- `outputs/_archive/smoke_and_sanity_runs`: short tests used only to verify code paths.
- pulse-trained Gilbert runs: useful to discuss limitations, but not yet strong enough as main accuracy results.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def prepare(repo_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_main_tables(output_dir)
    _plot_main_comparison(output_dir)
    _copy_raw_metrics(repo_root, output_dir)
    _write_readme(output_dir)


def main() -> None:
    repo_root = Path.cwd()
    prepare(repo_root=repo_root, output_dir=repo_root / "outputs/paper_ready")


if __name__ == "__main__":
    main()
