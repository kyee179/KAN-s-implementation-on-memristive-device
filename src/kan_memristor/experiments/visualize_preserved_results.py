"""Visualize preserved pre-image benchmark results.

The values in this module are copied from the non-image experiment notes in
``docs/``. They intentionally exclude the later image benchmark series so the
project summary stays focused on the original two research tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PreservedResult:
    model: str
    regression_mse: float
    classification_accuracy_pct: float
    source: str


PRESERVED_RESULTS: tuple[PreservedResult, ...] = (
    PreservedResult(
        model="MLP baseline",
        regression_mse=0.31035,
        classification_accuracy_pct=94.58,
        source="docs/kan_baseline_experiments.md",
    ),
    PreservedResult(
        model="Software B-spline KAN",
        regression_mse=0.00062,
        classification_accuracy_pct=99.27,
        source="docs/kan_baseline_experiments.md",
    ),
    PreservedResult(
        model="Gilbert physical KAN",
        regression_mse=0.41117,
        classification_accuracy_pct=94.34,
        source="docs/inter_layer_normalization_experiment.md",
    ),
    PreservedResult(
        model="GBF physical KAN",
        regression_mse=0.00781,
        classification_accuracy_pct=98.83,
        source="docs/gbf_kan_experiment.md",
    ),
)


def render(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = [result.model for result in PRESERVED_RESULTS]
    regression = [result.regression_mse for result in PRESERVED_RESULTS]
    classification = [result.classification_accuracy_pct for result in PRESERVED_RESULTS]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b"]

    axes[0].bar(names, regression, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Test MSE, log scale")
    axes[0].set_title("Complicated function regression")
    axes[0].grid(axis="y", alpha=0.25)
    for index, value in enumerate(regression):
        axes[0].text(index, value * 1.18, f"{value:.5g}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(names, classification, color=colors)
    axes[1].set_ylim(90.0, 100.0)
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].set_title("Taglietti-inspired yin-yang classification")
    axes[1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(classification):
        axes[1].text(index, value + 0.12, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)

    for axis in axes:
        axis.tick_params(axis="x", labelrotation=25)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")

    fig.suptitle("Preserved Pre-Image Experiment Results")
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    render(Path("docs/assets/preserved_results_comparison.svg"))


if __name__ == "__main__":
    main()
