"""Deterministic datasets for early KAN validation experiments."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SupervisedDataset:
    """Simple train/test split with NumPy arrays."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    task: str
    name: str


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def complicated_function(x: np.ndarray) -> np.ndarray:
    """A deliberately mixed smooth 2D function for KAN regression tests."""

    x0 = x[:, 0]
    x1 = x[:, 1]
    y = (
        np.sin(3.0 * np.pi * x0)
        + 0.55 * np.cos(5.0 * np.pi * x1)
        + 0.65 * np.sin(2.0 * np.pi * x0 * x1)
        + 0.35 * np.exp(-7.0 * ((x0 - 0.35) ** 2 + (x1 + 0.25) ** 2))
        - 0.25 * x0**3
        + 0.20 * x1**2
    )
    return y.astype(np.float32)[:, None]


def make_complicated_regression(
    n_train: int = 2048,
    n_test: int = 2048,
    seed: int = 7,
    noise_std: float = 0.02,
) -> SupervisedDataset:
    """Sample a nonlinear regression task on [-1, 1]^2."""

    generator = _rng(seed)
    x_train = generator.uniform(-1.0, 1.0, size=(n_train, 2)).astype(np.float32)
    x_test = generator.uniform(-1.0, 1.0, size=(n_test, 2)).astype(np.float32)
    y_train = complicated_function(x_train)
    y_test = complicated_function(x_test)
    if noise_std > 0:
        y_train = y_train + generator.normal(0.0, noise_std, size=y_train.shape).astype(np.float32)
    mean = y_train.mean(axis=0, keepdims=True)
    std = y_train.std(axis=0, keepdims=True) + 1e-7
    return SupervisedDataset(
        x_train=x_train,
        y_train=((y_train - mean) / std).astype(np.float32),
        x_test=x_test,
        y_test=((y_test - mean) / std).astype(np.float32),
        task="regression",
        name="complicated_function",
    )


def _sample_unit_disk(generator: np.random.Generator, n: int) -> np.ndarray:
    points: list[np.ndarray] = []
    remaining = n
    while remaining > 0:
        candidates = generator.uniform(-1.0, 1.0, size=(max(remaining * 2, 128), 2))
        candidates = candidates[np.sum(candidates**2, axis=1) <= 1.0]
        if candidates.size == 0:
            continue
        take = min(remaining, len(candidates))
        points.append(candidates[:take])
        remaining -= take
    return np.concatenate(points, axis=0).astype(np.float32)


def yinyang_labels(x: np.ndarray) -> np.ndarray:
    """Generate a procedural yin-yang style binary classification target.

    Taglietti et al. use a yin-yang shaped synthetic dataset. The paper gives
    the train/test sizes and architecture but does not publish sample points in
    the PDF, so this function provides a deterministic paper-inspired generator.
    """

    x0 = x[:, 0]
    x1 = x[:, 1]
    s_boundary = 0.33 * np.sin(np.pi * x1)
    labels = x0 > s_boundary

    upper_dot = (x0**2 + (x1 - 0.48) ** 2) < 0.12**2
    lower_dot = (x0**2 + (x1 + 0.48) ** 2) < 0.12**2
    labels = np.where(upper_dot, True, labels)
    labels = np.where(lower_dot, False, labels)
    return labels.astype(np.float32)[:, None]


def make_yinyang_classification(
    n_train: int = 4000,
    n_test: int = 4000,
    seed: int = 11,
) -> SupervisedDataset:
    """Paper-inspired yin-yang classification benchmark."""

    generator = _rng(seed)
    x_train = _sample_unit_disk(generator, n_train)
    x_test = _sample_unit_disk(generator, n_test)
    return SupervisedDataset(
        x_train=x_train,
        y_train=yinyang_labels(x_train),
        x_test=x_test,
        y_test=yinyang_labels(x_test),
        task="classification",
        name="taglietti_yinyang",
    )


def load_dataset(name: str, n_train: int, n_test: int, seed: int) -> SupervisedDataset:
    if name == "complicated_function":
        return make_complicated_regression(n_train=n_train, n_test=n_test, seed=seed)
    if name == "taglietti_yinyang":
        return make_yinyang_classification(n_train=n_train, n_test=n_test, seed=seed)
    raise ValueError(f"Unknown dataset: {name}")
