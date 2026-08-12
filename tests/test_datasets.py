import numpy as np

from kan_memristor.datasets import make_complicated_regression, make_image_classification_from_arrays, make_yinyang_classification


def test_complicated_regression_shapes():
    dataset = make_complicated_regression(n_train=32, n_test=16, seed=1)
    assert dataset.x_train.shape == (32, 2)
    assert dataset.y_train.shape == (32, 1)
    assert dataset.x_test.shape == (16, 2)
    assert dataset.y_test.shape == (16, 1)
    assert np.isfinite(dataset.y_train).all()


def test_yinyang_classification_labels_are_binary():
    dataset = make_yinyang_classification(n_train=64, n_test=32, seed=2)
    assert dataset.x_train.shape == (64, 2)
    assert set(np.unique(dataset.y_train)).issubset({0.0, 1.0})
    assert 0.0 < dataset.y_train.mean() < 1.0


def test_image_classification_arrays_are_flattened_and_scaled():
    x_train = np.array([[[0, 255], [128, 64]], [[255, 0], [64, 128]]], dtype=np.uint8)
    y_train = np.array([1, 2])
    dataset = make_image_classification_from_arrays(x_train, y_train, x_train, y_train, "tiny_images", n_train=2, n_test=1)
    assert dataset.x_train.shape == (2, 4)
    assert dataset.y_train.dtype == np.int64
    assert dataset.task == "multiclass_classification"
    assert np.all(dataset.x_train >= -1.0)
    assert np.all(dataset.x_train <= 1.0)
