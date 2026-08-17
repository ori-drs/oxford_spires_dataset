import numpy as np

from oxspires_tools.eval import get_distances, get_recon_metrics


EMPTY = np.empty((0, 3), dtype=float)
POINTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)


def test_get_distances_returns_stable_tuple_for_empty_ground_truth():
    assert get_distances(POINTS, EMPTY) == (None, None)


def test_get_distances_returns_stable_tuple_for_empty_input():
    assert get_distances(EMPTY, POINTS) == (None, None)


def test_get_recon_metrics_returns_none_for_empty_ground_truth():
    assert get_recon_metrics(POINTS, EMPTY) is None


def test_get_recon_metrics_returns_none_for_empty_input():
    assert get_recon_metrics(EMPTY, POINTS) is None
