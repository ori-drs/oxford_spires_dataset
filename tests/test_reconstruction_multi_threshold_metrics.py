import numpy as np
import pytest

from oxspires_tools.eval import get_recon_metrics_multi_thresholds


def test_multi_threshold_metrics_reuse_same_correspondences():
    reconstruction = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ground_truth = np.array([[0.0, 0.0, 0.0], [1.08, 0.0, 0.0]])

    results = get_recon_metrics_multi_thresholds(
        reconstruction,
        ground_truth,
        thresholds=[0.05, 0.10],
    )

    assert len(results) == 3
    assert results[0]["accuracy"] == pytest.approx(0.04)
    assert results[0]["completeness"] == pytest.approx(0.04)

    assert results[1] == {
        "threshold": 0.05,
        "precision": 0.5,
        "recall": 0.5,
        "f1_score": 0.5,
    }
    assert results[2] == {
        "threshold": 0.10,
        "precision": 1.0,
        "recall": 1.0,
        "f1_score": 1.0,
    }


def test_multi_threshold_metrics_rejects_empty_correspondence_window():
    reconstruction = np.array([[0.0, 0.0, 0.0]])
    ground_truth = np.array([[10.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="No reconstruction correspondences"):
        get_recon_metrics_multi_thresholds(
            reconstruction,
            ground_truth,
            thresholds=[0.05],
            max_distance=1.0,
        )
