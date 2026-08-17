import numpy as np

from oxspires_tools.eval import get_recon_metrics


def test_identical_clouds_have_perfect_symmetric_metrics():
    cloud = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    metrics = get_recon_metrics(cloud, cloud, precision_threshold=0.1, recall_threshold=0.1)

    assert metrics["accuracy"] == 0.0
    assert metrics["completeness"] == 0.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0


def test_symmetric_metrics_penalise_missing_geometry_in_recall():
    reconstruction = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ground_truth = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    metrics = get_recon_metrics(
        reconstruction,
        ground_truth,
        precision_threshold=0.1,
        recall_threshold=0.1,
    )

    assert metrics["accuracy"] == 0.0
    assert np.isclose(metrics["completeness"], 1.0 / 3.0)
    assert metrics["precision"] == 1.0
    assert np.isclose(metrics["recall"], 2.0 / 3.0)
    assert np.isclose(metrics["f1_score"], 0.8)


def test_one_sided_metrics_remain_available():
    reconstruction = np.array([[0.0, 0.0, 0.0]])
    ground_truth = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    metrics = get_recon_metrics(
        reconstruction,
        ground_truth,
        precision_threshold=0.1,
        compute_recall=False,
    )

    assert metrics["accuracy"] == 0.0
    assert metrics["precision"] == 1.0
    assert metrics["completeness"] is None
    assert metrics["recall"] is None
    assert metrics["f1_score"] is None
