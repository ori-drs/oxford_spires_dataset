from oxspires_tools.dataset import OxfordSpiresDataset


def test_find_raw_cloud_matches_across_second_boundary(tmp_path):
    lidar_dir = tmp_path / "raw" / "lidar-clouds"
    lidar_dir.mkdir(parents=True)
    candidate = lidar_dir / "1710000001.000000100.pcd"
    candidate.touch()

    dataset = OxfordSpiresDataset(tmp_path)
    query_ns = 1_710_000_000_999_999_800

    assert dataset.find_raw_cloud(query_ns, tol_ns=500) == candidate


def test_find_raw_cloud_respects_tolerance_across_second_boundary(tmp_path):
    lidar_dir = tmp_path / "raw" / "lidar-clouds"
    lidar_dir.mkdir(parents=True)
    candidate = lidar_dir / "1710000001.000000100.pcd"
    candidate.touch()

    dataset = OxfordSpiresDataset(tmp_path)
    query_ns = 1_710_000_000_999_999_800

    assert dataset.find_raw_cloud(query_ns, tol_ns=299) is None
