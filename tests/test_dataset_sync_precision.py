from oxspires_tools.dataset import OxfordSpiresDataset


def _prepare_sequence(tmp_path, image_timestamp: str, cloud_timestamp: str):
    cam_dir = tmp_path / "raw" / "cam0"
    cloud_dir = tmp_path / "processed" / "vilens-slam" / "undist-clouds"
    cam_dir.mkdir(parents=True)
    cloud_dir.mkdir(parents=True)
    (cam_dir / f"{image_timestamp}.jpg").touch()
    sec, nsec = cloud_timestamp.split(".")
    (cloud_dir / f"cloud_{sec}_{nsec}.pcd").touch()
    return OxfordSpiresDataset(tmp_path)


def test_sync_tolerance_keeps_sub_float_spacing_distinct(tmp_path):
    dataset = _prepare_sequence(tmp_path, "1710000000.000000000", "1710000000.000000100")

    assert dataset.check_image_lidar_sync(cam_id=0, tolerance_sec=50e-9) is False


def test_sync_tolerance_accepts_exact_nanosecond_boundary(tmp_path):
    dataset = _prepare_sequence(tmp_path, "1710000000.000000000", "1710000000.000000100")

    assert dataset.check_image_lidar_sync(cam_id=0, tolerance_sec=100e-9) is True


def test_exact_sync_still_requires_identical_timestamp(tmp_path):
    dataset = _prepare_sequence(tmp_path, "1710000000.000000000", "1710000000.000000100")

    assert dataset.check_image_lidar_sync(cam_id=0, tolerance_sec=0.0) is False
