from oxspires_tools.utils import get_image_pcd_sync_pair


def test_sync_pair_keeps_distinct_sub_float64_timestamps(tmp_path):
    image_dir = tmp_path / "images"
    pcd_dir = tmp_path / "clouds"
    image_dir.mkdir()
    pcd_dir.mkdir()

    image_0 = image_dir / "1710000000.000000000.jpg"
    image_100 = image_dir / "1710000000.000000100.jpg"
    image_0.touch()
    image_100.touch()

    cloud_0 = pcd_dir / "cloud_1710000000_000000000.pcd"
    cloud_100 = pcd_dir / "cloud_1710000000_000000100.pcd"
    cloud_0.touch()
    cloud_100.touch()

    pairs = get_image_pcd_sync_pair(image_dir, pcd_dir, ".jpg", timestamp_threshold=0.0)

    assert [(image.name, cloud.name, diff) for image, cloud, diff in pairs] == [
        (image_0.name, cloud_0.name, 0.0),
        (image_100.name, cloud_100.name, 0.0),
    ]


def test_sync_pair_reports_integer_nanosecond_difference(tmp_path):
    image_dir = tmp_path / "images"
    pcd_dir = tmp_path / "clouds"
    image_dir.mkdir()
    pcd_dir.mkdir()

    image = image_dir / "1710000000.000000100.jpg"
    cloud = pcd_dir / "cloud_1710000000_000000000.pcd"
    image.touch()
    cloud.touch()

    pairs = get_image_pcd_sync_pair(image_dir, pcd_dir, ".jpg", timestamp_threshold=1e-7)

    assert len(pairs) == 1
    assert pairs[0][0] == image
    assert pairs[0][1] == cloud
    assert pairs[0][2] == 1e-7
