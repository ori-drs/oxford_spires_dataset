import cv2
import numpy as np
import open3d as o3d
import pytest

from oxspires_tools.depth.projection import decode_points_from_depthmap, get_in_image_mask, project_points_on_image

# Assuming the functions are in a module called 'projection'
# You'll need to adjust the import based on your actual module structure
# from projection import project_points_on_image, get_in_image_mask


class TestGetInImageMask:
    """Test cases for get_in_image_mask function."""

    def test_all_points_inside(self):
        """Test when all points are inside the image boundaries."""
        points = np.array([[10, 20], [50, 60], [100, 150]])
        w, h = 200, 300
        mask = get_in_image_mask(points, w, h)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_all_points_outside(self):
        """Test when all points are outside the image boundaries."""
        points = np.array([[-10, 20], [250, 60], [100, 350]])
        w, h = 200, 300
        mask = get_in_image_mask(points, w, h)
        expected = np.array([False, False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_mixed_points(self):
        """Test when some points are inside and some outside."""
        points = np.array([[10, 20], [-5, 60], [100, 350], [150, 250]])
        w, h = 200, 300
        mask = get_in_image_mask(points, w, h)
        expected = np.array([True, False, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_boundary_points(self):
        """Test points exactly on the boundaries."""
        points = np.array([[0, 0], [199, 299], [200, 300], [0, 300], [200, 0]])
        w, h = 200, 300
        mask = get_in_image_mask(points, w, h)
        # Points at (w, h) and beyond should be False
        expected = np.array([True, True, False, False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_single_point(self):
        """Test with a single point."""
        points = np.array([[50, 75]])
        w, h = 100, 150
        mask = get_in_image_mask(points, w, h)
        expected = np.array([True])
        np.testing.assert_array_equal(mask, expected)

    def test_empty_points(self):
        """Test with empty points array."""
        points = np.array([]).reshape(0, 2)
        w, h = 100, 150
        mask = get_in_image_mask(points, w, h)
        expected = np.array([], dtype=bool)
        np.testing.assert_array_equal(mask, expected)


class TestProjectPointsOnImage:
    """Test cases for project_points_on_image function."""

    @pytest.fixture
    def sample_camera_params(self):
        """Sample camera parameters for testing."""
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
        D = np.array([0.1, -0.2, 0.05, -0.01], dtype=np.float32)
        return K, D

    @pytest.fixture
    def sample_3d_points(self):
        """Sample 3D points for testing."""
        return np.array([[1, 2, 5], [0, 0, 3], [-1, 1, 4]], dtype=np.float32)

    def test_opencv_fisheye_model(self, monkeypatch, sample_camera_params, sample_3d_points):
        """Test projection with OPENCV_FISHEYE camera model."""
        K, D = sample_camera_params
        points_3d = sample_3d_points
        w, h = 640, 480

        # Mock the fisheye projection result
        mock_projected = np.array([[[100, 150]], [[320, 240]], [[200, 300]]], dtype=np.float32)

        def mock_fisheye_project(*args, **kwargs):
            return (mock_projected, None)

        monkeypatch.setattr(cv2.fisheye, "projectPoints", mock_fisheye_project)

        valid_points, valid_mask = project_points_on_image(points_3d, K, D, w, h, "OPENCV_FISHEYE")

        # Check results
        assert isinstance(valid_points, np.ndarray)
        assert isinstance(valid_mask, np.ndarray)
        assert valid_mask.dtype == bool
        assert len(valid_mask) == len(points_3d)

    def test_opencv_model(self, monkeypatch, sample_camera_params, sample_3d_points):
        """Test projection with OPENCV camera model."""
        K, D = sample_camera_params
        points_3d = sample_3d_points
        w, h = 640, 480

        # Mock the standard projection result
        mock_projected = np.array([[[150, 200]], [[320, 240]], [[400, 350]]], dtype=np.float32)

        def mock_project(*args, **kwargs):
            return (mock_projected, None)

        monkeypatch.setattr(cv2, "projectPoints", mock_project)

        valid_points, valid_mask = project_points_on_image(points_3d, K, D, w, h, "OPENCV")

        # Check results
        assert isinstance(valid_points, np.ndarray)
        assert isinstance(valid_mask, np.ndarray)
        assert valid_mask.dtype == bool
        assert len(valid_mask) == len(points_3d)

    def test_unknown_camera_model(self, sample_camera_params, sample_3d_points):
        """Test that unknown camera model raises ValueError."""
        K, D = sample_camera_params
        points_3d = sample_3d_points
        w, h = 640, 480

        with pytest.raises(ValueError, match="Unknown camera model: UNKNOWN"):
            project_points_on_image(points_3d, K, D, w, h, "UNKNOWN")

    def test_single_point_projection(self, monkeypatch, sample_camera_params):
        """Test projection with a single 3D point."""
        K, D = sample_camera_params
        points_3d = np.array([[0, 0, 5]], dtype=np.float32)
        w, h = 640, 480

        # Mock single point projection
        mock_projected = np.array([[[320, 240]]], dtype=np.float32)

        def mock_fisheye_project(*args, **kwargs):
            return (mock_projected, None)

        monkeypatch.setattr(cv2.fisheye, "projectPoints", mock_fisheye_project)

        valid_points, valid_mask = project_points_on_image(points_3d, K, D, w, h, "OPENCV_FISHEYE")

        # Check that single point is handled correctly
        assert valid_points.shape[1] == 2
        assert len(valid_mask) == 1

    def test_points_outside_image(self, monkeypatch, sample_camera_params, sample_3d_points):
        """Test when projected points are outside image boundaries."""
        K, D = sample_camera_params
        points_3d = sample_3d_points
        w, h = 640, 480

        # Mock projection with points outside image
        mock_projected = np.array([[[-100, 150]], [[320, 240]], [[700, 600]]], dtype=np.float32)

        def mock_fisheye_project(*args, **kwargs):
            return (mock_projected, None)

        monkeypatch.setattr(cv2.fisheye, "projectPoints", mock_fisheye_project)

        valid_points, valid_mask = project_points_on_image(points_3d, K, D, w, h, "OPENCV_FISHEYE")

        # Only the middle point should be valid
        expected_mask = np.array([False, True, False])
        np.testing.assert_array_equal(valid_mask, expected_mask)
        assert len(valid_points) == 1
        np.testing.assert_array_equal(valid_points, [[320, 240]])

    def test_rounding_behavior(self, monkeypatch, sample_camera_params, sample_3d_points):
        """Test that projected points are properly rounded."""
        K, D = sample_camera_params
        points_3d = sample_3d_points
        w, h = 640, 480

        # Mock projection with fractional coordinates
        mock_projected = np.array([[[100.7, 150.3]], [[320.1, 240.9]], [[200.5, 300.4]]], dtype=np.float32)

        def mock_fisheye_project(*args, **kwargs):
            return (mock_projected, None)

        monkeypatch.setattr(cv2.fisheye, "projectPoints", mock_fisheye_project)

        valid_points, valid_mask = project_points_on_image(points_3d, K, D, w, h, "OPENCV_FISHEYE")

        # Check that all coordinates are integers (rounded)
        assert np.all(valid_points == np.round(valid_points))

    def test_empty_points_array(self, monkeypatch, sample_camera_params):
        """Test with empty 3D points array."""
        K, D = sample_camera_params
        points_3d = np.array([]).reshape(0, 3)
        w, h = 640, 480

        def mock_fisheye_project(*args, **kwargs):
            return (np.array([]).reshape(0, 1, 2), None)

        monkeypatch.setattr(cv2.fisheye, "projectPoints", mock_fisheye_project)

        valid_points, valid_mask = project_points_on_image(points_3d, K, D, w, h, "OPENCV_FISHEYE")

        assert valid_points.shape == (0, 2)
        assert valid_mask.shape == (0,)
        assert valid_mask.dtype == bool

    # def test_input_shapes(self, sample_camera_params):
    #     """Test input parameter shapes and types."""
    #     K, D = sample_camera_params
    #     points_3d = np.array([[1, 2, 3]], dtype=np.float32)
    #     w, h = 640, 480

    #     # Test wrong K matrix shape
    #     with pytest.raises((ValueError, cv2.error)):
    #         wrong_K = np.array([[500, 0], [0, 500]], dtype=np.float32)
    #         project_points_on_image(points_3d, wrong_K, D, w, h)

    #     # Test wrong points shape
    #     with pytest.raises((ValueError, IndexError)):
    #         wrong_points = np.array([[1, 2]], dtype=np.float32)  # Missing Z coordinate
    #         project_points_on_image(wrong_points, K, D, w, h)


class TestDecodePointsFromDepthmap:
    """Test cases for decode_points_from_depthmap function."""

    @pytest.fixture
    def sample_camera_params(self):
        """Sample camera parameters for testing."""
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
        D = np.array([0.1, -0.2, 0.05, -0.01], dtype=np.float32)
        return K, D

    @pytest.fixture
    def sample_depth_map(self):
        """Sample depth map for testing."""
        # Create a simple 10x10 depth map with some valid and invalid values
        depth = np.zeros((10, 10), dtype=np.uint16)
        depth[2:8, 2:8] = 1000  # Valid depth values in the center
        depth[5, 5] = 2000  # One deeper point
        return depth

    @pytest.fixture
    def sample_color_image_normalised(self):
        """Sample color image matching the depth map."""
        # Create a 10x10x3 color image
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[2:8, 2:8] = [255, 128, 64]  # Some color values
        image_normalised = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        return image_normalised

    def test_opencv_fisheye_model_basic(self, monkeypatch, sample_camera_params, sample_depth_map):
        """Test basic functionality with OPENCV_FISHEYE camera model."""
        K, D = sample_camera_params
        depth = sample_depth_map
        depth_encode_factor = 1000.0

        # Mock undistortPoints to return predictable values
        def mock_undistort_points(*args, **kwargs):
            num_points = args[0].shape[1]  # Number of points
            # Return some mock undistorted points on z=1 plane
            mock_points = np.zeros((1, num_points, 2), dtype=np.float32)
            for i in range(num_points):
                mock_points[0, i] = [i * 0.1, i * 0.1]
            return mock_points

        monkeypatch.setattr(cv2.fisheye, "undistortPoints", mock_undistort_points)

        pcd = decode_points_from_depthmap(depth, K, D, is_euclidean=False, depth_encode_factor=depth_encode_factor)

        # Check that we got a point cloud
        assert isinstance(pcd, o3d.geometry.PointCloud)
        points = np.asarray(pcd.points)
        assert len(points) > 0
        assert points.shape[1] == 3  # 3D points

    def test_opencv_model_basic(self, monkeypatch, sample_camera_params, sample_depth_map):
        """Test basic functionality with OPENCV camera model."""
        K, D = sample_camera_params
        depth = sample_depth_map
        depth_encode_factor = 1000.0

        # Mock undistortPoints for standard OpenCV
        def mock_undistort_points(*args, **kwargs):
            num_points = args[0].shape[1]
            mock_points = np.zeros((1, num_points, 2), dtype=np.float32)
            for i in range(num_points):
                mock_points[0, i] = [i * 0.05, i * 0.05]
            return mock_points

        monkeypatch.setattr(cv2, "undistortPoints", mock_undistort_points)

        pcd = decode_points_from_depthmap(
            depth, K, D, is_euclidean=False, depth_encode_factor=depth_encode_factor, camera_model="OPENCV"
        )

        assert isinstance(pcd, o3d.geometry.PointCloud)
        points = np.asarray(pcd.points)
        assert len(points) > 0
        assert points.shape[1] == 3

    def test_unknown_camera_model(self, sample_camera_params, sample_depth_map):
        """Test that unknown camera model raises ValueError."""
        K, D = sample_camera_params
        depth = sample_depth_map

        with pytest.raises(ValueError, match="Unknown camera model: UNKNOWN"):
            decode_points_from_depthmap(
                depth, K, D, is_euclidean=False, depth_encode_factor=1000.0, camera_model="UNKNOWN"
            )

    def test_euclidean_vs_z_distance(self, monkeypatch, sample_camera_params, sample_depth_map):
        """Test difference between euclidean and z-distance modes."""
        K, D = sample_camera_params
        depth = sample_depth_map
        depth_encode_factor = 1000.0

        def mock_undistort_points(*args, **kwargs):
            num_points = args[0].shape[1]
            mock_points = np.zeros((1, num_points, 2), dtype=np.float32)
            for i in range(num_points):
                mock_points[0, i] = [0.5, 0.5]  # All points at same normalized position
            return mock_points

        monkeypatch.setattr(cv2.fisheye, "undistortPoints", mock_undistort_points)

        # Test euclidean mode
        pcd_euclidean = decode_points_from_depthmap(
            depth, K, D, is_euclidean=True, depth_encode_factor=depth_encode_factor
        )

        # Test z-distance mode
        pcd_z = decode_points_from_depthmap(depth, K, D, is_euclidean=False, depth_encode_factor=depth_encode_factor)

        points_euclidean = np.asarray(pcd_euclidean.points)
        points_z = np.asarray(pcd_z.points)

        # Both should have same number of points
        assert len(points_euclidean) == len(points_z)

        # Points should be different due to normalization in euclidean mode
        if len(points_euclidean) > 0:
            # In euclidean mode, points are normalized then scaled by depth
            # In z mode, points are directly scaled by depth
            assert not np.allclose(points_euclidean, points_z)

    def test_with_color_image(self, monkeypatch, sample_camera_params, sample_depth_map, sample_color_image_normalised):
        """Test processing with color image."""
        K, D = sample_camera_params
        depth = sample_depth_map
        image = sample_color_image_normalised
        depth_encode_factor = 1000.0

        def mock_undistort_points(*args, **kwargs):
            num_points = args[0].shape[1]
            mock_points = np.zeros((1, num_points, 2), dtype=np.float32)
            for i in range(num_points):
                mock_points[0, i] = [i * 0.1, i * 0.1]
            return mock_points

        monkeypatch.setattr(cv2.fisheye, "undistortPoints", mock_undistort_points)

        pcd = decode_points_from_depthmap(
            depth, K, D, is_euclidean=False, depth_encode_factor=depth_encode_factor, image=image
        )

        # Check that colors were added
        assert isinstance(pcd, o3d.geometry.PointCloud)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        assert len(points) > 0
        assert len(colors) == len(points)  # Same number of colors as points
        assert colors.shape[1] == 3  # RGB colors

    def test_empty_depth_map(self, monkeypatch, sample_camera_params):
        """Test with depth map containing no valid values."""
        K, D = sample_camera_params
        depth = np.zeros((10, 10), dtype=np.uint16)  # All zeros = invalid
        depth_encode_factor = 1000.0

        def mock_undistort_points(*args, **kwargs):
            # Should not be called since no valid points
            return np.array([]).reshape(1, 0, 2)

        monkeypatch.setattr(cv2.fisheye, "undistortPoints", mock_undistort_points)

        pcd = decode_points_from_depthmap(depth, K, D, is_euclidean=False, depth_encode_factor=depth_encode_factor)

        # Should return empty point cloud
        assert isinstance(pcd, o3d.geometry.PointCloud)
        points = np.asarray(pcd.points)
        assert len(points) == 0

    def test_single_valid_pixel(self, monkeypatch, sample_camera_params):
        """Test with only one valid pixel in depth map."""
        K, D = sample_camera_params
        depth = np.zeros((10, 10), dtype=np.uint16)
        depth[5, 5] = 1000  # Only one valid pixel
        depth_encode_factor = 1000.0

        def mock_undistort_points(*args, **kwargs):
            # Return single point
            return np.array([[[0.5, 0.5]]], dtype=np.float32)

        monkeypatch.setattr(cv2.fisheye, "undistortPoints", mock_undistort_points)

        pcd = decode_points_from_depthmap(depth, K, D, is_euclidean=False, depth_encode_factor=depth_encode_factor)

        points = np.asarray(pcd.points)
        assert len(points) == 1
        assert points.shape == (1, 3)

    def test_none_undistort_result(self, monkeypatch, sample_camera_params, sample_depth_map):
        """Test handling when undistortPoints returns None."""
        K, D = sample_camera_params
        depth = sample_depth_map
        depth_encode_factor = 1000.0

        def mock_undistort_points(*args, **kwargs):
            return None

        monkeypatch.setattr(cv2.fisheye, "undistortPoints", mock_undistort_points)

        pcd = decode_points_from_depthmap(depth, K, D, is_euclidean=False, depth_encode_factor=depth_encode_factor)

        # Should return empty point cloud
        assert isinstance(pcd, o3d.geometry.PointCloud)
        points = np.asarray(pcd.points)
        assert len(points) == 0

    def test_depth_encoding_factor(self, monkeypatch, sample_camera_params):
        """Test different depth encoding factors."""
        K, D = sample_camera_params
        depth = np.ones((5, 5), dtype=np.uint16) * 1000  # All pixels have value 1000

        def mock_undistort_points(*args, **kwargs):
            num_points = args[0].shape[1]
            mock_points = np.zeros((1, num_points, 2), dtype=np.float32)
            return mock_points

        monkeypatch.setattr(cv2.fisheye, "undistortPoints", mock_undistort_points)

        # Test with different encoding factors
        pcd1 = decode_points_from_depthmap(depth, K, D, is_euclidean=False, depth_encode_factor=1000.0)
        pcd2 = decode_points_from_depthmap(depth, K, D, is_euclidean=False, depth_encode_factor=500.0)

        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)

        # Points should be at different depths due to different encoding factors
        if len(points1) > 0 and len(points2) > 0:
            # With factor 500, depth should be 2x larger than with factor 1000
            assert np.allclose(points2[:, 2], points1[:, 2] * 2, rtol=1e-5)

    def test_different_depth_values(self, monkeypatch, sample_camera_params):
        """Test with different depth values in the map."""
        K, D = sample_camera_params
        depth = np.zeros((5, 5), dtype=np.uint16)
        depth[1, 1] = 1000  # Close point
        depth[2, 2] = 2000  # Far point
        depth[3, 3] = 500  # Very close point
        depth_encode_factor = 1000.0

        def mock_undistort_points(*args, **kwargs):
            num_points = args[0].shape[1]
            mock_points = np.zeros((1, num_points, 2), dtype=np.float32)
            for i in range(num_points):
                mock_points[0, i] = [0.1, 0.1]  # Same normalized position
            return mock_points

        monkeypatch.setattr(cv2.fisheye, "undistortPoints", mock_undistort_points)

        pcd = decode_points_from_depthmap(depth, K, D, is_euclidean=False, depth_encode_factor=depth_encode_factor)

        points = np.asarray(pcd.points)
        assert len(points) == 3  # Three valid points

        # Points should have different z-values corresponding to their depths
        z_values = points[:, 2]

        # The z-values should be proportional to the expected depths
        # (exact values depend on the mock undistort function)
        assert len(np.unique(z_values)) == 3  # All different z-values

    def test_input_types_and_shapes(self, monkeypatch, sample_camera_params):
        """Test various input types and shapes."""
        K, D = sample_camera_params

        def mock_undistort_points(*args, **kwargs):
            return np.array([[[0.1, 0.1]]], dtype=np.float32)

        monkeypatch.setattr(cv2.fisheye, "undistortPoints", mock_undistort_points)

        # Test with different dtypes for depth
        depth_uint16 = np.array([[1000]], dtype=np.uint16)
        depth_float32 = np.array([[1000]], dtype=np.float32)
        depth_int32 = np.array([[1000]], dtype=np.int32)

        for depth in [depth_uint16, depth_float32, depth_int32]:
            pcd = decode_points_from_depthmap(depth, K, D, is_euclidean=False, depth_encode_factor=1000.0)
            assert isinstance(pcd, o3d.geometry.PointCloud)

    def test_large_depth_map(self, monkeypatch, sample_camera_params):
        """Test with larger depth map to check performance and memory handling."""
        K, D = sample_camera_params

        # Create a larger depth map (100x100)
        depth = np.zeros((100, 100), dtype=np.uint16)
        depth[10:90, 10:90] = 1000  # Large valid region
        depth_encode_factor = 1000.0

        def mock_undistort_points(*args, **kwargs):
            num_points = args[0].shape[1]
            mock_points = np.random.rand(1, num_points, 2).astype(np.float32) * 0.1
            return mock_points

        monkeypatch.setattr(cv2.fisheye, "undistortPoints", mock_undistort_points)

        pcd = decode_points_from_depthmap(depth, K, D, is_euclidean=False, depth_encode_factor=depth_encode_factor)

        points = np.asarray(pcd.points)
        # Should have 80*80 = 6400 valid points
        assert len(points) == 6400
        assert points.shape == (6400, 3)


if __name__ == "__main__":
    pytest.main([__file__])
