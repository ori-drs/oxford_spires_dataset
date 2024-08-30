from oxford_spires_utils.point_cloud import merge_downsample_clouds, merge_downsample_vilens_slam_clouds

if __name__ == "__main__":
    cloud_folder = "/home/yifu/data/nerf_data_pipeline/2024-03-18-chch_4/raw/individual_clouds"
    merge_downsample_vilens_slam_clouds(cloud_folder)

    nerf_cloud_list = [
        "/home/yifu/workspace/T-RO_2025/2024-03-18-chch-4/exported_clouds/rgb only/2024-08-26_003438_rgb.ply",
        "/home/yifu/workspace/T-RO_2025/2024-03-18-chch-4/exported_clouds/rgb only/2024-08-26_190329_rgb.ply",
        "/home/yifu/workspace/T-RO_2025/2024-03-18-chch-4/exported_clouds/rgb only/2024-08-26_191106_rgb.ply",
        "/home/yifu/workspace/T-RO_2025/2024-03-18-chch-4/exported_clouds/rgb only/2024-08-26_191924_rgb.ply",
    ]
    output_cloud_path = "/home/yifu/workspace/T-RO_2025/2024-03-18-chch-4/nerf_rgb_merged.ply"

    merge_downsample_clouds(nerf_cloud_list, output_cloud_path)
