from oxford_spires_utils.point_cloud import merge_downsample_vilens_slam_clouds

if __name__ == "__main__":
    cloud_folder = "/home/yifu/data/nerf_data_pipeline/2024-03-18-chch_4/raw/individual_clouds"
    merge_downsample_vilens_slam_clouds(cloud_folder)
