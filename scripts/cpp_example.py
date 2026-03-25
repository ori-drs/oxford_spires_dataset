import logging
from pathlib import Path

from oxspires_tools.cpp import OcTree, convertOctreeToPointCloud, processPCDFolder, removeUnknownPoints

logger = logging.getLogger(__name__)

save_dir = "/home/docker_dev/oxford_spires_dataset/test_octree"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
pcd_folder = str(save_dir / "test_lidar")
octomap_bt_path = str(save_dir / "gt_cloud.bt")


lidar_bt_path = str(save_dir / "lidar_cloud.bt")
resolution = 0.1
processPCDFolder(pcd_folder, resolution, lidar_bt_path)

octree = OcTree(lidar_bt_path)
logger.info(f"Resolution: {octree.getResolution()}")
logger.info(f"Size: {octree.size()}")
logger.info(f"Tree Depth: {octree.getTreeDepth()}")

occ_cloud_path = str(save_dir / "occ_cloud.pcd")
free_cloud_path = str(save_dir / "free_cloud.pcd")
convertOctreeToPointCloud(lidar_bt_path, free_cloud_path, occ_cloud_path)

filtered_cloud_path = str(save_dir / "filtered_cloud.pcd")
removeUnknownPoints(occ_cloud_path, octomap_bt_path, filtered_cloud_path)
