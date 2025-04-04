import logging
import os
import re
import sys
import time

import libtmux
import numpy as np
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface


def run_fast_lio_slam (path_to_rosbag, path_to_output):
    
    server = libtmux.Server()
    server.cmd('new-session', '-d', '-P', '-F#{session_id}')
    session = server.sessions[0]
    window_1 = session.new_window(attach=False, window_name="launch fast_lio_slam")
    window_2 = session.new_window(attach=False, window_name="launch aloam_velodyne")
    window_3 = session.new_window(attach=False, window_name="rosbag") 
    pane_1 = window_1.panes.get()
    pane_2 = window_2.panes.get()
    pane_3 = window_3.panes.get()

    pane_2.send_keys('cd ~/oxford-lab/lidar_slam_ws/')
    pane_2.send_keys('source devel/setup.bash')
    pane_2.send_keys('roslaunch aloam_velodyne aloam_hesai.launch arg_save_directory:={}'.format(path_to_output))
    print("SC-PGO launched!! - mapping_hesai.launch")

    time.sleep(5)

    pane_1.send_keys('cd ~/oxford-lab/lidar_slam_ws/')
    pane_1.send_keys('source devel/setup.bash')
    pane_1.send_keys('roslaunch fast_lio mapping_hesai.launch')
    print("Fast-LIO-SLAM launched!! - mapping_hesai.launch")

    time.sleep(5)
    
    pane_3.send_keys('cd {}'.format(path_to_rosbag))
    pane_3.send_keys('rosbag play *.bag --clock')
    print("Starting rosbag files in {}".format(path_to_rosbag))

    # Check if the rosbag is "Done."
    t  = time.time()
    flag_end = False
    while (1):
        pane_3.clear()
        time.sleep(1)
        cap_curr = pane_3.capture_pane()
        for line in cap_curr:
            if "[RUNNING]" in line:
                t  = time.time()
                print(line)
                sys.stdout.write("\033[F")
            if 'Done.' in line:
                t  = time.time()
                flag_end = True
                sys.stdout.write("\033[K")
                print('Done.') 
                break
        if flag_end:
            break
        if (time.time() - t) > 60.0: # To avoid infinite
            print('Timeout!!')
            break

    print("PROCESS FINISHED!!")
    print("Output in: {}optimized_poses.txt".format(path_to_output))
    print("*********************************************************")

    server.kill()


def convert_to_tum (path_to_output, path_to_sec):
    
    print("Convert from KITTI format to TUM!!")

    pose_path = file_interface.read_kitti_poses_file(path_to_output + "optimized_poses.txt")
    raw_timestamps_mat = file_interface.csv_read_matrix(path_to_output + "times.txt")
    error_msg = ("timestamp file must have one column of timestamps and same number of rows as the KITTI poses file")
    if len(raw_timestamps_mat) > 0 and len(raw_timestamps_mat[0]) != 1 or len(raw_timestamps_mat) != pose_path.num_poses:
        raise file_interface.FileInterfaceException(error_msg)
    try:
        timestamps_mat = np.array(raw_timestamps_mat).astype(float)
    except ValueError:
        raise file_interface.FileInterfaceException(error_msg)
    tum_traj = PoseTrajectory3D(poses_se3=pose_path.poses_se3, timestamps=timestamps_mat)

    file_interface.write_tum_trajectory_file(path_to_sec + "/output_slam" + "/fast_lio_tum.txt", tum_traj)

    print("PROCESS FINISHED!!")
    print("Output in: {}/output_slam/fast_lio_tum.txt".format(path_to_sec))
    print("*********************************************************")