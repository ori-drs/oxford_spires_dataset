import sys
import time

import libtmux

if __name__ == "__main__":

    # TODO: put it in def an get path from arg.
    
    path = '/home/oxford_spires_dataset/data/'
    sequence = '2024-03-18_chrst_chrch_01'
    seq_path = path + sequence
    
    server = libtmux.Server()
    server.cmd('new-session', '-d', '-P', '-F#{session_id}')
    session = server.sessions[0]
    window_1 = session.new_window(attach=False, window_name="launch fast_lio_slam")
    window_2 = session.new_window(attach=False, window_name="launch aloam_velodyne")
    window_3 = session.new_window(attach=False, window_name="rosbag") 
    pane_1 = window_1.panes.get()
    pane_2 = window_2.panes.get()
    pane_3 = window_3.panes.get()

    pane_2.send_keys('cd /home/catkin_ws/')
    pane_2.send_keys('source devel/setup.bash')
    pane_2.send_keys('roslaunch aloam_velodyne {}.launch'.format(sequence))

    pane_1.send_keys('cd /home/catkin_ws/')
    pane_1.send_keys('source devel/setup.bash')
    pane_1.send_keys('roslaunch fast_lio mapping_hesai.launch')
    
    pane_3.send_keys('cd {}/rosbag/'.format(seq_path))
    pane_3.send_keys('rosbag play *.bag --clock')

    # Check if the rosbag is "Done."
    print("Fast-LIO-SLAM launched!! - {}.launch".format(sequence))
    print("Starting rosbag files in {}/rosbag/ ...".format(seq_path))
    while (1):
        pane_3.clear()
        time.sleep(1)
        cap_curr = pane_3.capture_pane()
        for line in cap_curr:
            if "[RUNNING]" in line:
                print(line)
                sys.stdout.write("\033[F")
        if 'Done.' in cap_curr:
            sys.stdout.write("\033[K")
            print('Done.') 
            break

    print("PROCESS FINISHED!!")
    print("Output in: {}/output_loc/fast_lio_slam/optimized_poses.txt".format(sequence))

    # TODO: Copy to output folder

    # TODO: Transform to TUM format


    server.kill()