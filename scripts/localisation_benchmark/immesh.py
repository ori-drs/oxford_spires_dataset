import os
import sys
import time

import libtmux

sys.path.append("/home/mice85/oxford-lab/labrobotica/algorithms/oxford_spires_dataset")

from oxford_spires_utils.bash_command import run_command


def run_immesh (path_to_rosbag, path_to_output):
    
    server = libtmux.Server()
    server.cmd('new-session', '-d', '-P', '-F#{session_id}')
    session = server.sessions[0]
    window_1 = session.new_window(attach=False, window_name="launch immesh")
    window_2 = session.new_window(attach=False, window_name="rosbag play") 
    window_3 = session.new_window(attach=False, window_name="rosbag record")
    pane_1 = window_1.panes.get()
    pane_2 = window_2.panes.get()
    pane_3 = window_3.panes.get()
    
    pane_1.send_keys('cd ~/oxford-lab/lidar_slam_ws/')
    pane_1.send_keys('source devel/setup.bash')
    pane_1.send_keys('roslaunch ImMesh mapping_spire.launch')
    print("ImMesh launched!! - roslaunch ImMesh mapping_spire.launch")
    
    pane_2.send_keys('cd {}'.format(path_to_rosbag))
    pane_2.send_keys('rosbag play *.bag --clock')
    print("Starting rosbag files in {}".format(path_to_rosbag))

    pane_3.send_keys('cd {}'.format(path_to_output))
    pane_3.send_keys('rosbag record -O immesh_path.bag /path __name:=immesh_path')
    print("Rosbag record in {}".format(path_to_output))
    

    # Check if the rosbag is "Done."
    t  = time.time()
    while (1):
        pane_2.clear()
        time.sleep(1)
        cap_curr = pane_2.capture_pane()
        for line in cap_curr:
            if "[RUNNING]" in line:
                print(line)
                sys.stdout.write("\033[F")
            if 'Done.' in line:
                sys.stdout.write("\033[K")
                print('Done.') 
                break
        if 'Done.' in cap_curr: # To avoid missings
            sys.stdout.write("\033[K")
            print('Done.') 
            break
        if (time.time() - t) > 1200.0: # To avoid infinite
            print('Timeout!!')
            break
    
    run_command("rosnode kill /immesh_path", print_output=True)

    time.sleep(3)
    
    if os.path.exists("{}/immesh_path.bag.active".format(path_to_output)):
        print("{}/immesh_path.bag.active".format(path_to_output))
        print(os.path.exists("{}/immesh_path.bag.active".format(path_to_output)))

        run_command("rosbag reindex {}/immesh_path.bag.active".format(path_to_output), print_output=True)

        old_file = os.path.join(path_to_output, "immesh_path.bag.active")
        new_file = os.path.join(path_to_output, "immesh_path.bag")
        os.rename(old_file, new_file)
        os.remove("{}/immesh_path.bag.orig.active".format(path_to_output))

    print("PROCESS FINISHED!!")
    print("Output in: {}/immesh_path.bag".format(path_to_output))
    print("*********************************************************")

    server.kill()

def convert_to_tum (path_to_output):
    
    server = libtmux.Server()
    server.cmd('new-session', '-d', '-P', '-F#{session_id}')
    session = server.sessions[0]
    window_1 = session.new_window(attach=False, window_name="launch bag_to_traj")
    window_2 = session.new_window(attach=False, window_name="rosbag play") 
    pane_1 = window_1.panes.get()
    pane_2 = window_2.panes.get()
    
    pane_1.send_keys('cd ~/oxford-lab/oxford_ws/')
    pane_1.send_keys('source devel/setup.bash')
    pane_1.send_keys('roslaunch bag_to_traj bag_to_traj.launch arg_outfile_path:={}/immesh_tum.txt'.format(path_to_output))
    print("bag_to_traj launched!! - roslaunch bag_to_traj bag_to_traj.launch")
    
    pane_2.send_keys('cd {}'.format(path_to_output))
    pane_2.send_keys('rosbag play *.bag --clock')
    print("Starting rosbag files in {}".format(path_to_output))

    # Check if the rosbag is "Done."
    t  = time.time()
    while (1):
        pane_2.clear()
        time.sleep(1)
        cap_curr = pane_2.capture_pane()
        for line in cap_curr:
            if "[RUNNING]" in line:
                print(line)
                sys.stdout.write("\033[F")
            if 'Done.' in line:
                sys.stdout.write("\033[K")
                print('Done.') 
                break
        if 'Done.' in cap_curr: # To avoid missings
            sys.stdout.write("\033[K")
            print('Done.') 
            break
        if (time.time() - t) > 1200.0: # To avoid infinite
            print('Timeout!!')
            break

    print("PROCESS FINISHED!!")
    print("Output in: {}/ImMesh_tum.txt".format(path_to_output))
    print("*********************************************************")

    server.kill()

def create_output_folder (path_to_sec):
    path_to_output = path_to_sec + "/output_slam" + "/ImMesh_raw_output"
    if not os.path.exists(path_to_output):
            os.makedirs(path_to_output)
    return path_to_output

def get_sec_list (dataset_dir, flag_is_all=True):
    if flag_is_all:
        list_sec = os.listdir(dataset_dir)
    else:
        list_sec = ["2024-03-12-keble-college-03",
                    "2024-03-12-keble-college-04",
                    "2024-03-12-keble-college-05",
                    "2024-03-13-observatory-quarter-01",
                    "2024-03-13-observatory-quarter-02",
                    "2024-03-14-blenheim-palace-01",
                    "2024-03-14-blenheim-palace-02",
                    "2024-03-14-blenheim-palace-05",
                    "2024-03-18-christ-church-01",
                    "2024-03-18-christ-church-02",
                    "2024-03-18-christ-church-03",
                    "2024-03-20-christ-church-05",
                    "2024-05-20-bodleian-library-02",
                    "2024-05-20-bodleian-library-03",
                    "2024-05-20-bodleian-library-04",
                    "2024-05-20-bodleian-library-05"
                    ]
    return list_sec

if __name__ == "__main__":

    dataset_dir = '/media/mice85/blackdrive1/oxford_spires_dataset/data/' # TODO: get path from arg an define folders in the future class.
    flag_is_all = False

    list_sec = get_sec_list (dataset_dir, flag_is_all)
    
    print('Total sequence folders: ' + str(len(list_sec)))
    
    # Print list of sequences
    for sec in list_sec:

        path_to_sec = dataset_dir + sec
        path_to_rosbag = path_to_sec + "/rosbag/"

        path_to_output = create_output_folder (path_to_sec)

        run_immesh (path_to_rosbag, path_to_output)

        time.sleep(5)

        convert_to_tum (path_to_output)

        time.sleep(5)