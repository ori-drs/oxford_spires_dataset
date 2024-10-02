import time

import libtmux

if __name__ == "__main__":

    # TODO: put it in def an get path from arg.
    
    path = '/home/oxford_spires_dataset/data/'
    sequence = '2024-03-18-chrst-chrch-01'
    seq_path = path + sequence
    
    server = libtmux.Server()
    server.cmd('new-session', '-d', '-P', '-F#{session_id}')
    session = server.sessions[0]
    window_1 = session.new_window(attach=False, window_name="launch fast_lio_slam")
    pane_1 = window_1.panes.get()
    pane_1.send_keys('cd /home/catkin_ws/')
    pane_1.send_keys('source devel/setup.bash')
    pane_1.send_keys('roslaunch hba {}.launch'.format(sequence))


    # Check if the process is "iteration complete"
    print("HBA launched!! - {}.launch".format(sequence))
    flag = False
    while (1):
        pane_1.clear()
        time.sleep(1)
        cap_curr = pane_1.capture_pane()
        # print(cap_curr)
        for line in cap_curr:
            if "Residual" in line:
                print(line)
            elif "pgo complete" in line:
                print(line) 
                flag = True
        if flag:
            break
    
    time.sleep(3)
    print("PROCESS FINISHED!! Execute this script 3 times to replicate paper results")
    print("Output in: {}/output_slam/hba/hba_poses_robotics.csv".format(sequence))

    # TODO: Copy to output folder

    # TODO: Transform to TUM format


    server.kill()