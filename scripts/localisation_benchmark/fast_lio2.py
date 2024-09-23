import libtmux

if __name__ == "__main__":
    
    path = '/media/mice85/blackdrive/oxford_spires_dataset/data/'
    sequence = '2024-03-14_blenhein_1'
    seq_path = path + sequence
    
    server = libtmux.Server()
    server.cmd('new-session', '-d', '-P', '-F#{session_id}')
    session = server.sessions[0]
    window_1 = session.new_window(attach=False, window_name="roscore")
    window_2 = session.new_window(attach=False, window_name="rosbag") 
    pane_1 = window_1.split()
    pane_2 = window_2.split()
    
    pane_1.send_keys('cd ~/oxford-lab/oxford_ws/')
    pane_1.send_keys('source devel/setup.bash')
    pane_1.send_keys('roslaunch fast_lio {}.launch'.format(sequence))
    
    pane_2.send_keys('cd {}/rosbag/'.format(seq_path))
    pane_2.send_keys('rosbag play *.bag --clock')

    # Check when the rosbag is "Done."
    while (1):
        pane_2.clear()
        cap_curr = pane_2.capture_pane()
        for line in cap_curr:
            if "[RUNNING]" in line:
                print(line)
        if 'Done.' in cap_curr:
            break

    # TODO: Copy to output folder

    # TODO: Transform to TUM format


    server.kill()