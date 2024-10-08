import re
import sys

sys.path.append("/home/oxford_spires_dataset")

from oxford_spires_utils.bash_command import run_command

def evaluation_ape_rmse (path_gt, path_traj):
    
    output = run_command("evo_ape tum {} {} --align_origin --t_max_diff 0.01".format(path_gt, path_traj), print_output=False)
    
    for line in output.stdout:
        print(line, end="")
        if "rmse" in line:
            numbers = re.findall('\d+\.\d+|\d+', line)
            rmse = numbers[0]
    
    print("RMSE added to log: {}".format(rmse))

    return rmse

if __name__ == "__main__":

    path = '/home/oxford_spires_dataset/data/' # TODO: get path from arg an define folders in the future class.
    sequence = '2024-03-20-christ-church-05' # TODO: get sequence from arg an define folders in the future class.
    sub_dir = "/output_slam/"
    file_name = "vilens_poses_tum.txt"
    file_name_gt = "/ground_truth_traj/gt_lidar.txt"

    path_gt = path + sequence + file_name_gt
    path_traj = path + sequence + sub_dir + file_name

    print("RUNNING VILENS EVALUATION ...")
    rmse = evaluation_ape_rmse (path_gt, path_traj)

    file_name = "hba_poses_tum.txt"
    path_traj = path + sequence + sub_dir + file_name
    print("RUNNING HBA EVALUATION ...")
    rmse = evaluation_ape_rmse (path_gt, path_traj)
    