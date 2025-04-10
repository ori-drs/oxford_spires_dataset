import logging
import os
import re
import sys

sys.path.append("/home/mice85/oxford-lab/labrobotica/algorithms/oxford_spires_dataset")

from oxford_spires_utils.bash_command import run_command
from oxford_spires_utils.trajectory.file_interfaces import NeRFTrajReader, TUMTrajWriter
from oxford_spires_utils.trajectory.json_handler import JsonHandler


def convert_to_tum(path_to_output, path_to_sec):
    print("Convert from JSON format to TUM!!")

    # Remove other cameras
    input_json_path = path_to_output + "transforms_colmap_scaled.json"
    handler = JsonHandler(input_json_path)
    handler.filter_folder("images/alphasense_driver_ros_cam0_debayered_image_compressed")
    handler.save_json(path_to_output + "transforms_colmap_single.json")

    escaled_colmap_traj = NeRFTrajReader(path_to_output + "transforms_colmap_single.json").read_file()
    TUMTrajWriter("{}/output/colmap-tum.txt".format(path_to_sec)).write_file(escaled_colmap_traj)

    print("PROCESS FINISHED!!")
    print("Output in: {}/output/colmap-tum.txt".format(path_to_sec))
    print("*********************************************************")


def get_sec_list(dataset_dir, flag_is_all=True):
    if flag_is_all:
        list_sec = os.listdir(dataset_dir)
    else:
        list_sec = [
            "2024-03-12-keble-college-02",
            "2024-03-12-keble-college-03",
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
            "2024-05-20-bodleian-library-05",
        ]
    return list_sec


def evaluation_ape_rmse(path_to_gt, path_traj, dataset_dir, method):
    print("RUNNING VILENS EVALUATION ...")

    output = run_command(
        "evo_ape tum {} {} --align --t_max_diff 0.01".format(path_to_gt, path_traj), print_output=False
    )

    rmse = -1
    for line in output.stdout:
        print(line, end="")
        if "rmse" in line:
            numbers = re.findall("\d+\.\d+|\d+", line)
            rmse = numbers[0]

    logging.basicConfig(filename=dataset_dir + "results.log", filemode="a", level=logging.INFO)
    logging.info(path_to_sec)
    logging.info("APE - RMSE result ({}): {}".format(method, rmse))
    print("RMSE added to log: {}".format(rmse))

    return rmse


def create_output_folder(path_to_sec):
    path_to_output = path_to_sec + "/output" + "/colmap/"
    # if not os.path.exists(path_to_output):
    #         os.makedirs(path_to_output)
    return path_to_output


if __name__ == "__main__":
    # -------------------------------------------------------------------------------- #
    # TODO: get path from arg an define folders in the future class.
    package_dir = "/home/mice85/oxford-lab/labrobotica/algorithms/oxford_spires_dataset"
    dataset_dir = "/media/mice85/blackdrive1/oxford_spires_dataset/data/"
    flag_is_all = False
    # -------------------------------------------------------------------------------- #

    list_sec = get_sec_list(dataset_dir, flag_is_all)

    print("Total sequence folders: " + str(len(list_sec)))

    # Print list of sequences
    for sec in list_sec:
        path_to_sec = dataset_dir + sec
        path_to_gt = path_to_sec + "/processed/trajectory/gt-tum.txt"

        path_to_output = create_output_folder(path_to_sec)

        convert_to_tum(path_to_output, path_to_sec)

        path_traj = path_to_sec + "/output" + "/colmap-tum.txt"
        rmse = evaluation_ape_rmse(path_to_gt, path_traj, dataset_dir, "COLMAP")
