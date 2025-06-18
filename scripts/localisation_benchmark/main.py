import argparse
import os
import re
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

sys.path.append("/home/miguelangel/git/oxford_spires_dataset")

from oxford_spires_utils.bash_command import run_command

# TODO: loggin
# def setup_logging():

class LocalisationBenchmark:
    def __init__(self, loc_config):
        self.loc_config = loc_config
        self.list_sec = []
        self.gt_file = "/processed/trajectory/gt-tum.txt"
        self.vilens_file = "/processed/trajectory/vilens-slam-tum.txt"
        self.hf_repo_id = "ori-drs/oxford_spires_dataset"
    
    def process_fast_lio_slam (self):
        print("run_fast_lio_slam here")
        print("convert_to_tum here")
    
    def evaluation_ape_rmse (self, path_to_gt, path_traj):
        output = run_command("evo_ape tum {} {} --align --t_max_diff 0.01".format(path_to_gt, path_traj), print_output=False)
        
        rmse = -1
        for line in output.stdout:
            print(line, end="")
            if "rmse" in line:
                numbers = re.findall('\d+\.\d+|\d+', line)
                rmse = numbers[0]
        
        # logging.basicConfig(filename=dataset_dir + "results.log", filemode="a", level=logging.INFO)
        # logging.info(path_to_sec)
        # logging.info("APE - RMSE result ({}): {}".format(method, rmse))
        print("RMSE added to log: {}".format(rmse))

        return rmse

    def get_sec_list(self):
        if self.loc_config["flag_is_all"]:
            self.list_sec = os.listdir(self.loc_config["dataset_folder"])
        else:
            self.list_sec = self.loc_config["list_sec"]

    def download_data(self):
        for sec in self.list_sec:
            snapshot_download(
                repo_id = self.hf_repo_id,
                allow_patterns = sec + "/*",
                local_dir=self.loc_config["dataset_folder"],
                repo_type="dataset",
                use_auth_token=False,
            )

def create_output_folder (path_to_sec, method_folder):
    path_to_output = path_to_sec + "/output_slam" + method_folder
    if not os.path.exists(path_to_output):
            os.makedirs(path_to_output)
    return path_to_output


def get_args():
    parser = argparse.ArgumentParser(description="Localisation Benchmark")
    default_loc_config_file = Path(__file__).parent.parent.parent / "config" / "loc_benchmark.yaml"
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(default_loc_config_file),
    )
    return parser.parse_args()

if __name__ == "__main__":
    # setup_logging()
    loc_config_file = get_args().config_file
    # logger.info("Starting Reconstruction Benchmark")
    with open(loc_config_file, "r") as f:
        loc_config = yaml.safe_load(f)["localisation_benchmark"]

    loc_benchmark = LocalisationBenchmark(loc_config)
    loc_benchmark.get_sec_list()
    print('Total sequence folders: ' + str(len(loc_benchmark.list_sec)))

    if loc_benchmark.loc_config["flag_download_data"]:
        loc_benchmark.download_data()

    for sec in loc_benchmark.list_sec:

        path_to_sec = loc_benchmark.loc_config["dataset_folder"] + sec
        path_to_gt = path_to_sec + loc_benchmark.gt_file
        path_to_vilens = path_to_sec + loc_benchmark.vilens_file

        print("SEQUENCE: " + path_to_sec)
        print("GT: " + path_to_gt)


        loc_benchmark.evaluation_ape_rmse (path_to_gt, path_to_vilens)

        if loc_benchmark.loc_config["run_fast_lio_slam"]:
            print("RUNING FAST-LIO-SLAM")
            # path_to_output = create_output_folder (path_to_sec, "/fastlio_raw_output/")
            loc_benchmark.process_fast_lio_slam
        else:
            print("EVALUATING FILES PROVIDED")