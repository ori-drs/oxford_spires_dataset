from pathlib import Path

from oxford_spires_utils.io import convert_e57_to_pcd


def convert_e57_folder_to_pcd_folder(e57_folder, pcd_folder):
    Path(pcd_folder).mkdir(parents=True, exist_ok=True)
    e57_files = list(Path(e57_folder).glob("*.e57"))
    for e57_file in e57_files:
        pcd_file = Path(pcd_folder) / (e57_file.stem + ".pcd")
        print(f"Converting {e57_file} to {pcd_file}")
        convert_e57_to_pcd(e57_file, pcd_file, pcd_lib="pypcd4")


if __name__ == "__main__":
    # e57_file_path = "/media/yifu/Samsung_T71/oxford_spires/2024-03-13-maths/gt/individual/Math Inst- 001.e57"
    # output_pcd = "/home/yifu/workspace/oxford_spires_dataset/output.pcd"
    # new_pcd = "/home/yifu/workspace/oxford_spires_dataset/output_new.pcd"
    # convert_e57_to_pcd(e57_file_path, output_pcd)
    e57_folder = "/media/yifu/Samsung_T71/oxford_spires/2024-03-13-maths/gt/individual"
    pcd_folder = "/media/yifu/Samsung_T71/oxford_spires/2024-03-13-maths/gt/individual_pcd"
    convert_e57_folder_to_pcd_folder(e57_folder, pcd_folder)
