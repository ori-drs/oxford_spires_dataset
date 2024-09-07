from oxford_spires_utils.io import convert_e57_to_pcd

if __name__ == "__main__":
    e57_file_path = "/media/yifu/Samsung_T71/oxford_spires/2024-03-13-maths/gt/individual/Math Inst- 001.e57"
    output_pcd = "/home/yifu/workspace/oxford_spires_dataset/output.pcd"
    new_pcd = "/home/yifu/workspace/oxford_spires_dataset/output_new.pcd"
    convert_e57_to_pcd(e57_file_path, output_pcd)
