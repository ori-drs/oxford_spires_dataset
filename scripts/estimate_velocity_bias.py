"""Estimate IMU bias and velocity from GT poses using GTSAM batch optimization."""

import argparse
from pathlib import Path

import numpy as np

from oxspires_tools.lidar_undistortion.core import analyze_bias_random_walk, load_state_csv, make_T
from oxspires_tools.lidar_undistortion.gtsam_opt import (
    analyze_imu_chi2,
    compare_with_vilens,
    load_sensor_transforms,
    print_calib_comparison,
    run_optimization,
)
from oxspires_tools.lidar_undistortion.io import read_imu
from oxspires_tools.trajectory.file_interfaces.tum import TUMTrajReader

_DEFAULT_SENSOR_YAML = Path(__file__).parent.parent / "configs" / "sensor.yaml"


def get_args():
    parser = argparse.ArgumentParser(
        description="Estimate IMU bias/velocity from GT poses via GTSAM batch optimization."
    )
    parser.add_argument("--data_dir", type=Path, required=True, help="Root data directory")  # fmt: skip
    parser.add_argument("--imu_csv", type=Path, default=None, help="Path to imu_full.csv (default: data_dir/imu_full.csv)")  # fmt: skip
    parser.add_argument("--gt_tum", type=Path, default=None, help="Path to gt_tum.txt (default: data_dir/gt_tum.txt)")  # fmt: skip
    parser.add_argument("--state_csv", type=Path, default=None, help="Path to vilens_logs/state.csv for comparison/init")  # fmt: skip
    parser.add_argument("--sensor_yaml", type=Path, default=_DEFAULT_SENSOR_YAML, help="Path to sensor.yaml for T_BI and T_BL extrinsics")  # fmt: skip
    parser.add_argument("--acc_noise", type=float, default=0.001799)  # fmt: skip
    parser.add_argument("--gyr_noise", type=float, default=0.000257)  # fmt: skip
    parser.add_argument("--acc_bias_rw", type=float, default=2.69e-4)  # fmt: skip
    parser.add_argument("--gyr_bias_rw", type=float, default=1.57e-5)  # fmt: skip
    parser.add_argument("--label", type=str, default="set1", help="Label for first calibration set")  # fmt: skip
    parser.add_argument("--acc_noise_2", type=float, default=None)  # fmt: skip
    parser.add_argument("--gyr_noise_2", type=float, default=None)  # fmt: skip
    parser.add_argument("--acc_bias_rw_2", type=float, default=None)  # fmt: skip
    parser.add_argument("--gyr_bias_rw_2", type=float, default=None)  # fmt: skip
    parser.add_argument("--label_2", type=str, default="set2", help="Label for second calibration set")  # fmt: skip
    parser.add_argument("--output_csv", type=Path, default=None, help="Save result DataFrame to CSV")  # fmt: skip
    parser.add_argument("--gt_frame_offset", type=float, nargs=7, default=[0, 0, 0, 0, 0, 0, 1], metavar=("tx", "ty", "tz", "qx", "qy", "qz", "qw"), help="Pose of GT frame origin in world frame (default: identity)")  # fmt: skip
    return parser.parse_args()


def main():
    args = get_args()
    data_dir = args.data_dir

    imu_path = args.imu_csv or data_dir / "raw" / "imu.csv"
    gt_path = args.gt_tum or data_dir / "processed" / "trajectory" / "gt-tum.txt"
    state_path = args.state_csv or data_dir / "vilens_logs" / "state.csv"

    print(f"Loading sensor extrinsics: {args.sensor_yaml}")
    T_BI_mat, T_BI_list, T_BL_list = load_sensor_transforms(args.sensor_yaml)

    print(f"Loading IMU: {imu_path}")
    imu_df = read_imu(imu_path)

    print(f"Loading GT: {gt_path}")
    gt_traj = TUMTrajReader(str(gt_path)).read_file()
    print(f"  {gt_traj.num_poses} GT poses loaded")

    T_world_GT = make_T(args.gt_frame_offset)
    if not np.allclose(T_world_GT, np.eye(4)):
        print(f"  Applying GT frame offset: {args.gt_frame_offset}")
        gt_traj.transform(T_world_GT)

    state_df = None
    if state_path.exists():
        print(f"Loading state: {state_path}")
        state_df = load_state_csv(state_path)
    else:
        print(f"state.csv not found at {state_path}, using zero initial guesses")

    print(f"\nRunning optimisation ({args.label}) ...")
    graph, result, result_df, imu_factors = run_optimization(
        gt_traj,
        imu_df,
        args.acc_noise,
        args.gyr_noise,
        args.acc_bias_rw,
        args.gyr_bias_rw,
        state_df,
        T_BI_mat,
        T_BL_list,
    )
    print(result_df.head(10).to_string())

    if args.output_csv:
        result_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved results to {args.output_csv}")

    has_second = all(v is not None for v in [args.acc_noise_2, args.gyr_noise_2, args.acc_bias_rw_2, args.gyr_bias_rw_2])  # fmt: skip
    if has_second:
        print(f"\nRunning optimisation ({args.label_2}) ...")
        graph2, result2, result_df2, imu_factors2 = run_optimization(
            gt_traj,
            imu_df,
            args.acc_noise_2,
            args.gyr_noise_2,
            args.acc_bias_rw_2,
            args.gyr_bias_rw_2,
            state_df,
            T_BI_mat,
            T_BL_list,
        )
        params_a = {"acc_noise": args.acc_noise, "gyr_noise": args.gyr_noise, "acc_bias_rw": args.acc_bias_rw, "gyr_bias_rw": args.gyr_bias_rw}  # fmt: skip
        params_b = {"acc_noise": args.acc_noise_2, "gyr_noise": args.gyr_noise_2, "acc_bias_rw": args.acc_bias_rw_2, "gyr_bias_rw": args.gyr_bias_rw_2}  # fmt: skip
        chi2_a = analyze_imu_chi2(imu_factors, result)
        chi2_b = analyze_imu_chi2(imu_factors2, result2)
        rw_a = analyze_bias_random_walk(result_df, args.acc_bias_rw, args.gyr_bias_rw)
        rw_b = analyze_bias_random_walk(result_df2, args.acc_bias_rw_2, args.gyr_bias_rw_2)
        print_calib_comparison(args.label, args.label_2, params_a, params_b, chi2_a, chi2_b, rw_a, rw_b)

    if state_df is not None:
        compare_with_vilens(result_df, state_df, gt_traj, T_BI_list, T_BL_list)


if __name__ == "__main__":
    main()
