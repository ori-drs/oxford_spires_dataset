"""Undistort raw LiDAR scans using a pre-built dense trajectory (no per-scan IMU re-integration)."""

import argparse
from pathlib import Path

import evo.core.trajectory
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from oxspires_tools.dataset import OxfordSpiresDataset
from oxspires_tools.lidar_undistortion.core import PoseBuffer, make_T, undistort_cloud
from oxspires_tools.lidar_undistortion.gtsam_opt import build_dense_trajectory
from oxspires_tools.lidar_undistortion.io import read_imu, read_pcd_binary
from oxspires_tools.point_cloud import modify_pcd_viewpoint
from oxspires_tools.trajectory.file_interfaces.timestamp import TimeStamp
from oxspires_tools.trajectory.file_interfaces.tum import TUMTrajWriter

ACC_NOISE = 0.001799
GYR_NOISE = 0.000257
ACC_BIAS_RW = 2.69e-4
GYR_BIAS_RW = 1.57e-5

_DEFAULT_SENSOR_YAML = Path(__file__).parent.parent / "configs" / "sensor.yaml"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir", type=Path, required=True, help="Sequence root dir (contains raw/ and processed/)")  # fmt: skip
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for undistorted PCD files")  # fmt: skip
    parser.add_argument("--acc_noise", type=float, default=ACC_NOISE)  # fmt: skip
    parser.add_argument("--gyr_noise", type=float, default=GYR_NOISE)  # fmt: skip
    parser.add_argument("--acc_bias_rw", type=float, default=ACC_BIAS_RW)  # fmt: skip
    parser.add_argument("--gyr_bias_rw", type=float, default=GYR_BIAS_RW)  # fmt: skip
    parser.add_argument("--label", type=str, default="set1", help="Label for first calibration set")  # fmt: skip
    parser.add_argument("--acc_noise_2", type=float, default=None)  # fmt: skip
    parser.add_argument("--gyr_noise_2", type=float, default=None)  # fmt: skip
    parser.add_argument("--acc_bias_rw_2", type=float, default=None)  # fmt: skip
    parser.add_argument("--gyr_bias_rw_2", type=float, default=None)  # fmt: skip
    parser.add_argument("--label_2", type=str, default="set2", help="Label for second calibration set")  # fmt: skip
    parser.add_argument("--n_scans", type=int, default=-1, help="Process only first N scans (-1 = all)")  # fmt: skip
    parser.add_argument("--mode", choices=["slam", "raw", "image"], default="slam", help="slam: undistort SLAM keyframe clouds; raw: undistort all raw lidar clouds; image: undistort to image timestamps")  # fmt: skip
    parser.add_argument("--raw_cloud_tol_ms", type=float, default=None, help="Max time diff (ms) when matching raw clouds to timestamps (default: 5)")  # fmt: skip
    parser.add_argument("--image_dir", type=Path, default=None, help="Image folder for image mode (filenames are timestamps)")  # fmt: skip
    parser.add_argument("--max_img_lidar_diff_ms", type=float, default=25.0, help="Max time diff (ms) between image and nearest LiDAR cloud in image mode (default: 25)")  # fmt: skip
    parser.add_argument("--sensor_yaml", type=Path, default=_DEFAULT_SENSOR_YAML, help="Path to sensor.yaml for T_BI and T_BL extrinsics")  # fmt: skip
    parser.add_argument("--gt_frame_offset", type=float, nargs=7, default=[0, 0, 0, 0, 0, 0, 1], metavar=("tx", "ty", "tz", "qx", "qy", "qz", "qw"), help="Pose of GT frame origin in world frame (default: identity)")  # fmt: skip
    return parser.parse_args()


def main():
    args = get_args()
    dataset = OxfordSpiresDataset(args.seq_dir)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"Loading IMU: {dataset.get_filepath('raw_imu')}")
    imu_df = read_imu(dataset.get_filepath("raw_imu"))

    print(f"Loading GT poses: {dataset.get_filepath('gt_tum')}")
    gt_traj = dataset.load_gt_poses()
    print(f"  {gt_traj.num_poses} GT poses")

    T_world_GT = make_T(args.gt_frame_offset)
    if not np.allclose(T_world_GT, np.eye(4)):
        print(f"  Applying GT frame offset: {args.gt_frame_offset}")
        gt_traj.transform(T_world_GT)

    if args.mode == "slam":
        print(f"Loading SLAM poses: {dataset.get_filepath('slam_poses')}")
        slam_traj = dataset.load_slam_poses()
        print(f"  {slam_traj.num_poses} SLAM keyframes")

    # ── 2. GTSAM batch optimization ───────────────────────────────────────────
    from oxspires_tools.lidar_undistortion.core import analyze_bias_random_walk  # noqa: PLC0415
    from oxspires_tools.lidar_undistortion.gtsam_opt import (  # noqa: PLC0415
        analyze_imu_chi2,
        load_sensor_transforms,
        print_calib_comparison,
        run_optimization,
    )

    T_BI_mat, _T_BI_list, T_BL_list = load_sensor_transforms(args.sensor_yaml)

    print(f"\nRunning GTSAM batch optimization ({args.label}) ...")
    graph, result, result_df, imu_factors = run_optimization(
        gt_traj, imu_df, args.acc_noise, args.gyr_noise, args.acc_bias_rw, args.gyr_bias_rw, None, T_BI_mat, T_BL_list
    )
    print(f"  Optimization done. Sample bias_acc: {result_df[['ba_x', 'ba_y', 'ba_z']].mean().values}")
    print(f"  Sample bias_gyr: {result_df[['bg_x', 'bg_y', 'bg_z']].mean().values}")

    has_second = all(v is not None for v in [args.acc_noise_2, args.gyr_noise_2, args.acc_bias_rw_2, args.gyr_bias_rw_2])  # fmt: skip
    if has_second:
        print(f"\nRunning GTSAM batch optimization ({args.label_2}) ...")
        graph2, result2, result_df2, imu_factors2 = run_optimization(
            gt_traj,
            imu_df,
            args.acc_noise_2,
            args.gyr_noise_2,
            args.acc_bias_rw_2,
            args.gyr_bias_rw_2,
            None,
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

    gt_states = []
    for i in range(gt_traj.num_poses):
        ts = TimeStamp(t_float128=gt_traj.timestamps[i])
        ts_ns = ts.sec * 10**9 + ts.nsec
        row = result_df.iloc[i]
        gt_states.append(
            (
                ts_ns,
                gt_traj.poses_se3[i],
                np.array([row.vel_x, row.vel_y, row.vel_z]),
                np.array([row.ba_x, row.ba_y, row.ba_z]),
                np.array([row.bg_x, row.bg_y, row.bg_z]),
            )
        )

    # ── 2b. Build dense trajectory and global PoseBuffer ─────────────────────
    bias_csv_path = output_dir / "bias_velocity.csv"
    result_df.to_csv(bias_csv_path, index=False)
    print(f"  Saved bias/velocity CSV: {bias_csv_path}")

    dense_ts, dense_poses_WB = build_dense_trajectory(gt_states, imu_df, T_BI_mat)
    dense_ts_arr = np.array(dense_ts, dtype=np.int64)

    positions = np.array([T[:3, 3] for T in dense_poses_WB])
    q_xyzw = Rotation.from_matrix(np.stack([T[:3, :3] for T in dense_poses_WB])).as_quat()
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    timestamps_float128 = np.array([TimeStamp(sec=int(t // 10**9), nsec=int(t % 10**9)).t_float128 for t in dense_ts])
    dense_traj = evo.core.trajectory.PoseTrajectory3D(positions, q_wxyz, timestamps=timestamps_float128)
    tum_path = output_dir / "dense_trajectory_tum.txt"
    TUMTrajWriter(str(tum_path)).write_file(dense_traj)
    print(f"  Saved dense trajectory: {tum_path}")

    # Convert dense WB poses to WL frame for undistortion
    T_BL_mat = make_T(T_BL_list)
    dense_poses_WL = [T @ T_BL_mat for T in dense_poses_WB]
    pose_buffer_WL = PoseBuffer(dense_ts, dense_poses_WL)
    print(f"  Built global PoseBuffer with {len(dense_ts)} poses")

    # ── 3. Build scan list ────────────────────────────────────────────────────
    if args.mode == "slam":
        n_total = slam_traj.num_poses if args.n_scans < 0 else args.n_scans
        scan_items = []
        for i in range(n_total):
            slam_ts = TimeStamp(t_float128=slam_traj.timestamps[i])
            slam_ts_ns = slam_ts.sec * 10**9 + slam_ts.nsec
            scan_items.append((slam_ts_ns, None, slam_traj.poses_se3[i]))
    elif args.mode == "raw":
        raw_lidar_dir = dataset.get_filepath("raw_lidar")
        if not raw_lidar_dir.exists():
            print(f"WARNING: raw lidar dir does not exist: {raw_lidar_dir}")
        raw_paths = sorted(raw_lidar_dir.glob("*.pcd")) if raw_lidar_dir.exists() else []
        if not raw_paths:
            print(f"WARNING: no PCD files found in {raw_lidar_dir}")
        if args.n_scans >= 0:
            raw_paths = raw_paths[: args.n_scans]
        scan_items = []
        for p in raw_paths:
            try:
                ts = TimeStamp(t_string=p.stem)
                scan_items.append((ts.sec * 10**9 + ts.nsec, p, None))
            except (AssertionError, ValueError):
                continue
    elif args.mode == "image":
        if args.image_dir is None:
            raise ValueError("--image_dir is required for image mode")
        img_tol_ns = int(args.max_img_lidar_diff_ms * 1e6)
        img_paths = sorted(args.image_dir.glob("*.jpg")) + sorted(args.image_dir.glob("*.png"))
        img_paths = sorted(img_paths, key=lambda p: p.stem)
        if args.n_scans >= 0:
            img_paths = img_paths[: args.n_scans]
        scan_items = []
        img_lidar_pairs = []
        skipped_img = 0
        for img_path in img_paths:
            try:
                ts = TimeStamp(t_string=img_path.stem)
                img_ts_ns = ts.sec * 10**9 + ts.nsec
            except (AssertionError, ValueError):
                continue
            raw_path = dataset.find_raw_cloud(img_ts_ns, tol_ns=img_tol_ns)
            if raw_path is None:
                skipped_img += 1
                continue
            scan_items.append((img_ts_ns, raw_path, None))
            img_lidar_pairs.append((img_path, raw_path))
        print(f"  Image mode: {len(scan_items)} matched, {skipped_img} skipped (no LiDAR within {args.max_img_lidar_diff_ms} ms)")  # fmt: skip
        pairs_txt_path = output_dir / "image_lidar_pairs.txt"
        pairs_txt_path.write_text("\n".join(f"{img.name}    {pcd.name}" for img, pcd in img_lidar_pairs))
        print(f"  Saved image-lidar pairs: {pairs_txt_path}")

    # ── 4. Undistort ──────────────────────────────────────────────────────────
    default_tol_ms = 5.0
    raw_cloud_tol_ns = int((args.raw_cloud_tol_ms if args.raw_cloud_tol_ms is not None else default_tol_ms) * 1e6)
    errors = []
    skipped = 0
    pbar = tqdm(total=len(scan_items), desc=f"Undistorting scans [{args.mode}]")

    for scan_ts_ns, given_raw_path, T_WB_viewpoint in scan_items:
        ts_obj = TimeStamp(sec=scan_ts_ns // 10**9, nsec=scan_ts_ns % 10**9)
        out_path = output_dir / f"cloud_{ts_obj.sec}_{ts_obj.nsec:09d}.pcd"

        if out_path.exists():
            pbar.update(1)
            continue

        raw_path = (
            given_raw_path
            if given_raw_path is not None
            else dataset.find_raw_cloud(scan_ts_ns, tol_ns=raw_cloud_tol_ns)
        )
        if raw_path is None:
            errors.append(f"No raw cloud for ts={scan_ts_ns}")
            skipped += 1
            pbar.update(1)
            continue

        try:
            raw_cloud, _ = read_pcd_binary(raw_path)
            corrected_xyz = undistort_cloud(raw_cloud, pose_buffer_WL, scan_ts_ns)
        except Exception as e:
            errors.append(f"ts={scan_ts_ns}: {e}")
            skipped += 1
            pbar.update(1)
            continue

        body_xyz = (T_BL_mat[:3, :3] @ corrected_xyz.T + T_BL_mat[:3, 3:4]).T

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(body_xyz.astype(np.float64))
        o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False, compressed=False)

        dense_idx = np.clip(np.searchsorted(dense_ts_arr, scan_ts_ns, side="right") - 1, 0, len(dense_poses_WB) - 1)
        if T_WB_viewpoint is None:
            T_WB_viewpoint = dense_poses_WB[dense_idx]
        t = T_WB_viewpoint[:3, 3]
        q_xyzw = Rotation.from_matrix(T_WB_viewpoint[:3, :3]).as_quat()
        viewpoint = np.array([t[0], t[1], t[2], q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        modify_pcd_viewpoint(str(out_path), str(out_path), viewpoint)

        pbar.update(1)

    pbar.close()

    # ── 5. Summary ────────────────────────────────────────────────────────────
    saved = len(scan_items) - skipped
    print(f"\nDone. Saved {saved}/{len(scan_items)} clouds to {output_dir}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
