"""Undistort raw LiDAR scans for a Spires sequence using GTSAM-estimated velocity/bias from GT."""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evo.core.trajectory
from estiamte_velocity_bias import (
    ACC_BIAS_RW,
    ACC_NOISE,
    GYR_BIAS_RW,
    GYR_NOISE,
    build_factor_graph,
    build_preint_params,
    extract_results,
    optimize,
)
from nerf_data_pipeline.dataset.oxspires_dataset import OxfordSpiresDataset
from nerf_data_pipeline.lidar_undistortion import (
    PoseBuffer,
    integrate_imu,
    make_T,
    read_imu,
    read_pcd_binary,
    undistort_cloud,
)

from oxspires_tools.point_cloud import modify_pcd_viewpoint
from oxspires_tools.trajectory.file_interfaces import TimeStamp, TUMTrajWriter

T_BL_DEFAULT = [0.0, 0.0, 0.124, 0.0, 0.0, 1.0, 0.0]
T_BI_DEFAULT = [-0.018, 0.006, 0.058, 0.0, 0.0, 0.707, 0.707]


def build_dense_trajectory(gt_states: list, imu_df: pd.DataFrame, T_BI_mat: np.ndarray):
    """Integrate IMU segment-by-segment between GT keyframes to produce a dense T_WB trajectory."""
    T_IB = np.linalg.inv(T_BI_mat)
    dense_ts = []
    dense_poses_WB = []
    for i in range(len(gt_states) - 1):
        ts_ns_i, T_WB_i, vel_i, bias_acc_i, bias_gyr_i = gt_states[i]
        ts_ns_j = gt_states[i + 1][0]
        imu_window = imu_df[(imu_df["timestamp_ns"] >= ts_ns_i) & (imu_df["timestamp_ns"] <= ts_ns_j)].reset_index(
            drop=True
        )
        if len(imu_window) < 2:
            continue
        timestamps_ns, poses_WI = integrate_imu(
            imu_window,
            initial_velocity=vel_i,
            initial_pose_WI=T_WB_i @ T_BI_mat,
            bias_acc=bias_acc_i,
            bias_gyr=bias_gyr_i,
        )
        for t, T_WI in zip(timestamps_ns, poses_WI):
            if dense_ts and t == dense_ts[-1]:
                print(
                    f"Warning: {t} in dense trajectory is exactly on a GT keyframe timestamp; skipping to avoid duplicate"
                )
                continue  # skip when IMU sample coincides exactly with a GT keyframe timestamp
            dense_ts.append(t)
            dense_poses_WB.append(T_WI @ T_IB)
    return dense_ts, dense_poses_WB


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir", type=Path, required=True, help="Sequence root dir (contains raw/ and processed/)")  # fmt: skip
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for undistorted PCD files")  # fmt: skip
    parser.add_argument("--acc_noise", type=float, default=ACC_NOISE)  # fmt: skip
    parser.add_argument("--gyr_noise", type=float, default=GYR_NOISE)  # fmt: skip
    parser.add_argument("--acc_bias_rw", type=float, default=ACC_BIAS_RW)  # fmt: skip
    parser.add_argument("--gyr_bias_rw", type=float, default=GYR_BIAS_RW)  # fmt: skip
    parser.add_argument("--imu_buffer_ms", type=float, default=50.0, help="Extra IMU window beyond scan end (ms)")  # fmt: skip
    parser.add_argument("--n_scans", type=int, default=-1, help="Process only first N scans (-1 = all)")  # fmt: skip
    parser.add_argument("--mode", choices=["slam", "raw", "image"], default="slam", help="slam: undistort SLAM keyframe clouds; raw: undistort all raw lidar clouds; image: undistort to image timestamps")  # fmt: skip
    parser.add_argument("--raw_cloud_tol_ms", type=float, default=None, help="Max time diff (ms) when matching raw clouds to timestamps (default: 5)")  # fmt: skip
    parser.add_argument("--image_dir", type=Path, default=None, help="Image folder for image mode (filenames are timestamps)")  # fmt: skip
    parser.add_argument("--max_img_lidar_diff_ms", type=float, default=25.0, help="Max time diff (ms) between image and nearest LiDAR cloud in image mode (default: 25)")  # fmt: skip
    parser.add_argument("--gt_frame_offset", type=float, nargs=7, default=[0, 0, 0, 0, 0, 0, 1], metavar=("tx", "ty", "tz", "qx", "qy", "qz", "qw"), help="Pose of GT frame origin in world frame (default: identity)")  # fmt: skip
    return parser.parse_args()


def undistort_one_scan(
    raw_cloud: np.ndarray,
    imu_df: pd.DataFrame,
    gt_ts_ns: int,
    T_WB_gt: np.ndarray,
    velocity: np.ndarray,
    bias_acc: np.ndarray,
    bias_gyr: np.ndarray,
    T_BL: list,
    T_BI: list,
    desired_ns: int,
    buffer_ns: int,
) -> np.ndarray:
    """Undistort raw cloud to desired_ns. Returns xyz in lidar frame."""
    T_BL_mat = make_T(T_BL)
    T_BI_mat = make_T(T_BI)
    T_LI = np.linalg.inv(T_BL_mat) @ T_BI_mat
    T_IL = np.linalg.inv(T_LI)

    scan_end_ns = int(raw_cloud["timestamp"].max() * 1e9)

    initial_pose_WI = T_WB_gt @ T_BI_mat

    imu_window = imu_df[
        (imu_df["timestamp_ns"] >= gt_ts_ns) & (imu_df["timestamp_ns"] <= scan_end_ns + buffer_ns)
    ].reset_index(drop=True)

    if len(imu_window) < 2:
        raise RuntimeError(f"Too few IMU samples in window: {len(imu_window)}")

    timestamps_ns, poses_WI = integrate_imu(
        imu_window,
        initial_velocity=velocity,
        initial_pose_WI=initial_pose_WI,
        bias_acc=bias_acc,
        bias_gyr=bias_gyr,
    )
    poses_WL = [T @ T_IL for T in poses_WI]
    pose_buffer = PoseBuffer(timestamps_ns, poses_WL)
    return undistort_cloud(raw_cloud, pose_buffer, desired_ns)


def main():
    args = get_args()
    dataset = OxfordSpiresDataset(args.seq_dir)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"Loading IMU: {dataset.raw_imu_path}")
    imu_df = read_imu(dataset.raw_imu_path)

    print(f"Loading GT poses: {dataset.gt_tum_path}")
    gt_traj = dataset.load_gt_poses()
    print(f"  {gt_traj.num_poses} GT poses")

    T_world_GT = make_T(args.gt_frame_offset)
    if not np.allclose(T_world_GT, np.eye(4)):
        print(f"  Applying GT frame offset: {args.gt_frame_offset}")
        gt_traj.transform(T_world_GT)

    if args.mode == "slam":
        print(f"Loading SLAM poses: {dataset.slam_poses_path}")
        slam_traj = dataset.load_slam_poses()
        print(f"  {slam_traj.num_poses} SLAM keyframes")

    # ── 2. GTSAM batch optimization ───────────────────────────────────────────
    print("\nRunning GTSAM batch optimization ...")
    T_BI_mat = make_T(T_BI_DEFAULT)
    preint_params = build_preint_params(args.acc_noise, args.gyr_noise)
    graph, values = build_factor_graph(
        gt_traj,
        imu_df,
        preint_params,
        state_df=None,
        T_BI=T_BI_mat,
        acc_bias_rw=args.acc_bias_rw,
        gyr_bias_rw=args.gyr_bias_rw,
    )
    print(f"  Graph: {graph.size()} factors, {values.size()} variables")
    result = optimize(graph, values)
    result_df = extract_results(result, gt_traj)
    print(f"  Optimization done. Sample bias_acc: {result_df[['ba_x', 'ba_y', 'ba_z']].mean().values}")
    print(f"  Sample bias_gyr: {result_df[['bg_x', 'bg_y', 'bg_z']].mean().values}")

    # Build sorted lookup: list of (ts_ns, T_WB, vel, bias_acc, bias_gyr)
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
    gt_ts_arr = np.array([s[0] for s in gt_states], dtype=np.int64)

    # ── 2b. Save bias/velocity CSV and dense trajectory ───────────────────────
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
    print(f"  EVO viz: evo_traj tum {tum_path} --plot")

    # ── 3. Build scan list ────────────────────────────────────────────────────
    # Each item: (scan_ts_ns, raw_path_or_None, T_WB_viewpoint_or_None)
    # raw_path=None → find by timestamp match; T_WB_viewpoint=None → use dense trajectory
    # scan_ts_ns is used as both the raw cloud lookup key and the desired undistortion timestamp
    if args.mode == "slam":
        n_total = slam_traj.num_poses if args.n_scans < 0 else args.n_scans
        scan_items = []
        for i in range(n_total):
            slam_ts = TimeStamp(t_float128=slam_traj.timestamps[i])
            slam_ts_ns = slam_ts.sec * 10**9 + slam_ts.nsec
            scan_items.append((slam_ts_ns, None, slam_traj.poses_se3[i]))
    elif args.mode == "raw":
        if not dataset.raw_lidar_dir.exists():
            print(f"WARNING: raw lidar dir does not exist: {dataset.raw_lidar_dir}")
        raw_paths = sorted(dataset.raw_lidar_dir.glob("*.pcd")) if dataset.raw_lidar_dir.exists() else []
        if not raw_paths:
            print(f"WARNING: no PCD files found in {dataset.raw_lidar_dir}")
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
    buffer_ns = int(args.imu_buffer_ms * 1e6)
    T_BL_mat = make_T(T_BL_DEFAULT)
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

        raw_cloud, _ = read_pcd_binary(raw_path)
        scan_start_ns = int(raw_cloud["timestamp"].min() * 1e9)

        idx = np.searchsorted(gt_ts_arr, scan_start_ns, side="right") - 1
        if idx < 0:
            errors.append(f"No GT state before scan start ts={scan_ts_ns}")
            skipped += 1
            pbar.update(1)
            continue
        gt_ts_ns_state, T_WB_gt, vel, bias_acc, bias_gyr = gt_states[idx]

        try:
            corrected_xyz = undistort_one_scan(
                raw_cloud,
                imu_df,
                gt_ts_ns_state,
                T_WB_gt,
                vel,
                bias_acc,
                bias_gyr,
                T_BL_DEFAULT,
                T_BI_DEFAULT,
                desired_ns=scan_ts_ns,
                buffer_ns=buffer_ns,
            )
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
