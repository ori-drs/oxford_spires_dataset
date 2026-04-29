"""Estimate IMU bias and velocity from GT poses using GTSAM batch optimization."""

import argparse
from pathlib import Path
from typing import Optional

import gtsam
import numpy as np
import pandas as pd
from tqdm import tqdm

from oxspires_tools.lidar_undistortion.core import derive_initial_state, load_state_csv, make_T
from oxspires_tools.lidar_undistortion.io import read_imu
from oxspires_tools.trajectory.file_interfaces.timestamp import TimeStamp
from oxspires_tools.trajectory.file_interfaces.tum import TUMTrajReader

# Default extrinsics and noise params (frn-0019 Alphasense)
T_BI_DEFAULT = [-0.018, 0.006, 0.058, 0, 0, 0.707, 0.707]
T_BL_DEFAULT = [0, 0, 0.124, 0, 0, 1, 0]
GRAVITY = np.array([0, 0, -9.808083883386614])
ACC_NOISE = 0.001799
GYR_NOISE = 0.000257
ACC_BIAS_RW = 2.69e-4
GYR_BIAS_RW = 1.57e-5

X = gtsam.symbol_shorthand.X
V = gtsam.symbol_shorthand.V
B = gtsam.symbol_shorthand.B


def get_args():
    parser = argparse.ArgumentParser(
        description="Estimate IMU bias/velocity from GT poses via GTSAM batch optimization."
    )
    parser.add_argument("--data_dir", type=Path, required=True, help="Root data directory")
    parser.add_argument(
        "--imu_csv", type=Path, default=None, help="Path to imu_full.csv (default: data_dir/imu_full.csv)"
    )
    parser.add_argument("--gt_tum", type=Path, default=None, help="Path to gt_tum.txt (default: data_dir/gt_tum.txt)")
    parser.add_argument(
        "--state_csv", type=Path, default=None, help="Path to vilens_logs/state.csv for comparison/init"
    )
    parser.add_argument("--acc_noise", type=float, default=ACC_NOISE)
    parser.add_argument("--gyr_noise", type=float, default=GYR_NOISE)
    parser.add_argument("--acc_bias_rw", type=float, default=ACC_BIAS_RW)
    parser.add_argument("--gyr_bias_rw", type=float, default=GYR_BIAS_RW)
    parser.add_argument("--output_csv", type=Path, default=None, help="Save result DataFrame to CSV")
    parser.add_argument(
        "--gt_frame_offset",
        type=float,
        nargs=7,
        default=[0, 0, 0, 0, 0, 0, 1],
        metavar=("tx", "ty", "tz", "qx", "qy", "qz", "qw"),
        help="Pose of GT frame origin in world frame: tx ty tz qx qy qz qw (default: identity). "
        "Applied as T_W_B = T_world_GT @ T_GT_B to re-express GT poses in the world frame.",
    )
    return parser.parse_args()


def build_preint_params(acc_noise: float, gyr_noise: float) -> gtsam.PreintegrationParams:
    """Build GTSAM PreintegrationParams with frn-0019 defaults."""
    params = gtsam.PreintegrationParams(GRAVITY)
    params.setAccelerometerCovariance(np.eye(3) * acc_noise**2)
    params.setGyroscopeCovariance(np.eye(3) * gyr_noise**2)
    params.setIntegrationCovariance(np.eye(3) * 1e-8)
    return params


def _preintegrate_interval(
    imu_df: pd.DataFrame,
    t_start_ns: int,
    t_end_ns: int,
    params: gtsam.PreintegrationParams,
    bias: gtsam.imuBias.ConstantBias,
) -> gtsam.PreintegratedImuMeasurements:
    """Preintegrate IMU measurements in [t_start_ns, t_end_ns] with boundary handling."""
    pim = gtsam.PreintegratedImuMeasurements(params, bias)

    ts = imu_df["timestamp_ns"].values
    # Find samples strictly inside the interval
    mask = (ts > t_start_ns) & (ts < t_end_ns)
    inside = imu_df[mask]

    # Find sample just before t_start (for left boundary)
    before_mask = ts <= t_start_ns
    if before_mask.any():
        left_row = imu_df[before_mask].iloc[-1]
    else:
        left_row = imu_df.iloc[0]

    acc_left = np.array([left_row["acc_x"], left_row["acc_y"], left_row["acc_z"]])
    gyro_left = np.array([left_row["ang_vel_x"], left_row["ang_vel_y"], left_row["ang_vel_z"]])

    if len(inside) == 0:
        # No samples inside — integrate across the whole gap with left sample
        dt = (t_end_ns - t_start_ns) * 1e-9
        if dt > 0:
            pim.integrateMeasurement(acc_left, gyro_left, dt)
        return pim

    # First sub-interval: t_start_ns → first inside sample
    first_inside = inside.iloc[0]
    dt_first = (int(first_inside["timestamp_ns"]) - t_start_ns) * 1e-9
    if dt_first > 0:
        pim.integrateMeasurement(acc_left, gyro_left, dt_first)

    # Middle samples
    inside_arr = inside.reset_index(drop=True)
    for i in range(len(inside_arr) - 1):
        row = inside_arr.iloc[i]
        next_row = inside_arr.iloc[i + 1]
        acc = np.array([row["acc_x"], row["acc_y"], row["acc_z"]])
        gyro = np.array([row["ang_vel_x"], row["ang_vel_y"], row["ang_vel_z"]])
        dt = (int(next_row["timestamp_ns"]) - int(row["timestamp_ns"])) * 1e-9
        if dt > 0:
            pim.integrateMeasurement(acc, gyro, dt)

    # Last sub-interval: last inside sample → t_end_ns
    last_inside = inside_arr.iloc[-1]
    acc_last = np.array([last_inside["acc_x"], last_inside["acc_y"], last_inside["acc_z"]])
    gyro_last = np.array([last_inside["ang_vel_x"], last_inside["ang_vel_y"], last_inside["ang_vel_z"]])
    dt_last = (t_end_ns - int(last_inside["timestamp_ns"])) * 1e-9
    if dt_last > 0:
        pim.integrateMeasurement(acc_last, gyro_last, dt_last)

    return pim


def build_factor_graph(
    gt_traj,
    imu_df: pd.DataFrame,
    preint_params: gtsam.PreintegrationParams,
    state_df: Optional[pd.DataFrame],
    T_BI: np.ndarray,
    acc_bias_rw: float,
    gyr_bias_rw: float,
):
    """Build GTSAM NonlinearFactorGraph and initial Values."""
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    N = gt_traj.num_poses
    zero_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))

    def _nearest_state(ts_ns):
        if state_df is None:
            return None
        idx = (state_df["timestamp_ns"] - ts_ns).abs().idxmin()
        return state_df.loc[idx]

    # Tight pose prior noise
    pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3] * 3 + [1e-2] * 3))
    # Loose pose prior for node 0
    pose_prior_noise_0 = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 3 + [1e-6] * 3))
    vel_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, 2.0)
    bias_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2] * 3 + [1e-2] * 3))

    pbar = tqdm(total=N, desc="Building factor graph")

    for i in range(N):
        ts = TimeStamp(t_float128=gt_traj.timestamps[i])
        T_WB = gt_traj.poses_se3[i]
        T_WI = T_WB @ T_BI
        rot = gtsam.Rot3(T_WI[:3, :3])
        pos = gtsam.Point3(T_WI[:3, 3])
        pose_i = gtsam.Pose3(rot, pos)

        # Tight GT pose prior at every node
        noise = pose_prior_noise_0 if i == 0 else pose_prior_noise
        graph.addPriorPose3(X(i), pose_i, noise)

        # Initial values for pose
        values.insert(X(i), pose_i)

        # Initial velocity and bias guesses from state.csv or zeros
        s = _nearest_state(ts.sec * 10**9 + ts.nsec)
        if s is not None:
            T_BI_list = list(T_BI[:3, 3]) + list(gtsam.Rot3(T_BI[:3, :3]).toQuaternion().coeffs())
            init = derive_initial_state(s, T_BL_DEFAULT, T_BI_list)
            vel_guess = init["W_v_WI"]
            bias_acc = init["bias_acc"]
            bias_gyr = init["bias_gyr"]
        else:
            vel_guess = np.zeros(3)
            bias_acc = np.zeros(3)
            bias_gyr = np.zeros(3)

        values.insert(V(i), vel_guess)
        values.insert(B(i), gtsam.imuBias.ConstantBias(bias_acc, bias_gyr))

        if i == 0:
            # Priors on velocity and bias at node 0
            graph.addPriorPoint3(V(0), vel_guess, vel_prior_noise)
            graph.push_back(
                gtsam.PriorFactorConstantBias(
                    B(0),
                    gtsam.imuBias.ConstantBias(bias_acc, bias_gyr),
                    bias_prior_noise,
                )
            )
        else:
            # IMU preintegration factor
            prev_ts = TimeStamp(t_float128=gt_traj.timestamps[i - 1])
            bias_lin = gtsam.imuBias.ConstantBias(bias_acc, bias_gyr)
            pim = _preintegrate_interval(
                imu_df, prev_ts.sec * 10**9 + prev_ts.nsec, ts.sec * 10**9 + ts.nsec, preint_params, bias_lin
            )
            graph.push_back(gtsam.ImuFactor(X(i - 1), V(i - 1), X(i), V(i), B(i - 1), pim))

            # Bias random-walk factor
            dt = (ts.sec - prev_ts.sec) + (ts.nsec - prev_ts.nsec) * 1e-9
            sigma_ba = acc_bias_rw * np.sqrt(dt)
            sigma_bg = gyr_bias_rw * np.sqrt(dt)
            bias_rw_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma_ba] * 3 + [sigma_bg] * 3))
            graph.push_back(gtsam.BetweenFactorConstantBias(B(i - 1), B(i), zero_bias, bias_rw_noise))

        pbar.update(1)

    pbar.close()
    return graph, values


def optimize(graph: gtsam.NonlinearFactorGraph, values: gtsam.Values) -> gtsam.Values:
    """Run LevenbergMarquardt optimization."""
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
    return optimizer.optimize()


def extract_results(result: gtsam.Values, gt_traj) -> pd.DataFrame:
    """Extract velocity and bias estimates into a DataFrame."""
    rows = []
    for i in range(gt_traj.num_poses):
        ts = TimeStamp(t_float128=gt_traj.timestamps[i])
        vel = result.atVector(V(i))
        bias = result.atConstantBias(B(i))
        ba = bias.accelerometer()
        bg = bias.gyroscope()
        rows.append(
            {
                "timestamp_ns": ts.sec * 10**9 + ts.nsec,
                "vel_x": vel[0],
                "vel_y": vel[1],
                "vel_z": vel[2],
                "ba_x": ba[0],
                "ba_y": ba[1],
                "ba_z": ba[2],
                "bg_x": bg[0],
                "bg_y": bg[1],
                "bg_z": bg[2],
            }
        )
    return pd.DataFrame(rows)


def compare_with_vilens(
    result_df: pd.DataFrame, state_df: pd.DataFrame, gt_traj, T_BI_list: list, max_dt_s: float = 0.5
):
    """Compare GTSAM estimates against VILENS state.csv, restricted to poses within max_dt_s of a state row."""
    print("\n=== Comparison with VILENS state.csv ===")
    vel_diffs = []
    ba_diffs = []
    bg_diffs = []
    n_skipped = 0

    for i in range(gt_traj.num_poses):
        ts = TimeStamp(t_float128=gt_traj.timestamps[i])
        ts_ns = ts.sec * 10**9 + ts.nsec
        idx = (state_df["timestamp_ns"] - ts_ns).abs().idxmin()
        s = state_df.loc[idx]
        dt = abs(int(s["timestamp_ns"]) - ts_ns) * 1e-9
        if dt > max_dt_s:
            n_skipped += 1
            continue

        init = derive_initial_state(s, T_BL_DEFAULT, T_BI_list)
        W_v_WI_vilens = init["W_v_WI"]

        row = result_df.iloc[i]
        W_v_WI_gtsam = np.array([row["vel_x"], row["vel_y"], row["vel_z"]])
        ba_gtsam = np.array([row["ba_x"], row["ba_y"], row["ba_z"]])
        bg_gtsam = np.array([row["bg_x"], row["bg_y"], row["bg_z"]])

        ba_vilens = np.array([s.biasAcc_x, s.biasAcc_y, s.biasAcc_z])
        bg_vilens = np.array([s.biasGyr_x, s.biasGyr_y, s.biasGyr_z])

        vel_diffs.append(W_v_WI_gtsam - W_v_WI_vilens)
        ba_diffs.append(ba_gtsam - ba_vilens)
        bg_diffs.append(bg_gtsam - bg_vilens)

    if not vel_diffs:
        print(f"  No overlapping poses found within max_dt_s={max_dt_s}s — skipping comparison.")
        return

    print(f"  Comparing {len(vel_diffs)} poses (skipped {n_skipped} outside ±{max_dt_s}s of state.csv)")
    vel_diffs = np.array(vel_diffs)
    ba_diffs = np.array(ba_diffs)
    bg_diffs = np.array(bg_diffs)

    def rms(x):
        return np.sqrt(np.mean(x**2, axis=0))

    print(f"{'Metric':<20} {'x':>10} {'y':>10} {'z':>10}")
    print("-" * 52)
    print(f"{'vel RMS (m/s)':<20} {rms(vel_diffs)[0]:>10.4f} {rms(vel_diffs)[1]:>10.4f} {rms(vel_diffs)[2]:>10.4f}")
    print(
        f"{'vel mean (m/s)':<20} {vel_diffs.mean(0)[0]:>10.4f} {vel_diffs.mean(0)[1]:>10.4f} {vel_diffs.mean(0)[2]:>10.4f}"
    )
    print(f"{'bias_acc RMS':<20} {rms(ba_diffs)[0]:>10.5f} {rms(ba_diffs)[1]:>10.5f} {rms(ba_diffs)[2]:>10.5f}")
    print(f"{'bias_gyr RMS':<20} {rms(bg_diffs)[0]:>10.6f} {rms(bg_diffs)[1]:>10.6f} {rms(bg_diffs)[2]:>10.6f}")


def main():
    args = get_args()
    data_dir = args.data_dir

    imu_path = args.imu_csv or data_dir / "imu_full.csv"
    gt_path = args.gt_tum or data_dir / "gt_tum.txt"
    state_path = args.state_csv or data_dir / "vilens_logs" / "state.csv"

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

    T_BI_mat = make_T(T_BI_DEFAULT)

    preint_params = build_preint_params(args.acc_noise, args.gyr_noise)

    graph, values = build_factor_graph(
        gt_traj,
        imu_df,
        preint_params,
        state_df,
        T_BI_mat,
        args.acc_bias_rw,
        args.gyr_bias_rw,
    )

    print(f"\nGraph: {graph.size()} factors, {values.size()} variables")
    print("Running LevenbergMarquardt optimization...")
    result = optimize(graph, values)

    result_df = extract_results(result, gt_traj)
    print(result_df.head(10).to_string())

    if args.output_csv:
        result_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved results to {args.output_csv}")

    if state_df is not None:
        T_BI_list = T_BI_DEFAULT
        compare_with_vilens(result_df, state_df, gt_traj, T_BI_list)


if __name__ == "__main__":
    main()
