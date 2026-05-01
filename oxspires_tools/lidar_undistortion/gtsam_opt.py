"""GTSAM batch optimisation for IMU bias/velocity estimation and calibration analysis."""

from typing import Optional

import gtsam
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from oxspires_tools.lidar_undistortion.core import derive_initial_state, integrate_imu
from oxspires_tools.se3 import se3_matrix_to_xyz_quat_xyzw
from oxspires_tools.sensor import Sensor
from oxspires_tools.trajectory.file_interfaces.timestamp import TimeStamp

GRAVITY = np.array([0, 0, -9.808083883386614])


def load_sensor_transforms(sensor_yaml) -> tuple:
    """Load T_base_imu (4x4), T_base_imu_t_xyz_q_xyzw, T_base_lidar_t_xyz_q_xyzw from sensor.yaml."""
    with open(sensor_yaml) as f:
        sensor = Sensor(**yaml.safe_load(f)["sensor"])
    T_base_imu = sensor.tf.get_transform("imu", "base")
    xyz, quat_xyzw = se3_matrix_to_xyz_quat_xyzw(T_base_imu)
    T_base_imu_t_xyz_q_xyzw = list(xyz) + list(quat_xyzw)
    return T_base_imu, T_base_imu_t_xyz_q_xyzw, sensor.T_base_lidar_t_xyz_q_xyzw


X = gtsam.symbol_shorthand.X
V = gtsam.symbol_shorthand.V
B = gtsam.symbol_shorthand.B


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
    mask = (ts > t_start_ns) & (ts < t_end_ns)
    inside = imu_df[mask]

    before_mask = ts <= t_start_ns
    if before_mask.any():
        left_row = imu_df[before_mask].iloc[-1]
    else:
        left_row = imu_df.iloc[0]

    acc_left = np.array([left_row["acc_x"], left_row["acc_y"], left_row["acc_z"]])
    gyro_left = np.array([left_row["ang_vel_x"], left_row["ang_vel_y"], left_row["ang_vel_z"]])

    if len(inside) == 0:
        dt = (t_end_ns - t_start_ns) * 1e-9
        if dt > 0:
            pim.integrateMeasurement(acc_left, gyro_left, dt)
        return pim

    first_inside = inside.iloc[0]
    dt_first = (int(first_inside["timestamp_ns"]) - t_start_ns) * 1e-9
    if dt_first > 0:
        pim.integrateMeasurement(acc_left, gyro_left, dt_first)

    inside_arr = inside.reset_index(drop=True)
    for i in range(len(inside_arr) - 1):
        row = inside_arr.iloc[i]
        next_row = inside_arr.iloc[i + 1]
        acc = np.array([row["acc_x"], row["acc_y"], row["acc_z"]])
        gyro = np.array([row["ang_vel_x"], row["ang_vel_y"], row["ang_vel_z"]])
        dt = (int(next_row["timestamp_ns"]) - int(row["timestamp_ns"])) * 1e-9
        if dt > 0:
            pim.integrateMeasurement(acc, gyro, dt)

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
    T_base_imu: np.ndarray,
    acc_bias_rw: float,
    gyr_bias_rw: float,
    T_BL: list,
):
    """Build GTSAM NonlinearFactorGraph, initial Values, and list of ImuFactor objects."""
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    imu_factors = []

    N = gt_traj.num_poses
    zero_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    xyz, quat_xyzw = se3_matrix_to_xyz_quat_xyzw(T_base_imu)
    T_base_imu_t_xyz_q_xyzw = list(xyz) + list(quat_xyzw)

    def _nearest_state(ts_ns):
        if state_df is None:
            return None
        idx = (state_df["timestamp_ns"] - ts_ns).abs().idxmin()
        return state_df.loc[idx]

    pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3] * 3 + [1e-2] * 3))
    pose_prior_noise_0 = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 3 + [1e-6] * 3))
    vel_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, 2.0)
    bias_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2] * 3 + [1e-2] * 3))

    pbar = tqdm(total=N, desc="Building factor graph")

    for i in range(N):
        ts = TimeStamp(t_float128=gt_traj.timestamps[i])
        T_WB = gt_traj.poses_se3[i]
        T_WI = T_WB @ T_base_imu
        rot = gtsam.Rot3(T_WI[:3, :3])
        pos = gtsam.Point3(T_WI[:3, 3])
        pose_i = gtsam.Pose3(rot, pos)

        noise = pose_prior_noise_0 if i == 0 else pose_prior_noise
        graph.addPriorPose3(X(i), pose_i, noise)
        values.insert(X(i), pose_i)

        s = _nearest_state(ts.sec * 10**9 + ts.nsec)
        if s is not None:
            init = derive_initial_state(s, T_BL, T_base_imu_t_xyz_q_xyzw)
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
            graph.addPriorPoint3(V(0), vel_guess, vel_prior_noise)
            graph.push_back(
                gtsam.PriorFactorConstantBias(
                    B(0),
                    gtsam.imuBias.ConstantBias(bias_acc, bias_gyr),
                    bias_prior_noise,
                )
            )
        else:
            prev_ts = TimeStamp(t_float128=gt_traj.timestamps[i - 1])
            bias_lin = gtsam.imuBias.ConstantBias(bias_acc, bias_gyr)
            pim = _preintegrate_interval(
                imu_df, prev_ts.sec * 10**9 + prev_ts.nsec, ts.sec * 10**9 + ts.nsec, preint_params, bias_lin
            )
            imu_f = gtsam.ImuFactor(X(i - 1), V(i - 1), X(i), V(i), B(i - 1), pim)
            imu_factors.append(imu_f)
            graph.push_back(imu_f)

            dt = (ts.sec - prev_ts.sec) + (ts.nsec - prev_ts.nsec) * 1e-9
            sigma_ba = acc_bias_rw * np.sqrt(dt)
            sigma_bg = gyr_bias_rw * np.sqrt(dt)
            bias_rw_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma_ba] * 3 + [sigma_bg] * 3))
            graph.push_back(gtsam.BetweenFactorConstantBias(B(i - 1), B(i), zero_bias, bias_rw_noise))

        pbar.update(1)

    pbar.close()
    return graph, values, imu_factors


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


def run_optimization(gt_traj, imu_df, acc_noise, gyr_noise, acc_bias_rw, gyr_bias_rw, state_df, T_base_imu, T_BL: list):
    """Run full GTSAM batch optimisation and return (graph, result, result_df, imu_factors)."""
    preint_params = build_preint_params(acc_noise, gyr_noise)
    graph, values, imu_factors = build_factor_graph(
        gt_traj, imu_df, preint_params, state_df, T_base_imu, acc_bias_rw, gyr_bias_rw, T_BL
    )
    result = optimize(graph, values)
    result_df = extract_results(result, gt_traj)
    return graph, result, result_df, imu_factors


def analyze_imu_chi2(imu_factors: list, result: gtsam.Values) -> dict:
    """Chi-squared consistency check for ImuFactor residuals (DOF=9, ideal mean_error=4.5)."""
    DOF = 9
    errors = np.array([f.error(result) for f in imu_factors])
    mean_err = float(errors.mean())
    return {
        "n_factors": len(errors),
        "mean_error": mean_err,
        "expected_error": DOF / 2.0,
        "chi2_ratio": mean_err / (DOF / 2.0),
    }


def print_calib_comparison(label_a, label_b, params_a, params_b, chi2_a, chi2_b, rw_a, rw_b):
    """Print side-by-side calibration validation for two parameter sets."""
    W = 26
    print(f"\n{'':=<70}")
    print("IMU Calibration Comparison")
    print(f"{'':=<70}")
    print(f"{'':>{W}}  {('  ' + label_a):>20}  {('  ' + label_b):>20}")
    print(f"{'acc_noise_density':>{W}}  {params_a['acc_noise']:>20.6e}  {params_b['acc_noise']:>20.6e}")
    print(f"{'gyr_noise_density':>{W}}  {params_a['gyr_noise']:>20.6e}  {params_b['gyr_noise']:>20.6e}")
    print(f"{'acc_bias_rw':>{W}}  {params_a['acc_bias_rw']:>20.6e}  {params_b['acc_bias_rw']:>20.6e}")
    print(f"{'gyr_bias_rw':>{W}}  {params_a['gyr_bias_rw']:>20.6e}  {params_b['gyr_bias_rw']:>20.6e}")

    print("\n  Chi-squared (noise density)  — ideal chi2_ratio = 1.0")
    print(f"{'n_imu_factors':>{W}}  {chi2_a['n_factors']:>20d}  {chi2_b['n_factors']:>20d}")
    print(f"{'mean_error':>{W}}  {chi2_a['mean_error']:>20.3f}  {chi2_b['mean_error']:>20.3f}")
    print(f"{'expected_error (DOF/2=4.5)':>{W}}  {chi2_a['expected_error']:>20.3f}  {chi2_b['expected_error']:>20.3f}")
    print(f"{'chi2_ratio':>{W}}  {chi2_a['chi2_ratio']:>20.3f}  {chi2_b['chi2_ratio']:>20.3f}")

    print("\n  Bias random walk  — empirical std(Δbias/√dt) vs calibrated")
    for i, ax in enumerate(["x", "y", "z"]):
        print(
            f"{'acc_rw empirical_' + ax:>{W}}  {rw_a['acc_empirical_rw'][i]:>20.3e}  {rw_b['acc_empirical_rw'][i]:>20.3e}"
        )
    print(f"{'acc_rw calibrated':>{W}}  {rw_a['acc_calibrated_rw']:>20.3e}  {rw_b['acc_calibrated_rw']:>20.3e}")
    for i, ax in enumerate(["x", "y", "z"]):
        print(
            f"{'gyr_rw empirical_' + ax:>{W}}  {rw_a['gyr_empirical_rw'][i]:>20.3e}  {rw_b['gyr_empirical_rw'][i]:>20.3e}"
        )
    print(f"{'gyr_rw calibrated':>{W}}  {rw_a['gyr_calibrated_rw']:>20.3e}  {rw_b['gyr_calibrated_rw']:>20.3e}")
    print(f"{'':=<70}")


def compare_with_vilens(
    result_df: pd.DataFrame,
    state_df: pd.DataFrame,
    gt_traj,
    T_base_imu_t_xyz_q_xyzw: list,
    T_BL: list,
    max_dt_s: float = 0.5,
):
    """Compare GTSAM estimates against VILENS state.csv."""
    print("\n=== Comparison with VILENS state.csv ===")
    vel_diffs, ba_diffs, bg_diffs = [], [], []
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

        init = derive_initial_state(s, T_BL, T_base_imu_t_xyz_q_xyzw)
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
                continue
            dense_ts.append(t)
            dense_poses_WB.append(T_WI @ T_IB)
    return dense_ts, dense_poses_WB


def _interpolate_left_correction(T_correction: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate a world-frame correction transform from identity to T_correction."""
    T_alpha = np.eye(4)
    rotvec = Rotation.from_matrix(T_correction[:3, :3]).as_rotvec()
    T_alpha[:3, :3] = Rotation.from_rotvec(alpha * rotvec).as_matrix()
    T_alpha[:3, 3] = alpha * T_correction[:3, 3]
    return T_alpha


def build_gt_anchored_trajectory(gt_states: list, imu_df: pd.DataFrame, T_BI_mat: np.ndarray):
    """Build dense T_WB poses with every IMU segment corrected to the next GT keyframe.

    IMU integration provides the high-rate motion shape within each GT interval. The
    endpoint error of that integrated segment is then distributed across the interval,
    so every segment starts at GT_i and ends exactly at GT_{i+1}.
    """
    T_IB = np.linalg.inv(T_BI_mat)
    dense_ts = []
    dense_poses_WB = []

    for i in range(len(gt_states) - 1):
        ts_ns_i, T_WB_i, vel_i, bias_acc_i, bias_gyr_i = gt_states[i]
        ts_ns_j, T_WB_j = gt_states[i + 1][0], gt_states[i + 1][1]
        dt_ns = ts_ns_j - ts_ns_i
        if dt_ns <= 0:
            continue

        imu_window = imu_df[(imu_df["timestamp_ns"] >= ts_ns_i) & (imu_df["timestamp_ns"] <= ts_ns_j)].reset_index(
            drop=True
        )

        internal_ts = []
        internal_raw_poses_WB = []
        raw_endpoint_WB = T_WB_i
        if len(imu_window) >= 2:
            timestamps_ns, poses_WI = integrate_imu(
                imu_window,
                initial_velocity=vel_i,
                initial_pose_WI=T_WB_i @ T_BI_mat,
                bias_acc=bias_acc_i,
                bias_gyr=bias_gyr_i,
            )
            raw_poses_WB = [T_WI @ T_IB for T_WI in poses_WI]
            if raw_poses_WB:
                raw_endpoint_WB = raw_poses_WB[-1]
            for t, T_raw_WB in zip(timestamps_ns, raw_poses_WB):
                if ts_ns_i < t < ts_ns_j:
                    internal_ts.append(t)
                    internal_raw_poses_WB.append(T_raw_WB)

        T_endpoint_correction = T_WB_j @ np.linalg.inv(raw_endpoint_WB)
        segment_ts = [ts_ns_i] + internal_ts + [ts_ns_j]
        segment_raw_poses = [T_WB_i] + internal_raw_poses_WB + [raw_endpoint_WB]

        for t, T_raw_WB in zip(segment_ts, segment_raw_poses):
            if dense_ts and t <= dense_ts[-1]:
                continue
            alpha = (t - ts_ns_i) / dt_ns
            if t == ts_ns_i:
                T_WB = T_WB_i
            elif t == ts_ns_j:
                T_WB = T_WB_j
            else:
                T_WB = _interpolate_left_correction(T_endpoint_correction, alpha) @ T_raw_WB
            dense_ts.append(t)
            dense_poses_WB.append(T_WB)

    return dense_ts, dense_poses_WB
