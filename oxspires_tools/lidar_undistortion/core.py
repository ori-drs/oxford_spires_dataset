"""Core undistortion functions: transforms, state loading, IMU integration, point-cloud correction."""

from pathlib import Path

import gtsam
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

STATE_COLS = [
    "sec",
    "nsec",
    "W_r_WB_x",
    "W_r_WB_y",
    "W_r_WB_z",
    "q_WB_x",
    "q_WB_y",
    "q_WB_z",
    "q_WB_w",
    "B_v_WB_x",
    "B_v_WB_y",
    "B_v_WB_z",
    "B_w_WB_x",
    "B_w_WB_y",
    "B_w_WB_z",
    "biasAcc_x",
    "biasAcc_y",
    "biasAcc_z",
    "biasGyr_x",
    "biasGyr_y",
    "biasGyr_z",
    "tBiasAng_x",
    "tBiasAng_y",
    "tBiasAng_z",
    "tBiasLin_x",
    "tBiasLin_y",
    "tBiasLin_z",
]


def make_T(xyzw: list) -> np.ndarray:
    """Build 4x4 SE3 matrix from [tx, ty, tz, qx, qy, qz, qw]."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(xyzw[3:]).as_matrix()
    T[:3, 3] = xyzw[:3]
    return T


def load_state_csv(path: Path) -> pd.DataFrame:
    """Load VILENS state.csv, skipping an optional header row if present."""
    with open(path) as f:
        skip = 0 if f.readline().split(",")[0].strip().lstrip("-").isdigit() else 1
    df = pd.read_csv(path, header=None, names=STATE_COLS, skiprows=skip, dtype={0: np.int64, 1: np.int64})
    df["timestamp_ns"] = df["sec"] * np.int64(1_000_000_000) + df["nsec"]
    return df


def derive_initial_state(state_row, T_BL: list, T_BI: list) -> dict:
    """Derive T_WL, W_v_WI, bias_acc, bias_gyr from a state row."""
    r = state_row
    T_WB = make_T([r.W_r_WB_x, r.W_r_WB_y, r.W_r_WB_z, r.q_WB_x, r.q_WB_y, r.q_WB_z, r.q_WB_w])
    T_WI = T_WB @ make_T(T_BI)
    T_WL = T_WB @ make_T(T_BL)
    R_WB = T_WB[:3, :3]
    B_v_WB = np.array([r.B_v_WB_x, r.B_v_WB_y, r.B_v_WB_z])
    B_w_WB = np.array([r.B_w_WB_x, r.B_w_WB_y, r.B_w_WB_z])
    W_v_WB = R_WB @ B_v_WB
    W_w_WI = R_WB @ B_w_WB
    W_r_BI = T_WI[:3, 3] - T_WB[:3, 3]
    W_v_WI = W_v_WB + np.cross(W_w_WI, W_r_BI)
    return {
        "T_WL": T_WL,
        "W_v_WI": W_v_WI,
        "bias_acc": np.array([r.biasAcc_x, r.biasAcc_y, r.biasAcc_z]),
        "bias_gyr": np.array([r.biasGyr_x, r.biasGyr_y, r.biasGyr_z]),
    }


def integrate_imu(
    imu_window: pd.DataFrame,
    initial_velocity: np.ndarray = np.zeros(3),
    initial_pose_WI: np.ndarray = np.eye(4),
    bias_acc: np.ndarray = np.zeros(3),
    bias_gyr: np.ndarray = np.zeros(3),
):
    """Step-by-step GTSAM IMU integration starting from the given T_WI pose and velocity.

    Returns (timestamps_ns list, SE3 4x4 list) of IMU-frame poses T_WI.
    """
    params = gtsam.PreintegrationParams.MakeSharedU(9.8067)
    params.setAccelerometerCovariance(np.eye(3) * 0.1**2)
    params.setGyroscopeCovariance(np.eye(3) * 0.01**2)
    params.setIntegrationCovariance(np.eye(3) * 1e-8)
    bias = gtsam.imuBias.ConstantBias(bias_acc, bias_gyr)

    timestamps_ns = []
    poses = []

    rot = gtsam.Rot3(initial_pose_WI[:3, :3])
    pos = gtsam.Point3(initial_pose_WI[:3, 3])
    current_nav = gtsam.NavState(gtsam.Pose3(rot, pos), initial_velocity)

    for i in range(len(imu_window) - 1):
        row = imu_window.iloc[i]
        t_ns = int(row["timestamp_ns"])
        next_t_ns = int(imu_window.iloc[i + 1]["timestamp_ns"])
        dt = (next_t_ns - t_ns) * 1e-9
        if dt <= 0:
            continue

        acc = np.array([row["acc_x"], row["acc_y"], row["acc_z"]])
        gyro = np.array([row["ang_vel_x"], row["ang_vel_y"], row["ang_vel_z"]])

        timestamps_ns.append(t_ns)
        poses.append(current_nav.pose().matrix())

        pim = gtsam.PreintegratedImuMeasurements(params, bias)
        pim.integrateMeasurement(acc, gyro, dt)
        current_nav = pim.predict(current_nav, bias)

    timestamps_ns.append(int(imu_window.iloc[-1]["timestamp_ns"]))
    poses.append(current_nav.pose().matrix())

    return timestamps_ns, poses


class PoseBuffer:
    """SLERP/linear-interpolated pose buffer keyed by timestamp (ns)."""

    def __init__(self, timestamps_ns: list, poses: list):
        self._ts = np.array(timestamps_ns, dtype=np.float64)
        self._trans = np.array([T[:3, 3] for T in poses])
        rot_mats = np.stack([T[:3, :3] for T in poses])
        self._slerp = Slerp(self._ts, Rotation.from_matrix(rot_mats))

    def interpolate_batch(self, query_ns: np.ndarray) -> np.ndarray:
        """Interpolate poses at N timestamps. Returns (N, 4, 4)."""
        q = np.clip(query_ns.astype(np.float64), self._ts[0], self._ts[-1])

        idx = np.searchsorted(self._ts, q)
        idx = np.clip(idx, 1, len(self._ts) - 1)
        t0 = self._ts[idx - 1]
        t1 = self._ts[idx]
        alpha = (q - t0) / (t1 - t0)
        trans = self._trans[idx - 1] + alpha[:, None] * (self._trans[idx] - self._trans[idx - 1])

        rots = self._slerp(q).as_matrix()

        T = np.zeros((len(q), 4, 4))
        T[:, :3, :3] = rots
        T[:, :3, 3] = trans
        T[:, 3, 3] = 1.0
        return T


def undistort_cloud(cloud: np.ndarray, pose_buffer: PoseBuffer, t_desired_ns: int, t_start_ns=None) -> np.ndarray:
    """Apply per-point motion compensation. Returns corrected xyz (N, 3).

    Supports two timestamp formats:
    - 'timestamp' field: absolute seconds (float)
    - 't' field: relative nanoseconds from scan start; requires t_start_ns
    """
    if "timestamp" in cloud.dtype.names:
        t_points_ns = (cloud["timestamp"] * 1e9).astype(np.int64)
    elif "t" in cloud.dtype.names:
        if t_start_ns is None:
            raise ValueError("t_start_ns required when cloud has 't' field (relative ns)")
        t_points_ns = t_start_ns + cloud["t"].astype(np.int64)
    else:
        raise ValueError(f"Cloud has no timestamp field; fields: {cloud.dtype.names}")

    T_desired = pose_buffer.interpolate_batch(np.array([t_desired_ns]))[0]
    T_desired_inv = np.linalg.inv(T_desired)
    T_points = pose_buffer.interpolate_batch(t_points_ns)

    T_rel = T_desired_inv @ T_points  # (N, 4, 4)

    xyz = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=-1)
    return np.einsum("nij,nj->ni", T_rel[:, :3, :3], xyz) + T_rel[:, :3, 3]
