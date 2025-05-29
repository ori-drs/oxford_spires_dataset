'''
# import nerf_data_pipeline.pose.lie_group.se3 as se3
import numpy as np
import theseus as th
from theseus.geometry.se3 import SE3, SO3
import torch
from evo.core.trajectory import PosePath3D
from copy import deepcopy


def generate_noise_SE3(var_R, var_t):
    """
    Generate a random SE3 matrix
    @param var_R: variance of rotation
    @param var_t: variance of translation
    @return T: 4x4 SE3 matrix
    """
    size = 1
    R = SO3.exp_map(
        th.constants.PI
        * var_R
        * torch.randn(
            size,
            3,
        )
    )
    t = torch.randn(size, 3) * var_t
    T = torch.eye(4)
    T[:3, :3] = R.tensor
    T[:3, 3] = t
    return T.numpy()


def add_odom_noise(traj, var_R, var_t):
    """
    Add odometry noise to the trajectory
    @param traj: trajectory to add noise to, PosePath3D
    @param var_R: variance of rotation
    @param var_t: variance of translation
    @return traj_odom: trajectory with odometry noise, PosePath3D
    """
    assert issubclass(type(traj), PosePath3D)
    assert traj.check()[0], traj.check()[1]
    traj_odom = deepcopy(traj)
    for i in range(traj.positions_xyz.shape[0] - 1):
        T_WA = traj.poses_se3[i]
        T_WB = traj.poses_se3[i + 1]
        T_AB = np.linalg.inv(T_WA) @ T_WB
        noise = generate_noise_SE3(var_R, var_t)
        traj_odom.poses_se3[i + 1] = traj_odom.poses_se3[i] @ noise @ T_AB
    return traj_odom
'''
