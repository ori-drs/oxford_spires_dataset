import json

import numpy as np
from evo.core import sync


def align(vilens_traj, colmap_traj, output_dir):
    vilens_traj_sync, colmap_traj_sync = sync.associate_trajectories(vilens_traj, colmap_traj, max_diff=0.05)
    print(f"Trajectory pair length: {len(vilens_traj_sync.poses_se3)}")
    r_a, t_a, align_s = colmap_traj_sync.align(vilens_traj_sync, correct_scale=True)
    align_data = {
        "rotation": r_a.tolist(),
        "translation": t_a.tolist(),
        "scale": align_s,
    }
    print("Save align data to evo_align_results.json")
    with open(output_dir / "evo_align_results.json", "w") as f:
        json.dump(align_data, f, indent=2)
    print(json.dumps(align_data, indent=2))
    T_vilens_colmap = np.eye(4)
    T_vilens_colmap[:3, :3] = np.array(r_a)
    T_vilens_colmap[:3, 3] = np.array(t_a)
    scale = align_s
    return vilens_traj_sync, colmap_traj_sync, T_vilens_colmap, scale
