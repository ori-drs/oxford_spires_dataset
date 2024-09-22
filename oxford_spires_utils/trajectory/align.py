import json

import numpy as np
from evo.core import sync
from evo.core.metrics import APE


def align(vilens_traj, colmap_traj, output_dir):
    vilens_traj_sync, colmap_traj_sync = sync.associate_trajectories(vilens_traj, colmap_traj, max_diff=0.05)
    print(f"Trajectory pair length: {len(vilens_traj_sync.poses_se3)}")
    r_a, t_a, align_s = colmap_traj_sync.align(vilens_traj_sync, correct_scale=True)
    align_data = {
        "rotation": r_a.tolist(),
        "translation": t_a.tolist(),
        "scale": align_s,
    }
    ape = APE()
    ape.process_data((vilens_traj_sync, colmap_traj_sync))
    ape.get_all_statistics()
    print("APE statistics:")
    print(ape.get_all_statistics())
    mean_ape = ape.get_all_statistics()["mean"]
    if mean_ape > 0.1:
        print(f"APE mean {mean_ape} > 0.1, consider checking the alignment")
    print("Save align data to evo_align_results.json")
    with open(output_dir / "evo_align_results.json", "w") as f:
        json.dump(align_data, f, indent=2)
    print(json.dumps(align_data, indent=2))
    T_vilens_colmap = np.eye(4)
    T_vilens_colmap[:3, :3] = np.array(r_a) * align_s
    T_vilens_colmap[:3, 3] = np.array(t_a)
    return T_vilens_colmap
