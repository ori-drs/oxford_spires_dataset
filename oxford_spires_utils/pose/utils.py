import open3d as o3d
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from matplotlib import cm


class PosePlotter:
    def __init__(
        self,
        traj: PoseTrajectory3D,
        similarity_matrix=None,
        pose_viz_format="axis",
        axis_viz_size=1.0,
        colour=None,
        skip_every=0,
        min_sim=0.5,  # minimum similarity for a link to be drawn
    ):
        self.traj = traj
        self.pose_viz_format = pose_viz_format
        self.axis_viz_size = axis_viz_size
        self.colour = colour
        self.skip_every = skip_every  # 0 means no skipping
        self.similarity_matrix = similarity_matrix
        self.min_sim = min_sim

    def plot_poses(self, axis_viz_size=1.0):
        viz_objects = []
        viz_objects = self.visualise_poses(self.traj, viz_objects)
        # viz_objects = self.add_line_between_sequentail_poses(self.traj, viz_objects)
        if self.similarity_matrix is not None:
            viz_objects = self.add_similarity_link(self.traj, self.similarity_matrix, viz_objects, self.min_sim)
        o3d.visualization.draw_geometries(viz_objects)

    def visualise_poses(self, traj, viz_objects):
        assert isinstance(traj, PosePath3D)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        viz_objects.append(origin)
        # add axis to show the poses
        for i in range(traj.positions_xyz.shape[0]):
            if self.pose_viz_format == "axis":
                pose_viz = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis_viz_size)
            elif self.pose_viz_format == "point":
                pose_viz = o3d.geometry.TriangleMesh.create_sphere(radius=self.axis_viz_size)
            else:
                raise ValueError("Invalid pose_viz_format")
            T = traj.poses_se3[i]
            pose_viz.transform(T)
            if self.colour is not None:
                assert len(self.colour) == 3
                pose_viz.paint_uniform_color(self.colour)
            viz_objects.append(pose_viz)
        return viz_objects

    def add_line_between_sequentail_poses(self, traj, viz_objects):
        # lines between each timestamp to check if timestamps are ordered
        for i in range(traj.positions_xyz.shape[0] - 1):
            line_point_1 = traj.poses_se3[i][:3, 3]
            line_point_2 = traj.poses_se3[i + 1][:3, 3]
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector([line_point_1, line_point_2])
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            viz_objects.append(line)
        return viz_objects

    def add_similarity_link(self, traj, similarity_matrix, viz_objects, min_sim=0.4, cmap="viridis"):
        assert isinstance(traj, PosePath3D)
        assert similarity_matrix.shape[0] == similarity_matrix.shape[1]
        n = similarity_matrix.shape[0]
        assert n == traj.positions_xyz.shape[0]
        vmin, vmax = similarity_matrix.max() * min_sim, similarity_matrix.max()
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i, j]
                if similarity >= vmin:
                    C_i = traj.poses_se3[i][:3, 3]
                    C_j = traj.poses_se3[j][:3, 3]
                    link = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector([C_i, C_j]), lines=o3d.utility.Vector2iVector([[0, 1]])
                    )
                    norm_similarity = (similarity - vmin) / (vmax - vmin)
                    color = cm.get_cmap(cmap)(norm_similarity)[:3]
                    link.colors = o3d.utility.Vector3dVector([color])
                    viz_objects.append(link)
        return viz_objects
