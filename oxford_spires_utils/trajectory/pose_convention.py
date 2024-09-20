import numpy as np


class PoseConvention:
    """
    robotics (default)
    x forward, y left, z up

    computer vision / colmap
    x right, y down, z foward

    computer graphics / blender / nerf
    x right, y up, z backward
    """

    supported_conventions = ["robotics", "vision", "graphics"]
    graphics2robotics = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
    # robotics2blender = blender2robotics.T
    vision2robotics = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    # robotics2colmap = colmap2robotics.T
    vision2graphics = vision2robotics @ graphics2robotics.T
    # blender2colmap = colmap2blender.T

    # T_WB x T_BA, used to transform from B to A
    transforms = {
        "robotics": {"robotics": np.eye(4), "graphics": graphics2robotics.T, "vision": vision2robotics.T},
        "vision": {"robotics": vision2robotics, "graphics": vision2graphics, "vision": np.eye(4)},
        "graphics": {"robotics": graphics2robotics, "graphics": np.eye(4), "vision": vision2graphics.T},
    }

    @staticmethod
    def rename_convention(convention):
        if convention in ["nerf", "blender"]:
            convention = "graphics"
        elif convention in ["colmap"]:
            convention = "vision"
        assert convention in PoseConvention.supported_conventions, f"Unsupported convention: {convention}"
        return convention

    @staticmethod
    def get_transform(input_convention, output_convention):
        input_convention = PoseConvention.rename_convention(input_convention)
        output_convention = PoseConvention.rename_convention(output_convention)
        return PoseConvention.transforms[input_convention][output_convention]
