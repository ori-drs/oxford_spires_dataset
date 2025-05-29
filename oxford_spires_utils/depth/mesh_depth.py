import matplotlib.pyplot as plt
import numpy as np
import pyrender


def get_depth_from_mesh(mesh, T_world_cam, save_path=None):
    scene = pyrender.Scene()
    scene.add(mesh)
    # Set up the camera
    # camera = pyrender.PerspectiveCamera(yfov=np.pi / 1.5, aspectRatio=1.0)
    camera = pyrender.IntrinsicsCamera(720, 540, 720, 540)
    camera_pose = np.array([[1.0, 0.0, 0.0, 0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0], [0.0, 0.0, 0.0, 1.0]])
    camera_pose = T_world_cam @ camera_pose
    scene.add(camera, pose=camera_pose)

    # Add a light
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi / 16.0, outerConeAngle=np.pi / 6.0)
    scene.add(light, pose=camera_pose)

    # Create a renderer
    r = pyrender.OffscreenRenderer(1440, 1080)

    # Render the depth
    depth = r.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)

    # Normalize depth for visualization
    max_depth = depth.max()
    min_depth = depth.min()
    depth_normalized = (depth - min_depth) / (max_depth - min_depth)

    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    # Save the depth image

    if save_path is not None:
        plt.imsave(save_path, depth_uint8, cmap="plasma")
    return depth


def get_vertices_visibility_from_mesh(vertices, depth, T_world_cam, image_size=400):
    # Create homogeneous coordinates
    vertices_homogeneous = np.column_stack((vertices, np.ones(len(vertices))))

    # Transform vertices to camera space
    vertices_camera = (np.linalg.inv(T_world_cam) @ vertices_homogeneous.T).T[:, :3]

    # Project to 2D
    # camera = pyrender.PerspectiveCamera(yfov=np.pi / 1.5, aspectRatio=1.0)
    img_height = 1080
    img_width = 1440
    fx = img_width / 2
    fy = img_height / 2
    cx = img_width / 2
    cy = img_height / 2
    # camera = pyrender.IntrinsicsCamera(
    intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    intrinsic_matrix = np.array(intrinsic_matrix)
    vertices_projected = (intrinsic_matrix @ vertices_camera.T).T
    # proj_matrix = camera.get_projection_matrix(1440, 1080)
    # vertices_projected = proj_matrix @ np.column_stack((vertices_camera, np.ones(len(vertices_camera)))).T
    # vertices_projected = vertices_projected.T

    vertices_projected[:, 0] /= vertices_projected[:, 2]
    vertices_projected[:, 1] /= vertices_projected[:, 2]
    pixels = np.floor(vertices_projected[:, :2] + 0.5).astype(np.int32)
    in_bounds = np.all((pixels >= 0) & (pixels < [img_width, img_height]), axis=1)
    pixels_in_image = pixels[in_bounds]
    depths_in_image = vertices_camera[in_bounds, 2]
    positive_depth_mask = depths_in_image > 0
    visible_depth_mask = depths_in_image < depth[pixels_in_image[:, 1], pixels_in_image[:, 0]]
    valid_depth_mask = positive_depth_mask & visible_depth_mask

    final_pixel_in_image = pixels_in_image[valid_depth_mask]
    final_depths_in_image = depths_in_image[valid_depth_mask]

    # get depth image
    depth_image = np.zeros((img_height, img_width))
    depth_image[final_pixel_in_image[:, 1], final_pixel_in_image[:, 0]] = final_depths_in_image

    # # Create a mask for vertices that are in bounds and closer than the rendered depth
    # in_image_visible_mask = vertex_depths[full_in_bounds] < depth[pixels[full_in_bounds, 1], pixels[full_in_bounds, 0]]

    full_mask = np.zeros(len(vertices), dtype=bool)
    visible_indices = np.where(in_bounds)[0]
    full_mask[visible_indices[valid_depth_mask]] = True

    return full_mask, depth_image
