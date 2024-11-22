import numpy as np
import cv2
import os
from pathlib import Path
import yaml

class ImageRectifier:
    def __init__(self,calib_file_path):
        # Load the YAML file
        with open(calib_file_path, 'r') as file:
            calib_data = yaml.safe_load(file)

        # Restructure the YAML data
        camera_model = calib_data['sensor']['camera_model']
        cameras = calib_data['sensor']['cameras']

        # Create the desired dictionary structure
        self.sensor_dict = {
            "camera_model": camera_model,
            "cameras": {camera['label']: camera for camera in cameras}
        }
        self._init_rectification_maps()
        
    def _init_rectification_maps(self):
        for cam_label,cam_data in self.sensor_dict['cameras'].items():
            balance = 0.0  
            scale_factor = 1.0 
            new_width = int(cam_data['image_width'] * scale_factor)
            new_height = int(cam_data['image_height'] * scale_factor)
            new_size = (new_width, new_height)

            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
               convert_to_intrinsic_matrix( np.array(cam_data['intrinsics'])), 
                np.array(cam_data['extra_params']),
                (new_width, new_height),
                np.eye(3),
                balance=balance,
                new_size=new_size
            )
            print(f"{cam_label} has:")
            print(new_K)
            
            self.sensor_dict['cameras'][cam_label]['map1'] , self.sensor_dict['cameras'][cam_label]['map2']  = cv2.fisheye.initUndistortRectifyMap(
               convert_to_intrinsic_matrix(np.array(cam_data['intrinsics'])),
               np.array(cam_data['extra_params']),
                np.eye(3), 
                new_K,
                new_size,
                cv2.CV_32FC1
            )
            self.sensor_dict['cameras'][cam_label]['rectified_width'] = new_width
            self.sensor_dict['cameras'][cam_label]['rectified_height'] = new_height
            
            print(f"Camera {cam_label} rectified image size: Width={new_width}, Height={new_height}, please copy this value \n")
            print("")


    def process_image(self, image, output_path, cam_label):
        # Remap the image
        rectified = cv2.remap(
            image, 
            self.sensor_dict['cameras'][cam_label]['map1'],
            self.sensor_dict['cameras'][cam_label]['map2'],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
            )

        # Save the rectified image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(str(output_path), rectified)

    def process_directory(self, input_dir, output_dir):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for cam_label, cam_data in self.sensor_dict['cameras'].items():
            path_parts = cam_data["topic"].split("/", 1) 
            img_subfolder = path_parts[0] + path_parts[1].replace("/", "_") 
            for img_path in input_path.rglob('*'):
                if img_path.parent == img_subfolder:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        print(f"Failed to read image: {img_path}")
                        continue
                    rel_path = img_path.relative_to(input_path)
                    output_file = output_path / rel_path
                    self.process_image(image, output_file, cam_label)
                    # print(f"Processed: {img_path.name}")

def convert_to_intrinsic_matrix(intrinsics):
    if len(intrinsics) != 4:
        raise ValueError("Intrinsics must be a list or array with exactly 4 elements: [fx, fy, cx, cy].")
    fx, fy, cx, cy = intrinsics
    intrinsic_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)
    return intrinsic_matrix



if __name__ == "__main__":
    '''
    Args:
    - calib_file_path: base calibration file 
    - distorted_image_input_path: distorted images folder directory 
    - rectified_image_output_path: output folder name to save undistorted rectified images
    '''
    
    calib_file_path = './config/sensor.yaml'
    distorted_image_input_path = '/data/2024-03-12-keble-college-01/images'
    rectified_image_output_path = '/data/2024-03-12-keble-college-01/test/images_rectified'

    # Initialize rectifier
    rectifier = ImageRectifier(calib_file_path)
    rectifier.process_directory(distorted_image_input_path, rectified_image_output_path)
    print("Please copy and paste output rectified intrinsics")
