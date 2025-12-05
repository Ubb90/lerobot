import numpy as np    
import cv2

TARGET_IMAGE_SIZE = 128
RIGHT_CROP = 100
TARGET_CLOUD_POINTS = 1024

def crop_to_square_and_resize(self, image, target_size=None):
    """
    Crop image to square by removing left side, then resize to target_size.
    
    Args:
        image: Input image as numpy array (H, W, C)
        target_size: Target size for the square output (defaults to TARGET_IMAGE_SIZE)
        
    Returns:
        Processed image as numpy array (target_size, target_size, C)
    """
    if target_size is None:
        target_size = TARGET_IMAGE_SIZE
        
    h, w = image.shape[:2]
    right_side_crop = RIGHT_CROP

    # First crop from the right side
    image = image[:, :-right_side_crop]
    h, w = image.shape[:2]
    
    # Calculate square crop size (use minimum dimension)
    crop_size = min(h, w)
    
    # Calculate crop coordinates (crop from left side)
    if w > h:
        # Image is wider than tall, crop from left
        start_x = w - crop_size  # Start from right side to keep right portion
        start_y = 0
    else:
        # Image is taller than wide, crop from top
        start_x = 0
        start_y = 0
        
    # Crop to square
    cropped = image[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    # Resize to target size
    if len(cropped.shape) == 3:
        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    else:
        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        
    return resized

def create_point_cloud_from_depth(self, rgb_image, depth_image, fx, fy, cx, cy):
    """
    Create point cloud from RGB and depth images using camera intrinsics.
    
    Args:
        rgb_image: RGB image array (H, W, 3)
        depth_image: Depth image array (H, W)
        fx, fy: Focal lengths
        cx, cy: Principal point coordinates
        
    Returns:
        Point cloud array (1024, 6) with [x, y, z, r, g, b] format
    """
    h, w = depth_image.shape
    
    # Create coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert to 3D coordinates
    z = depth_image.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack XYZ coordinates (H, W, 3)
    xyz = np.stack([x, y, z], axis=-1)
    
    # Flatten to (H*W, 3)
    xyz_flat = xyz.reshape(-1, 3)
    rgb_flat = rgb_image.reshape(-1, 3).astype(np.float32)
    
    # Filter out invalid points (where depth is invalid)
    valid_mask = np.isfinite(xyz_flat).all(axis=1) & (xyz_flat[:, 2] > 0)
    
    xyz_valid = xyz_flat[valid_mask]
    rgb_valid = rgb_flat[valid_mask]
    
    # Sample or pad to exactly 1024 points
    num_valid = len(xyz_valid)
    target_points = TARGET_CLOUD_POINTS
    
    if num_valid == 0:
        # No valid points, return zeros
        print("Warning: No valid points in point cloud, returning zeros")
        return np.zeros((target_points, 6), dtype=np.float32)
    elif num_valid >= target_points:
        # Randomly sample 1024 points
        indices = np.random.choice(num_valid, target_points, replace=False)
        xyz_sampled = xyz_valid[indices]
        rgb_sampled = rgb_valid[indices]
    else:
        # Pad with repeated samples to reach 1024 points
        print("Warning: Not enough valid points in point cloud, padding with repeated samples")
        indices = np.random.choice(num_valid, target_points, replace=True)
        xyz_sampled = xyz_valid[indices]
        rgb_sampled = rgb_valid[indices]
    
    # Concatenate XYZ and RGB to create [1024, 6] array
    point_cloud = np.concatenate([xyz_sampled, rgb_sampled], axis=1)
    
    return point_cloud, xyz_sampled

def main():
    
    ## FIXME: load the full images (rgb_original and depth_original)
    
    # Process images to 128x128
    rgb_array_processed = crop_to_square_and_resize(rgb_array_full)
    depth_array_processed = crop_to_square_and_resize(depth_array_full)
    
    # Get camera parameters for point cloud creation
    if camera_info is not None:
        calib_params = camera_info.camera_configuration.calibration_parameters
        left_cam = calib_params.left_cam
        
        # Calculate scaling factors for intrinsic parameters
        original_height, original_width = rgb_array_full.shape[:2]
        
        # Account for right side crop
        crop_offset_right = RIGHT_CROP
        adjusted_width = original_width - crop_offset_right
        
        # Now calculate crop to square from the adjusted image
        crop_size = min(original_height, adjusted_width)
        scale_factor = TARGET_IMAGE_SIZE / crop_size
        
        # Adjust intrinsic parameters for cropped and resized image
        fx_scaled = left_cam.fx * scale_factor
        fy_scaled = left_cam.fy * scale_factor
        
        # Adjust principal point for cropping and scaling
        # The right crop doesn't affect cx since we crop from the right
        adjusted_cx = left_cam.cx  # cx remains the same after right crop
        
        if adjusted_width > original_height:
            # Cropped from left and right sides
            crop_offset_x_final = adjusted_width - crop_size
            cx_scaled = (adjusted_cx - crop_offset_x_final) * scale_factor
            cy_scaled = left_cam.cy * scale_factor
        else:
            # Cropped from top and bottom
            cx_scaled = adjusted_cx * scale_factor
            cy_scaled = left_cam.cy * scale_factor
        
        # Create point cloud from processed RGB and depth
        pointcloud_array_color, pointcloud_array = create_point_cloud_from_depth(
            rgb_array_processed, depth_array_processed, 
            fx_scaled, fy_scaled, cx_scaled, cy_scaled
        )
