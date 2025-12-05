# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple script to control a robot from teleoperation with recording capabilities.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true \
    --record_path=/path/to/recordings
```

Example with ZED camera (images processed to 128x128):

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true \
    --record_path=/path/to/recordings \
    --use_zed_camera=true \
    --zed_resolution=VGA \
    --zed_fps=30
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/tty.usbmodem5A460828611 \
  --teleop.right_arm_port=/dev/tty.usbmodem5A460826981 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import os
import time
import math
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import numpy as np
import rerun as rr
import json
from PIL import Image
import cv2

# ROS2 imports for tf tree publishing
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from tf2_ros import TransformBroadcaster
    from geometry_msgs.msg import TransformStamped, PoseStamped, Quaternion, Point
    from sensor_msgs.msg import JointState, Image as RosImage, PointCloud2, PointField, CameraInfo
    from std_msgs.msg import Header, String as RosString
    import threading
    import struct
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("Warning: ROS2 not available. TF tree publishing will be disabled.")

# Global variable for target image size
TARGET_IMAGE_SIZE = 128
TARGET_CLOUD_POINTS = 1024
RIGHT_CROP = 100  # Number of pixels to crop from the right side

# Global URDF and kinematic chain (loaded once at startup)
URDF_ROBOT = None
KINEMATIC_CHAIN = []
END_EFFECTOR_LINK = 'gripper_ee'

# Mapping from lerobot joint names to URDF joint names
LEROBOT_TO_URDF_JOINT_NAMES = {
    'Shoulder_Rotation': 'shoulder_pan',
    'Shoulder_Pitch': 'shoulder_lift',
    'Elbow': 'elbow_flex',
    'Wrist_Pitch': 'wrist_flex',
    'Wrist_Roll': 'wrist_roll',
    'Gripper': 'gripper',
}

def create_transform_matrix(x, y, z, roll, pitch, yaw):
    """Create a 4x4 transformation matrix from translation and rotation (RPY)."""
    cos_r, sin_r = math.cos(roll), math.sin(roll)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    
    # Roll-Pitch-Yaw rotation matrix
    R = np.array([
        [cos_y*cos_p, cos_y*sin_p*sin_r - sin_y*cos_r, cos_y*sin_p*cos_r + sin_y*sin_r],
        [sin_y*cos_p, sin_y*sin_p*sin_r + cos_y*cos_r, sin_y*sin_p*cos_r - cos_y*sin_r],
        [-sin_p, cos_p*sin_r, cos_p*cos_r]
    ])
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def create_rotation_matrix(axis, angle):
    """Create a 4x4 rotation matrix around given axis by given angle (Rodrigues formula)."""
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)  # Normalize axis
    
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    ux, uy, uz = axis
    
    # Rodrigues' rotation formula
    R = np.array([
        [cos_a + ux*ux*(1-cos_a), ux*uy*(1-cos_a) - uz*sin_a, ux*uz*(1-cos_a) + uy*sin_a],
        [uy*ux*(1-cos_a) + uz*sin_a, cos_a + uy*uy*(1-cos_a), uy*uz*(1-cos_a) - ux*sin_a],
        [uz*ux*(1-cos_a) - uy*sin_a, uz*uy*(1-cos_a) + ux*sin_a, cos_a + uz*uz*(1-cos_a)]
    ])
    
    # Create 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = R
    return T


def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion (w, x, y, z) - WXYZ format."""
    trace = np.trace(R)
    
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z], dtype=np.float32)


class ROS2TFTreePublisher:
    """
    ROS2 tf tree publisher for robot joint states and end-effector pose.
    Publishes:
    - JointState messages with current joint positions
    - TransformStamped messages for each joint in the tf tree
    - PoseStamped message for the end-effector pose
    """
    
    def __init__(self, robot_name: str = "robot", base_frame: str = "base_link", 
                 ee_frame: str = "gripper_ee", urdf_path: str | None = None):
        """
        Initialize the TF tree publisher.
        
        Args:
            robot_name: Name of the robot (used for ROS node and topic names)
            base_frame: Name of the robot base frame
            ee_frame: Name of the end-effector frame
            urdf_path: Path to URDF for forward kinematics computation
        """
        if not ROS2_AVAILABLE:
            print("Warning: ROS2 not available. TF tree publisher will be disabled.")
            self.ros_node = None
            self.tf_broadcaster = None
            self.executor = None
            self.executor_thread = None
            return
        
        self.robot_name = robot_name
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.urdf_path = urdf_path
        self.enabled = False
        self.spin_thread = None
        self.keep_spinning = True
        
        try:
            # Initialize ROS2 if not already done
            try:
                if not rclpy.ok():
                    rclpy.init()
            except RuntimeError:
                # Already initialized
                pass
            
            # Create the ROS2 node
            self.ros_node = Node(f"{robot_name}_tf_publisher")
            self.tf_broadcaster = TransformBroadcaster(self.ros_node)
            
            # Create publishers
            self.joint_state_pub = self.ros_node.create_publisher(
                JointState, f"/{robot_name}/joint_states", 10
            )
            self.hardware_joint_state_pub = self.ros_node.create_publisher(
                JointState, "/hardware/joint_states", 10
            )
            self.ee_pose_pub = self.ros_node.create_publisher(
                PoseStamped, f"/{robot_name}/ee_pose", 10
            )
            
            # Camera publishers
            self.rgb_image_pub = self.ros_node.create_publisher(
                RosImage, f"/{robot_name}/camera/rgb/image_raw", 10
            )
            self.depth_image_pub = self.ros_node.create_publisher(
                RosImage, f"/{robot_name}/camera/depth/image_raw", 10
            )
            self.pointcloud_pub = self.ros_node.create_publisher(
                PointCloud2, f"/{robot_name}/camera/pointcloud", 10
            )
            
            # Calibration parameters publisher
            self.camera_calib_pub = self.ros_node.create_publisher(
                CameraInfo, "/dataset/camera_calibration", 10
            )
            
            # Store joint information
            self.joint_names = []
            self.joint_positions = []
            self.joint_velocities = []
            self.joint_efforts = []
            
            # Start background thread to spin the node
            self.spin_thread = threading.Thread(target=self._spin_node, daemon=True)
            self.spin_thread.start()
            
            # Give thread time to start
            time.sleep(0.1)
            
            self.enabled = True
            print(f"ROS2 TF tree publisher initialized for {robot_name}")
            print(f"  Publishing to: /{robot_name}/joint_states, /{robot_name}/ee_pose")
            
        except Exception as e:
            print(f"ERROR: Failed to initialize ROS2 TF tree publisher: {e}")
            import traceback
            traceback.print_exc()
            self.enabled = False
            self.ros_node = None
            self.tf_broadcaster = None
    
    def _spin_node(self):
        """Background thread to continuously spin the ROS2 node."""
        try:
            while self.keep_spinning:
                rclpy.spin_once(self.ros_node, timeout_sec=0.01)
        except Exception as e:
            print(f"Error spinning ROS2 node: {e}")
    
    def update_joint_state(self, joint_positions_dict: dict, timestamp: float | None = None):
        """
        Update and publish joint state and TF transforms.
        
        Args:
            joint_positions_dict: Dictionary with joint names as keys and positions (in radians) as values
            timestamp: Unix timestamp (defaults to current time)
        """
        if not self.enabled or self.ros_node is None:
            return
        
        try:
            if timestamp is None:
                from datetime import datetime
                timestamp = datetime.now().timestamp()
            
            # Create timestamp for ROS messages
            from builtin_interfaces.msg import Time
            secs = int(timestamp)
            nsecs = int((timestamp - secs) * 1e9)
            
            # Update joint names and positions
            self.joint_names = list(joint_positions_dict.keys())
            self.joint_positions = [float(joint_positions_dict[name]) for name in self.joint_names]
            self.joint_velocities = [0.0] * len(self.joint_names)  # Not available from observation
            self.joint_efforts = [0.0] * len(self.joint_names)  # Not available from observation
            
            # Publish JointState
            joint_state_msg = JointState()
            joint_state_msg.header.stamp.sec = secs
            joint_state_msg.header.stamp.nanosec = nsecs
            joint_state_msg.header.frame_id = self.base_frame
            joint_state_msg.name = self.joint_names
            joint_state_msg.position = self.joint_positions
            joint_state_msg.velocity = self.joint_velocities
            joint_state_msg.effort = self.joint_efforts
            
            self.joint_state_pub.publish(joint_state_msg)
            self.hardware_joint_state_pub.publish(joint_state_msg)
            
            
            # Publish end-effector pose
            self._publish_ee_pose(joint_positions_dict, secs, nsecs)
            
        except Exception as e:
            print(f"Warning: Error updating joint state: {e}")
            
    def _publish_ee_pose(self, joint_positions_dict: dict, secs: int, nsecs: int):
        """
        Publish all joint transforms to build the complete TF tree and the ee_pose.
        
        Args:
            joint_positions_dict: Dictionary with joint names and positions
            secs: Seconds component of timestamp
            nsecs: Nanoseconds component of timestamp
        """
        try:
            if self.urdf_path is None:
                return
            
            # Publish all joint transforms (this now includes gripper_ee_joint via recursion)
            self._publish_joint_transforms(joint_positions_dict, secs, nsecs)
            
            # Publish ee_pose - query the gripper_ee frame from the TF tree
            # The gripper_ee frame is now published by _publish_joint_transforms
            # We just need to compute its position from the joint angles
            ee_pose = self._compute_gripper_ee_pose(joint_positions_dict, secs, nsecs)
            if ee_pose is not None:
                self.ee_pose_pub.publish(ee_pose)
            
        except Exception as e:
            print(f"Warning: Error publishing joint transforms: {e}")
    
    def _compute_gripper_ee_pose(self, joint_positions_dict: dict, secs: int, nsecs: int):
        """
        Compute the gripper_ee PoseStamped by tracking transforms through the TF tree.
        Uses the URDF to walk from base_link to gripper_ee, applying all transformations.
        
        Returns:
            PoseStamped message or None if computation fails
        """
        try:
            # Use the shared function to compute gripper_ee pose
            ee_data = compute_gripper_ee_pose(joint_positions_dict, self.urdf_path)
            
            if not ee_data:
                return None
            
            # Create PoseStamped message
            pose_msg = PoseStamped()
            pose_msg.header.stamp.sec = secs
            pose_msg.header.stamp.nanosec = nsecs
            pose_msg.header.frame_id = self.base_frame
            
            # Set position
            pose_msg.pose.position.x = float(ee_data['position'][0])
            pose_msg.pose.position.y = float(ee_data['position'][1])
            pose_msg.pose.position.z = float(ee_data['position'][2])
            
            # Set orientation (quaternion) - convert from WXYZ to XYZW for ROS
            pose_msg.pose.orientation.x = float(ee_data['quaternion'][1])
            pose_msg.pose.orientation.y = float(ee_data['quaternion'][2])
            pose_msg.pose.orientation.z = float(ee_data['quaternion'][3])
            pose_msg.pose.orientation.w = float(ee_data['quaternion'][0])
            
            return pose_msg
            
        except Exception as e:
            print(f"Warning: Error computing gripper_ee pose: {e}")
            return None
    
    def _publish_joint_transforms(self, joint_positions_dict: dict, secs: int, nsecs: int):
        """
        Publish TF transforms for each joint in the kinematic chain.
        This builds the complete robot TF tree with intermediate frames.
        Each joint transform is published relative to its parent frame.
        
        Args:
            joint_positions_dict: Dictionary with joint names and positions
            secs: Seconds component of timestamp
            nsecs: Nanoseconds component of timestamp
        """
        global URDF_ROBOT, KINEMATIC_CHAIN
        
        if not URDF_ROBOT or not KINEMATIC_CHAIN:
            return
        
        # First, publish all joints in the kinematic chain
        for joint in KINEMATIC_CHAIN:
            self._publish_single_joint_transform(joint, joint_positions_dict, secs, nsecs)
        
        # Then, find and publish any fixed joints that are children of the last joint in the chain
        # This includes gripper_ee_joint and any other fixed joints
        if KINEMATIC_CHAIN:
            last_link = KINEMATIC_CHAIN[-1].child
            self._publish_fixed_joint_descendants(last_link, joint_positions_dict, secs, nsecs)
    
    def _publish_single_joint_transform(self, joint, joint_positions_dict: dict, secs: int, nsecs: int):
        """Publish transform for a single joint. Returns (position, quaternion) tuple."""
        try:
            # Get joint origin
            origin = joint.origin
            if not origin:
                return None, None
            
            dx, dy, dz = origin.xyz
            roll, pitch, yaw = origin.rpy
            
            # Parent frame is either base or the previous joint's child
            parent_frame = joint.parent
            child_frame = joint.child
            
            # Create base transformation matrix for joint origin
            transform_msg = TransformStamped()
            transform_msg.header.stamp.sec = secs
            transform_msg.header.stamp.nanosec = nsecs
            transform_msg.header.frame_id = parent_frame
            transform_msg.child_frame_id = child_frame
            
            # Set translation (origin position)
            transform_msg.transform.translation.x = float(dx)
            transform_msg.transform.translation.y = float(dy)
            transform_msg.transform.translation.z = float(dz)
            
            # Set rotation
            if joint.type == 'revolute':
                # For revolute joints: origin RPY + joint angle rotation
                axis = np.array(joint.axis, dtype=np.float64)
                axis = axis / np.linalg.norm(axis)
                
                # Get joint angle with default 0.0
                urdf_joint_name = LEROBOT_TO_URDF_JOINT_NAMES.get(joint.name, joint.name)
                joint_angle = joint_positions_dict.get(urdf_joint_name, 0.0)
                
                # First apply origin RPY
                cos_r, sin_r = math.cos(roll), math.sin(roll)
                cos_p, sin_p = math.cos(pitch), math.sin(pitch)
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                
                R_origin = np.array([
                    [cos_y*cos_p, cos_y*sin_p*sin_r - sin_y*cos_r, cos_y*sin_p*cos_r + sin_y*sin_r],
                    [sin_y*cos_p, sin_y*sin_p*sin_r + cos_y*cos_r, sin_y*sin_p*cos_r - cos_y*sin_r],
                    [-sin_p, cos_p*sin_r, cos_p*cos_r]
                ])
                
                # Then apply joint rotation
                R_joint = create_rotation_matrix(axis, joint_angle)[:3, :3]
                
                # Combine rotations
                R_total = R_origin @ R_joint
                quat = rotation_matrix_to_quaternion(R_total)
            else:
                # For fixed joints: just use origin RPY
                cos_r, sin_r = math.cos(roll), math.sin(roll)
                cos_p, sin_p = math.cos(pitch), math.sin(pitch)
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                
                R_fixed = np.array([
                    [cos_y*cos_p, cos_y*sin_p*sin_r - sin_y*cos_r, cos_y*sin_p*cos_r + sin_y*sin_r],
                    [sin_y*cos_p, sin_y*sin_p*sin_r + cos_y*cos_r, sin_y*sin_p*cos_r - cos_y*sin_r],
                    [-sin_p, cos_p*sin_r, cos_p*cos_r]
                ])
                quat = rotation_matrix_to_quaternion(R_fixed)
            
            # Set rotation from quaternion - convert from WXYZ to XYZW for ROS
            transform_msg.transform.rotation.x = float(quat[1])
            transform_msg.transform.rotation.y = float(quat[2])
            transform_msg.transform.rotation.z = float(quat[3])
            transform_msg.transform.rotation.w = float(quat[0])
            
            # Publish this transform
            self.tf_broadcaster.sendTransform(transform_msg)
            
            # Return the orientation quaternion for this joint
            return quat
            
        except Exception as e:
            print(f"Warning: Error publishing transform for joint {joint.name}: {e}")
            return None
    
    def _publish_fixed_joint_descendants(self, parent_link: str, joint_positions_dict: dict, secs: int, nsecs: int):
        """Recursively publish all fixed joints that descend from a given parent link."""
        global URDF_ROBOT
        
        if not URDF_ROBOT:
            return
        
        # Find all joints whose parent is this link
        for joint in URDF_ROBOT.joints:
            if joint.parent == parent_link and joint.type == 'fixed':
                # Publish this fixed joint
                self._publish_single_joint_transform(joint, joint_positions_dict, secs, nsecs)
                
                # Recursively publish any fixed joints that descend from this joint's child
                self._publish_fixed_joint_descendants(joint.child, joint_positions_dict, secs, nsecs)
    
    def publish_camera_data(self, zed_data: dict, timestamp: float | None = None):
        """
        Publish camera data (RGB, depth, pointcloud) to ROS2 topics.
        
        Args:
            zed_data: Dictionary containing 'rgb', 'depth', and 'pointcloud_color' (processed images, not original)
            timestamp: Unix timestamp (defaults to current time)
        """
        if not self.enabled or self.ros_node is None:
            return
        
        try:
            if timestamp is None:
                from datetime import datetime
                timestamp = datetime.now().timestamp()
            
            # Create timestamp for ROS messages
            from builtin_interfaces.msg import Time
            secs = int(timestamp)
            nsecs = int((timestamp - secs) * 1e9)
            
            # Create header
            header = Header()
            header.stamp.sec = secs
            header.stamp.nanosec = nsecs
            header.frame_id = "camera_link"
            
            # Publish RGB image
            if 'rgb' in zed_data and zed_data['rgb'] is not None:
                rgb_msg = RosImage()
                rgb_msg.header = header
                rgb_msg.height = zed_data['rgb'].shape[0]
                rgb_msg.width = zed_data['rgb'].shape[1]
                rgb_msg.encoding = "rgb8"
                rgb_msg.is_bigendian = 0
                rgb_msg.step = zed_data['rgb'].shape[1] * 3
                rgb_msg.data = zed_data['rgb'].tobytes()
                self.rgb_image_pub.publish(rgb_msg)
            
            # Publish depth image
            if 'depth' in zed_data and zed_data['depth'] is not None:
                depth_msg = RosImage()
                depth_msg.header = header
                depth_msg.height = zed_data['depth'].shape[0]
                depth_msg.width = zed_data['depth'].shape[1]
                depth_msg.encoding = "32FC1"  # 32-bit float
                depth_msg.is_bigendian = 0
                depth_msg.step = zed_data['depth'].shape[1] * 4
                depth_msg.data = zed_data['depth'].astype(np.float32).tobytes()
                self.depth_image_pub.publish(depth_msg)
            
            # Publish point cloud
            if 'pointcloud_color' in zed_data and zed_data['pointcloud_color'] is not None:
                pc_msg = PointCloud2()
                pc_msg.header = header
                
                # Point cloud has shape (N, 6) with [x, y, z, r, g, b]
                pointcloud = zed_data['pointcloud_color']
                pc_msg.height = 1
                pc_msg.width = pointcloud.shape[0]
                
                # Define fields for PointCloud2
                fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
                ]
                pc_msg.fields = fields
                pc_msg.is_bigendian = False
                pc_msg.point_step = 16  # 4 floats * 4 bytes
                pc_msg.row_step = pc_msg.point_step * pc_msg.width
                pc_msg.is_dense = True
                
                # Pack RGB into a single float (as is convention in ROS)
                cloud_data = []
                for i in range(pointcloud.shape[0]):
                    x, y, z = pointcloud[i, 0:3]
                    r, g, b = pointcloud[i, 3:6].astype(np.uint8)
                    # Pack RGB into uint32, then reinterpret as float
                    rgb_uint = (int(r) << 16) | (int(g) << 8) | int(b)
                    rgb_float = struct.unpack('f', struct.pack('I', rgb_uint))[0]
                    cloud_data.append(struct.pack('ffff', x, y, z, rgb_float))
                
                pc_msg.data = b''.join(cloud_data)
                self.pointcloud_pub.publish(pc_msg)
                
        except Exception as e:
            print(f"Warning: Error publishing camera data: {e}")
    
    def publish_camera_calibration(self, camera_params: dict):
        """
        Publish camera calibration parameters to /dataset/camera_calibration topic using CameraInfo message.
        
        Args:
            camera_params: Dictionary containing camera intrinsic parameters with keys:
                - fx, fy, cx, cy: intrinsic parameters
                - k1, k2, k3, p1, p2: distortion coefficients
                - resolution: dict with 'width' and 'height'
                - baseline: stereo baseline (optional)
        """
        if not self.enabled or self.ros_node is None:
            print("Warning: Cannot publish camera calibration - ROS2 node not enabled or not available")
            return
        
        try:
            # Create CameraInfo message
            camera_info_msg = CameraInfo()
            
            # Set header
            camera_info_msg.header.frame_id = "camera_link"
            camera_info_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
            
            # Set image dimensions
            camera_info_msg.height = camera_params['resolution']['height']
            camera_info_msg.width = camera_params['resolution']['width']
            
            # Set distortion model
            camera_info_msg.distortion_model = "plumb_bob"
            
            # Set distortion coefficients [k1, k2, p1, p2, k3]
            camera_info_msg.d = [
                camera_params['k1'],
                camera_params['k2'],
                camera_params['p1'],
                camera_params['p2'],
                camera_params['k3']
            ]
            
            # Set camera intrinsic matrix K (3x3)
            # [fx  0 cx]
            # [ 0 fy cy]
            # [ 0  0  1]
            camera_info_msg.k = [
                camera_params['fx'], 0.0, camera_params['cx'],
                0.0, camera_params['fy'], camera_params['cy'],
                0.0, 0.0, 1.0
            ]
            
            # Set rectification matrix R (3x3) - identity for monocular camera
            camera_info_msg.r = [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            ]
            
            # Set projection matrix P (3x4)
            # [fx'  0  cx' Tx]
            # [ 0  fy' cy' Ty]
            # [ 0   0   1   0]
            # For monocular camera, Tx = Ty = 0
            camera_info_msg.p = [
                camera_params['fx'], 0.0, camera_params['cx'], 0.0,
                0.0, camera_params['fy'], camera_params['cy'], 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
            
            # Set binning (no binning)
            camera_info_msg.binning_x = 0
            camera_info_msg.binning_y = 0
            
            # Set ROI (region of interest) - full image
            camera_info_msg.roi.x_offset = 0
            camera_info_msg.roi.y_offset = 0
            camera_info_msg.roi.height = 0  # 0 means full resolution
            camera_info_msg.roi.width = 0   # 0 means full resolution
            camera_info_msg.roi.do_rectify = False
            
            # Publish the message
            self.camera_calib_pub.publish(camera_info_msg)
            
        except Exception as e:
            print(f"Warning: Error publishing camera calibration: {e}")
            import traceback
            traceback.print_exc()
    
    def shutdown(self):
        """Cleanup resources."""
        if self.enabled and self.ros_node is not None:
            try:
                self.ros_node.destroy_node()
                self.enabled = False
            except Exception as e:
                print(f"Warning: Error shutting down TF publisher: {e}")


def load_urdf(urdf_path):
    """Load and parse the URDF file, build kinematic chain."""
    global URDF_ROBOT, KINEMATIC_CHAIN, END_EFFECTOR_LINK
    
    try:
        from urdf_parser_py.urdf import URDF
    except ImportError:
        print("Warning: urdf_parser_py not available. Cannot load URDF for FK.")
        return False
    
    if not os.path.exists(urdf_path):
        print(f"Warning: URDF file not found at {urdf_path}. FK will not work.")
        return False
    
    try:
        URDF_ROBOT = URDF.from_xml_file(urdf_path)
        build_kinematic_chain()
        return True
    except Exception as e:
        print(f"Warning: Failed to load URDF: {e}")
        return False


def build_kinematic_chain():
    """Build kinematic chain from base_link to end effector."""
    global URDF_ROBOT, KINEMATIC_CHAIN, END_EFFECTOR_LINK
    
    if not URDF_ROBOT:
        return
    
    KINEMATIC_CHAIN = []
    current_link = END_EFFECTOR_LINK
    
    # Work backwards from end effector to base
    while current_link:
        parent_joint = None
        for joint in URDF_ROBOT.joints:
            if joint.child == current_link:
                parent_joint = joint
                break
        
        if parent_joint:
            KINEMATIC_CHAIN.insert(0, parent_joint)
            current_link = parent_joint.parent
            if current_link == 'track_body':
                break
        else:
            break


def compute_gripper_ee_pose(joint_positions_dict, urdf_path):
    """
    Compute gripper_ee frame pose from joint positions using forward kinematics.
    This includes ALL joints from base_link to gripper_ee (including the gripper_ee_joint).
    
    Args:
        joint_positions_dict: Dict with joint names as keys and angles (in radians) as values
        urdf_path: Path to the URDF file for loading kinematic chain
        
    Returns:
        Dict with 'position' (x,y,z) and 'quaternion' (w,x,y,z) keys - quaternion in WXYZ format
    """
    global URDF_ROBOT, KINEMATIC_CHAIN
    
    # Load URDF if not already loaded
    if not KINEMATIC_CHAIN:
        if not load_urdf(urdf_path):
            raise RuntimeError("URDF could not be loaded for FK computation.")
    
    # Build a map of link -> joint connections for faster lookup
    link_to_joint = {}
    for joint in URDF_ROBOT.joints:
        link_to_joint[joint.child] = joint
    
    # Walk from gripper_ee back to base_link to build the chain
    chain = []
    current = 'gripper_ee'
    while current != 'base_link' and current in link_to_joint:
        chain.insert(0, link_to_joint[current])
        current = link_to_joint[current].parent
    
    if current != 'base_link':
        raise RuntimeError(f"Could not trace gripper_ee back to base_link, stopped at {current}")
    
    # Now apply all transformations from base_link to gripper_ee
    transform = np.eye(4)
    for joint in chain:
        origin = joint.origin
        if not origin:
            continue
        
        dx, dy, dz = origin.xyz
        roll, pitch, yaw = origin.rpy
        
        # Build the rotation matrix
        if joint.type == 'revolute':
            # For revolute joints: origin RPY + joint angle rotation
            cos_r, sin_r = math.cos(roll), math.sin(roll)
            cos_p, sin_p = math.cos(pitch), math.sin(pitch)
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            
            R_origin = np.array([
                [cos_y*cos_p, cos_y*sin_p*sin_r - sin_y*cos_r, cos_y*sin_p*cos_r + sin_y*sin_r],
                [sin_y*cos_p, sin_y*sin_p*sin_r + cos_y*cos_r, sin_y*sin_p*cos_r - cos_y*sin_r],
                [-sin_p, cos_p*sin_r, cos_p*cos_r]
            ])
            
            axis = np.array(joint.axis, dtype=np.float64)
            axis = axis / np.linalg.norm(axis)
            urdf_joint_name = LEROBOT_TO_URDF_JOINT_NAMES.get(joint.name, joint.name)
            joint_angle = joint_positions_dict.get(urdf_joint_name, 0.0)
            R_joint = create_rotation_matrix(axis, joint_angle)[:3, :3]
            R_total = R_origin @ R_joint
        else:
            # For fixed joints: just use origin RPY
            cos_r, sin_r = math.cos(roll), math.sin(roll)
            cos_p, sin_p = math.cos(pitch), math.sin(pitch)
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            
            R_total = np.array([
                [cos_y*cos_p, cos_y*sin_p*sin_r - sin_y*cos_r, cos_y*sin_p*cos_r + sin_y*sin_r],
                [sin_y*cos_p, sin_y*sin_p*sin_r + cos_y*cos_r, sin_y*sin_p*cos_r - cos_y*sin_r],
                [-sin_p, cos_p*sin_r, cos_p*cos_r]
            ])
        
        # Build 4x4 transform and accumulate
        T = np.eye(4)
        T[:3, :3] = R_total
        T[:3, 3] = [dx, dy, dz]
        transform = transform @ T
    
    # Extract position and orientation
    position = transform[:3, 3].astype(np.float32)
    rotation_matrix = transform[:3, :3]
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    
    return {
        'position': position,
        'quaternion': quaternion
    }


def compute_end_effector_pose(joint_positions_dict, urdf_path):
    """
    Legacy function for backward compatibility.
    Now just calls compute_gripper_ee_pose which includes the gripper_ee_joint offset.
    
    Args:
        joint_positions_dict: Dict with joint names as keys and angles (in radians) as values
        urdf_path: Path to the URDF file for loading kinematic chain
        
    Returns:
        Dict with 'position' (x,y,z) and 'quaternion' (w,x,y,z) keys - quaternion in WXYZ format
    """
    return compute_gripper_ee_pose(joint_positions_dict, urdf_path)



try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    sl = None
    print("Warning: pyzed not available. ZED camera functionality will be disabled.")

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


class ZEDCamera:
    """ZED Camera wrapper using pyzed library."""
    
    def __init__(self, camera_id=0, resolution="HD720", fps=30):
        if not ZED_AVAILABLE:
            raise ImportError("pyzed is not available. Please install it to use ZED camera.")
        
        # Initialize only if pyzed is available
        self.camera = sl.Camera()
        self.camera_id = camera_id
        
        # Create initialization parameters
        self.init_params = sl.InitParameters()
        
        # Set resolution using string mapping to avoid linter issues
        resolution_map = {
            "VGA": getattr(sl.RESOLUTION, 'VGA', None),
            "HD720": getattr(sl.RESOLUTION, 'HD720', None),
            "HD1080": getattr(sl.RESOLUTION, 'HD1080', None), 
            "HD2K": getattr(sl.RESOLUTION, 'HD2K', None)
        }
        self.init_params.camera_resolution = resolution_map.get(resolution, getattr(sl.RESOLUTION, 'HD720', None))
            
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = getattr(sl.DEPTH_MODE, 'NEURAL', None)
        self.init_params.coordinate_units = getattr(sl.UNIT, 'METER', None)
        self.init_params.depth_stabilization = 1  # Use 1 instead of True for boolean parameters
        
        # Runtime parameters
        self.runtime_params = sl.RuntimeParameters()
        
        # Images
        self.rgb_image = sl.Mat()
        self.depth_image = sl.Mat()
        # Note: We'll recreate point cloud from RGB and depth instead of using ZED's native point cloud
        
        # Camera info
        self.camera_info = None
        self.is_connected = False
        
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
        valid = np.isfinite(xyz_flat).all(axis=1) & (xyz_flat[:, 2] > 0)
        
        # Apply spatial bounding box filter
        PC_CROP_BOUNDS = {
            "x": (0.0, 0.32),
            "y": (-0.2, 0.25),
            "z": (0.75, 1.10),
        }
        
        bx0, bx1 = PC_CROP_BOUNDS["x"]
        by0, by1 = PC_CROP_BOUNDS["y"]
        bz0, bz1 = PC_CROP_BOUNDS["z"]
        in_box = (
            (xyz_flat[:, 0] >= bx0) & (xyz_flat[:, 0] <= bx1) &
            (xyz_flat[:, 1] >= by0) & (xyz_flat[:, 1] <= by1) &
            (xyz_flat[:, 2] >= bz0) & (xyz_flat[:, 2] <= bz1)
        )
        valid &= in_box
        
        xyz_valid = xyz_flat[valid]
        rgb_valid = rgb_flat[valid]
        
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
        
    def connect(self):
        """Connect to the ZED camera."""
        if self.camera_id is not None and self.camera_id != 0:
            self.init_params.set_from_serial_number(int(self.camera_id))
            
        print(f"Attempting to connect to ZED camera (ID: {self.camera_id})...")
        err = self.camera.open(self.init_params)
        success_code = getattr(sl.ERROR_CODE, 'SUCCESS', None)
        if err != success_code:
            print(f"ZED camera connection failed with error code: {err}")
            # Try to get more specific error information
            if hasattr(sl.ERROR_CODE, 'CAMERA_NOT_DETECTED'):
                if err == getattr(sl.ERROR_CODE, 'CAMERA_NOT_DETECTED'):
                    raise RuntimeError(f"ZED camera not detected. Make sure the camera is connected and not being used by another application.")
            raise RuntimeError(f"Failed to open ZED camera: {err}")
        
        # Get camera information
        self.camera_info = self.camera.get_camera_information()
        self.is_connected = True
        print(f"ZED Camera connected successfully!")
        print(f"  Model: {self.camera_info.camera_model}")
        print(f"  Serial: {self.camera_info.serial_number}")
        print(f"  Resolution: {self.camera_info.camera_configuration.resolution.width}x{self.camera_info.camera_configuration.resolution.height}")
        print(f"  Firmware: {getattr(self.camera_info, 'camera_firmware_version', 'Unknown')}")
        
    def disconnect(self):
        """Disconnect from the ZED camera."""
        if self.is_connected:
            self.camera.close()
            self.is_connected = False
            
    def capture(self):
        """Capture RGB, depth, and point cloud data, processed to 128x128."""
        if not self.is_connected:
            raise RuntimeError("Camera is not connected")
        
        success_code = getattr(sl.ERROR_CODE, 'SUCCESS', None)
        grab_result = self.camera.grab(self.runtime_params)
        
        if grab_result == success_code:
            # Retrieve RGB image
            left_view = getattr(sl.VIEW, 'LEFT', None)
            self.camera.retrieve_image(self.rgb_image, left_view)
            
            # Retrieve depth image
            depth_measure = getattr(sl.MEASURE, 'DEPTH', None)
            self.camera.retrieve_measure(self.depth_image, depth_measure)
            
            # Convert to numpy arrays
            bgra_array_full = self.rgb_image.get_data()  # ZED provides BGRA format
            depth_array_full = self.depth_image.get_data()
            
            # Convert BGRA to RGB
            rgb_array_full = cv2.cvtColor(bgra_array_full, cv2.COLOR_BGRA2RGB)
                        
            # Process images to 128x128
            rgb_array_processed = self.crop_to_square_and_resize(rgb_array_full)
            depth_array_processed = self.crop_to_square_and_resize(depth_array_full)
            
            # Get camera parameters for point cloud creation
            if self.camera_info is not None:
                calib_params = self.camera_info.camera_configuration.calibration_parameters
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
                pointcloud_array_color, pointcloud_array = self.create_point_cloud_from_depth(
                    rgb_array_processed, depth_array_processed, 
                    fx_scaled, fy_scaled, cx_scaled, cy_scaled
                )
            else:
                raise RuntimeError("Camera info is not available for point cloud creation")
                        
            return {
                'rgb': rgb_array_processed,
                'depth': depth_array_processed,
                'pointcloud_color': pointcloud_array_color,
                'pointcloud': pointcloud_array,
                'rgb_original': rgb_array_full,
                'depth_original': depth_array_full
            }
        else:
            print(f"ZED camera grab failed with error code: {grab_result}")
            return None
            
    def get_camera_parameters(self):
        """Get camera intrinsic parameters."""
        if not self.is_connected or self.camera_info is None:
            raise RuntimeError("Camera is not connected")
            
        calib_params = self.camera_info.camera_configuration.calibration_parameters
        left_cam = calib_params.left_cam
        
        return {
            'fx': float(left_cam.fx),
            'fy': float(left_cam.fy),
            'cx': float(left_cam.cx),
            'cy': float(left_cam.cy),
            'k1': float(left_cam.disto[0]),
            'k2': float(left_cam.disto[1]),
            'p1': float(left_cam.disto[2]),
            'p2': float(left_cam.disto[3]),
            'k3': float(left_cam.disto[4]),
            'resolution': {
                'width': int(self.camera_info.camera_configuration.resolution.width),
                'height': int(self.camera_info.camera_configuration.resolution.height)
            },
            'baseline': float(calib_params.get_camera_baseline()),
            'serial_number': int(self.camera_info.serial_number),
            'camera_model': str(self.camera_info.camera_model).replace('CAMERA_MODEL.', ''),
            'firmware_version': getattr(self.camera_info, 'camera_firmware_version', 0)
        }


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    record_path: str | None = None  # Path where to save the recording (used when save_to_disk=True)
    # ZED camera options
    use_zed_camera: bool = False
    zed_camera_id: int = 0
    zed_resolution: str = "VGA"  # VGA, HD720, HD1080, HD2K (will be processed to 128x128)
    zed_fps: int = 30
    # URDF path for forward kinematics
    urdf_path: str = "/home/baxter/Documents/LeTrack/ros_ws/src/so_100_track/urdf/so_100_arm_wheel.urdf"
    # Test mode - skip robot/teleoperator connection (useful for camera-only testing)
    test_mode: bool = False
    # ROS2 TF tree publisher options
    publish_tf: bool = True  # Enable ROS2 tf tree publishing
    tf_robot_name: str = "robot"  # Robot name for ROS2 node and topics
    tf_base_frame: str = "base_link"  # Base frame for TF tree
    tf_ee_frame: str = "gripper_ee"  # End-effector frame name
    # Save to disk option
    save_to_disk: bool = False  # Enable saving observations to disk


def save_observation_to_folder(observation: dict, record_path: str, timestep: int, zed_data: dict | None = None, camera_params: dict | None = None, urdf_path: str | None = None):
    """
    Save observation data to a folder structure.
    
    Args:
        observation: Dictionary containing robot observation data
        record_path: Base path where to save the recordings
        timestep: Current timestep number
        zed_data: Dictionary containing ZED camera data (rgb, depth, pointcloud)
        camera_params: Camera intrinsic parameters (saved only once)
    """
    # Create timestep folder
    timestep_folder = os.path.join(record_path, str(timestep))
    robot_folder = os.path.join(timestep_folder, "robot")
    os.makedirs(robot_folder, exist_ok=True)
    
    # Extract joint positions and build joint dict for FK
    joint_positions = []
    gripper_positions = []
    joint_dict = {}  # For forward kinematics calculation
    
    for key, value in observation.items():
        if key.endswith('.pos'):
            if 'gripper' in key.lower():
                gripper_positions.append(value)
            else:
                joint_positions.append(value)
                # Store in dict for FK calculation with URDF joint names
                lerobot_joint_name = key.replace('.pos', '')
                # Map lerobot joint name to URDF joint name
                urdf_joint_name = LEROBOT_TO_URDF_JOINT_NAMES.get(lerobot_joint_name, lerobot_joint_name)
                # IMPORTANT: Convert degrees to radians for FK computation
                joint_dict[urdf_joint_name] = np.deg2rad(value)
    
    # Compute end effector pose using the shared function (includes gripper_ee_joint)
    ee_pose = None
    ee_rotation = None
    try:
        if urdf_path is not None:
            ee_data = compute_gripper_ee_pose(joint_dict, urdf_path)
            if ee_data:
                ee_pose = ee_data['position']
                ee_rotation = ee_data['quaternion']
    except Exception as e:
        print(f"Warning: Failed to compute end effector pose: {e}")
    
    # Save joint positions (excluding gripper) - convert degrees to radians
    if joint_positions:
        joint_positions_array = np.array(joint_positions, dtype=np.float32)
        # Convert from degrees to radians
        joint_positions_radians = np.deg2rad(joint_positions_array)
        joint_positions_path = os.path.join(robot_folder, "joint_positions.npy")
        np.save(joint_positions_path, joint_positions_radians)
    
    # Save gripper positions separately
    if gripper_positions:
        gripper_positions_array = np.array(gripper_positions, dtype=np.float32)
        gripper_positions_radians = np.deg2rad(gripper_positions_array)
        gripper_positions_path = os.path.join(robot_folder, "gripper.npy")
        np.save(gripper_positions_path, gripper_positions_radians)
    
    # Save end effector pose if available
    if ee_pose is not None:
        ee_pose_path = os.path.join(robot_folder, "ee_pose.npy")
        np.save(ee_pose_path, ee_pose)
    
    # Save end effector rotation if available
    if ee_rotation is not None:
        ee_rotation_path = os.path.join(robot_folder, "ee_rotation.npy")
        np.save(ee_rotation_path, ee_rotation)

    # Save ZED camera data if provided
    if zed_data is not None:
        zed_folder = os.path.join(timestep_folder, "zed")
        os.makedirs(zed_folder, exist_ok=True)
        
        # Save camera calibration parameters in each timestep's camera folder
        if camera_params is not None:
            camera_calib_path = os.path.join(zed_folder, "camera_calibration.json")
            with open(camera_calib_path, 'w') as f:
                json.dump(camera_params, f, indent=4)
        
        # Save RGB image as numpy array
        if 'rgb' in zed_data and zed_data['rgb'] is not None:
            rgb_path = os.path.join(zed_folder, "rgb.npy")
            np.save(rgb_path, zed_data['rgb'])
            
            # Save RGB image as PNG
            rgb_img = Image.fromarray(zed_data['rgb'].astype(np.uint8))
            rgb_png_path = os.path.join(zed_folder, "rgb.png")
            rgb_img.save(rgb_png_path)
        
        # Save original RGB image (non-cropped)
        if 'rgb_original' in zed_data and zed_data['rgb_original'] is not None:
            rgb_original_path = os.path.join(zed_folder, "rgb_original.npy")
            np.save(rgb_original_path, zed_data['rgb_original'])
            
            # Save original RGB image as PNG
            rgb_original_img = Image.fromarray(zed_data['rgb_original'].astype(np.uint8))
            rgb_original_png_path = os.path.join(zed_folder, "rgb_original.png")
            rgb_original_img.save(rgb_original_png_path)
        
        # Save depth image as numpy array
        if 'depth' in zed_data and zed_data['depth'] is not None:
            depth_path = os.path.join(zed_folder, "depth.npy")
            np.save(depth_path, zed_data['depth'])
        
        # Save original depth image (non-cropped)
        if 'depth_original' in zed_data and zed_data['depth_original'] is not None:
            depth_original_path = os.path.join(zed_folder, "depth_original.npy")
            np.save(depth_original_path, zed_data['depth_original'])
        
        # Save point cloud as numpy array
        if 'pointcloud' in zed_data and zed_data['pointcloud'] is not None:
            pointcloud_path = os.path.join(zed_folder, "pointcloud.npy")
            pointcloud_color_path = os.path.join(zed_folder, "pointcloud_color.npy")
            np.save(pointcloud_path, zed_data['pointcloud'])
            np.save(pointcloud_color_path, zed_data['pointcloud_color'])

    # Save camera intrinsic parameters (only once, at timestep 0)
    if camera_params is not None and timestep == 0:
        camera_params_path = os.path.join(record_path, "zed_camera_parameters.json")
        with open(camera_params_path, 'w') as f:
            json.dump(camera_params, f, indent=4)


def teleop_loop(
    teleop: Teleoperator | None, robot: Robot | None, fps: int, display_data: bool = False, duration: float | None = None, record_path: str | None = None, zed_camera: ZEDCamera | None = None, test_mode: bool = False, urdf_path: str | None = None, tf_publisher: "ROS2TFTreePublisher | None" = None, save_to_disk: bool = False
):
    if not test_mode and (teleop is None or robot is None):
        raise RuntimeError("Robot and teleoperator must be provided when not in test mode")
    
    if not test_mode and robot is not None:
        display_len = max(len(key) for key in robot.action_features)
    else:
        display_len = 10  # Default for test mode
        
    start = time.perf_counter()
    timestep = 0
    camera_params_saved = False
    last_save_time = time.perf_counter()  # Track time of last frame save
    frame_interval = 1.0 / fps  # Calculate target interval between frames
    
    while True:
        # Get action and observation first (at full robot speed)
        action = None
        observation = {}
        
        if not test_mode and teleop is not None and robot is not None:
            action = teleop.get_action()
            observation = robot.get_observation()
        else:
            # In test mode, create dummy data
            action = {"dummy_joint.pos": 0.0}
            observation = {"dummy_joint.pos": 0.0, "gripper.pos": 0.5}
        
        # Send action to robot IMMEDIATELY (before any throttling)
        if not test_mode and robot is not None and action is not None:
            robot.send_action(action)
        
        # FPS throttling for RECORDING only (not for robot control)
        loop_start = time.perf_counter()
        time_since_last_save = loop_start - last_save_time
        should_save = time_since_last_save >= frame_interval
        
        # If not enough time has passed for saving, wait and continue
        if not should_save:
            wait_time = min(0.01, frame_interval - time_since_last_save)  # Wait up to 10ms or remaining time
            time.sleep(wait_time)
            continue  # Skip to next iteration
        
        # Capture ZED camera data (always, at its own rate)
        zed_data = None
        camera_params = None
        if zed_camera is not None:
            try:
                zed_data = zed_camera.capture()
                if zed_data is None:
                    print("Warning: ZED camera capture returned None - check camera connection")
                else:
                    # Always get camera params (needed for saving and publishing)
                    camera_params = zed_camera.get_camera_parameters()
                    
                    # Publish calibration parameters every time (so late subscribers can receive it)
                    if tf_publisher is not None:
                        tf_publisher.publish_camera_calibration(camera_params)
                        if not camera_params_saved:
                            print("Publishing camera calibration parameters to /dataset/camera_calibration")
                            camera_params_saved = True
            except Exception as e:
                print(f"Warning: Failed to capture ZED camera data: {e}")
                import traceback
                traceback.print_exc()
        
        # Publish to ROS2 TF tree
        if tf_publisher is not None and observation:
            try:
                # Extract joint positions from observation
                joint_positions_dict = {}
                for key, value in observation.items():
                    if key.endswith('.pos'):
                        # Remove '.pos' suffix to get joint name
                        joint_name = key.replace('.pos', '')
                        # Convert degrees to radians
                        joint_positions_dict[joint_name] = np.deg2rad(float(value))
                
                # Update TF publisher with current timestamp
                current_time = time.time()
                tf_publisher.update_joint_state(joint_positions_dict, current_time)
                
                # Publish camera data if available
                if zed_data is not None:
                    tf_publisher.publish_camera_data(zed_data, current_time)
            except Exception as e:
                print(f"Warning: Failed to publish TF data: {e}")
        
        # Only save observation when save_to_disk is enabled and recording is throttled appropriately
        if save_to_disk and record_path is not None:
            save_observation_to_folder(observation, record_path, timestep, zed_data, camera_params, urdf_path)
            timestep += 1
            last_save_time = loop_start
            
        # Calculate actual loop time (for display purposes only)
        loop_s = time.perf_counter() - loop_start

        # Display status
        if test_mode:
            print("\n" + "-" * 30)
            print("TEST MODE - Camera Only")
            if zed_data is not None:
                print(f"ZED: RGB {zed_data['rgb'].shape}, Depth {zed_data['depth'].shape}, PC {zed_data['pointcloud'].shape}")
            else:
                print("ZED: No data captured")
        else:
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            if action is not None:
                for motor, value in action.items():
                    print(f"{motor:<{display_len}} | {value:>7.2f}")
                
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        print(f"Timestep: {timestep} (FPS throttling: 1/{frame_interval:.2f}s = {1/frame_interval:.1f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return

        if not test_mode and action is not None:
            move_cursor_up(len(action) + 4)
        else:
            move_cursor_up(5)

@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    # Validate that at least one of save_to_disk or publish_tf is enabled
    if not cfg.save_to_disk and not cfg.publish_tf:
        raise ValueError("At least one of 'save_to_disk' or 'publish_tf' must be enabled")
    
    # Validate that record_path is provided when save_to_disk is enabled
    if cfg.save_to_disk and cfg.record_path is None:
        raise ValueError("'record_path' must be provided when 'save_to_disk' is enabled")
    
    if cfg.display_data:
        init_rerun(session_name="teleoperation")

    teleop = None
    robot = None
    tf_publisher = None
    
    # Initialize ROS2 TF tree publisher if requested
    if cfg.publish_tf and ROS2_AVAILABLE:
        try:
            tf_publisher = ROS2TFTreePublisher(
                robot_name=cfg.tf_robot_name,
                base_frame=cfg.tf_base_frame,
                ee_frame=cfg.tf_ee_frame,
                urdf_path=cfg.urdf_path
            )
            logging.info("ROS2 TF tree publisher initialized successfully")
        except Exception as e:
            logging.warning(f"Failed to initialize ROS2 TF tree publisher: {e}")
            tf_publisher = None
    elif cfg.publish_tf and not ROS2_AVAILABLE:
        logging.warning("ROS2 TF tree publishing requested but ROS2 is not available")
    
    # Initialize robot and teleoperator only if not in test mode
    if not cfg.test_mode:
        try:
            teleop = make_teleoperator_from_config(cfg.teleop)
            robot = make_robot_from_config(cfg.robot)
            print("Robot and teleoperator configurations loaded successfully")
        except Exception as e:
            print(f"Failed to create robot/teleoperator configurations: {e}")
            if cfg.use_zed_camera:
                print("Continuing in test mode with ZED camera only...")
                cfg.test_mode = True
            else:
                raise
    else:
        print("Running in test mode - robot and teleoperator will be skipped")
    
    # Initialize ZED camera if requested
    zed_camera = None
    if cfg.use_zed_camera and ZED_AVAILABLE:
        try:
            zed_camera = ZEDCamera(
                camera_id=cfg.zed_camera_id,
                resolution=cfg.zed_resolution,
                fps=cfg.zed_fps
            )
            zed_camera.connect()
            print("ZED camera initialized successfully")
        except Exception as e:
            print(f"Failed to initialize ZED camera: {e}")
            zed_camera = None
    elif cfg.use_zed_camera and not ZED_AVAILABLE:
        print("Warning: ZED camera requested but pyzed library is not available")

    # Connect to robot and teleoperator only if not in test mode
    if not cfg.test_mode and teleop is not None and robot is not None:
        try:
            teleop.connect()
            robot.connect()
            print("Robot and teleoperator connected successfully")
        except Exception as e:
            print(f"Failed to connect robot/teleoperator: {e}")
            if cfg.use_zed_camera and zed_camera is not None:
                print("Continuing in test mode with ZED camera only...")
                cfg.test_mode = True
                # Disconnect any partially connected devices
                try:
                    if teleop is not None:
                        teleop.disconnect()
                except:
                    pass
                try:
                    if robot is not None:
                        robot.disconnect()
                except:
                    pass
            else:
                raise

    try:
        teleop_loop(
            teleop, robot, cfg.fps, 
            display_data=cfg.display_data, 
            duration=cfg.teleop_time_s, 
            record_path=cfg.record_path,
            zed_camera=zed_camera,
            test_mode=cfg.test_mode,
            urdf_path=cfg.urdf_path,
            tf_publisher=tf_publisher,
            save_to_disk=cfg.save_to_disk
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        if zed_camera is not None:
            zed_camera.disconnect()
        if tf_publisher is not None:
            tf_publisher.shutdown()
        if not cfg.test_mode:
            if teleop is not None:
                try:
                    teleop.disconnect()
                except:
                    pass
            if robot is not None:
                try:
                    robot.disconnect()
                except:
                    pass


def main():
    teleoperate()


if __name__ == "__main__":
    main()
