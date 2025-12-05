#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Execute a policy on a robot using ROS2 topics for observation and action.

This script subscribes to ROS2 topics for camera feeds and robot state,
runs policy inference locally, and publishes actions via ROS2 topics.

Example command:
```shell
python src/lerobot/scripts/lerobot_policy_execute.py \
    --policy.path="/path/to/pretrained_model" \
    --policy.device=cuda \
    --task="Pick up the cube" \
    --dataset_repo_id="user/dataset" \
    --dataset_root="/path/to/dataset" \
    --camera_topics="['/so101track_cube/camera/rgb/image_raw', '/so101track_cube/wrist_camera/rgb/image_raw']" \
    --camera_keys="['scene_camera', 'wrist_camera']" \
    --robot_state_topic="/so101track_cube/joint_states" \
    --robot_pose_topic="/so101track_cube/right_arm/end_effector/pose" \
    --ee_pose_topic="/right_hand/pose" \
    --gripper_topic="/right_hand/trigger" \
    --control_frequency=2.0 \
    --fps=30

Note: dataset_repo_id and dataset_root are optional but recommended.
      They provide normalization stats for the policy.
```
"""

import logging
import time
import threading
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List

import cv2
import draccus
import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device, init_logging


@dataclass
class PolicyExecuteConfig:
    """Configuration for policy execution with ROS2."""

    # Policy configuration (will be loaded from pretrained path in __post_init__)
    policy: PreTrainedConfig | None = field(default=None, metadata={"help": "Policy configuration"})

    # Task instruction for the robot to execute
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})

    # Dataset path for loading metadata (optional - used for normalization stats)
    dataset_repo_id: str | None = field(
        default=None, metadata={"help": "Dataset repository ID for loading metadata (e.g., 'user/dataset')"}
    )
    dataset_root: str | Path | None = field(
        default=None, metadata={"help": "Root directory where the dataset is stored"}
    )

    # Control behavior configuration
    fps: int = field(default=30, metadata={"help": "Observation streaming rate in Hz"})
    control_frequency: float = field(default=2.0, metadata={"help": "Action execution frequency in Hz"})

    # ROS2 topic configuration
    camera_topics: List[str] = field(
        default_factory=lambda: [
            "/so101track_cube/camera/rgb/image_raw",
            "/so101track_cube/wrist_camera/rgb/image_raw",
        ],
        metadata={"help": "List of camera topics to subscribe to"},
    )
    camera_keys: List[str] = field(
        default_factory=lambda: ["scene_camera", "wrist_camera"],
        metadata={"help": "List of camera keys corresponding to camera topics"},
    )
    robot_state_topic: str = field(
        default="/so101track_cube/joint_states", metadata={"help": "Topic for robot joint states"}
    )
    robot_pose_topic: str = field(
        default="/so101track_cube/right_arm/end_effector/pose",
        metadata={"help": "Topic for robot end effector pose"},
    )
    ee_pose_topic: str = field(
        default="/right_hand/pose", metadata={"help": "Topic to publish end effector pose commands"}
    )
    gripper_topic: str = field(
        default="/right_hand/trigger", metadata={"help": "Topic to publish gripper commands"}
    )

    # Debug configuration
    show_images: bool = field(default=False, metadata={"help": "Display camera images for debugging"})

    # Convergence settings
    wait_for_convergence: bool = field(
        default=True, metadata={"help": "Wait for robot to reach target before next action"}
    )
    convergence_threshold: float = field(
        default=0.01, metadata={"help": "Distance threshold for convergence (meters)"}
    )

    # Rename map for observation keys
    rename_map: dict[str, str] = field(default_factory=dict)

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    @property
    def control_dt(self) -> float:
        """Control loop time step, in seconds"""
        return 1 / self.control_frequency

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.control_frequency <= 0:
            raise ValueError(f"control_frequency must be positive, got {self.control_frequency}")

        if len(self.camera_topics) != len(self.camera_keys):
            raise ValueError(
                f"camera_topics ({len(self.camera_topics)}) and "
                f"camera_keys ({len(self.camera_keys)}) must have same length"
            )

        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            from pathlib import Path
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)
        
        if self.policy is None:
            raise ValueError("Policy configuration is required. Use --policy.path to specify a pretrained model path.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


class PolicyExecutor(Node):
    """ROS2-based policy executor.

    This node:
    1. Subscribes to camera and robot state ROS2 topics
    2. Runs policy inference locally
    3. Publishes end effector poses and gripper commands via ROS2
    """

    def __init__(self, config: PolicyExecuteConfig):
        super().__init__("policy_executor")

        self.config = config
        self.logger = logging.getLogger("policy_executor")

        # Data storage
        self.camera_images = {}
        self.latest_joint_states = None
        self.latest_robot_pose = None

        # Control synchronization
        self.shutdown_event = threading.Event()
        self.last_published_target = None
        self.last_pose_update_time = None
        self.total_distance_moved = 0.0

        # Policy and processors
        self.policy: PreTrainedPolicy | None = None
        self.preprocessor: PolicyProcessorPipeline | None = None
        self.postprocessor: PolicyProcessorPipeline | None = None

        # Setup ROS2 subscribers and publishers
        self._setup_ros2(QoSProfile(depth=10))

        # Timer for control loop
        self.control_timer = self.create_timer(self.config.control_dt, self.control_loop_callback)

        self.logger.info("Policy Executor initialized and ready")

    def _setup_ros2(self, qos_profile):
        """Setup ROS2 subscribers and publishers."""
        # Camera subscribers
        for i, topic in enumerate(self.config.camera_topics):
            camera_key = self.config.camera_keys[i]
            # Use a factory function to create properly typed callbacks
            def make_camera_callback(key: str):
                def callback(msg):
                    self.camera_callback(msg, key)
                return callback
            
            self.create_subscription(
                Image, topic, make_camera_callback(camera_key), qos_profile
            )
            self.logger.info(f"Subscribed to camera topic: {topic} -> {camera_key}")

        # Robot state subscriber
        self.create_subscription(JointState, self.config.robot_state_topic, self.joint_state_callback, qos_profile)
        self.logger.info(f"Subscribed to joint state topic: {self.config.robot_state_topic}")

        # Robot pose subscriber
        self.create_subscription(Pose, self.config.robot_pose_topic, self.robot_pose_callback, qos_profile)
        self.logger.info(f"Subscribed to robot pose topic: {self.config.robot_pose_topic}")

        # Publishers
        self.ee_pose_pub = self.create_publisher(Pose, self.config.ee_pose_topic, qos_profile)
        self.logger.info(f"Publishing end effector poses to: {self.config.ee_pose_topic}")

        self.gripper_pub = self.create_publisher(Bool, self.config.gripper_topic, qos_profile)
        self.logger.info(f"Publishing gripper commands to: {self.config.gripper_topic}")

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def camera_callback(self, msg: Image, camera_key: str):
        """Callback for camera image messages."""
        try:
            # Convert ROS Image to numpy array
            if msg.encoding == "rgb8":
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                cv_image = img_array.reshape((msg.height, msg.width, 3))
            elif msg.encoding == "bgr8":
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                bgr_image = img_array.reshape((msg.height, msg.width, 3))
                cv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            elif msg.encoding == "mono8":
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                mono_image = img_array.reshape((msg.height, msg.width))
                cv_image = cv2.cvtColor(mono_image, cv2.COLOR_GRAY2RGB)
            else:
                self.logger.error(f"Unsupported image encoding: {msg.encoding}")
                return

            self.camera_images[camera_key] = cv_image
            self.logger.debug(
                f"[CAMERA UPDATE] {camera_key}: shape={cv_image.shape}, mean={cv_image.mean():.2f}"
            )

            if self.config.show_images:
                cv2.imshow(camera_key, cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

        except Exception as e:
            self.logger.error(f"Error converting camera image: {e}")

    def robot_pose_callback(self, msg: Pose):
        """Callback for robot pose messages."""
        try:
            pos = msg.position
            orient = msg.orientation

            new_pose = np.array([pos.x, pos.y, pos.z, orient.w, orient.x, orient.y, orient.z])

            # Track movement and log periodically
            current_time = time.time()
            if self.latest_robot_pose is not None:
                position_delta = np.linalg.norm(new_pose[:3] - self.latest_robot_pose[:3])
                self.total_distance_moved += position_delta

                # Log every 5 seconds
                if self.last_pose_update_time is None or (current_time - self.last_pose_update_time) >= 5.0:
                    self.logger.info(
                        f"[ROBOT POSE] Position: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}] | "
                        f"Orientation: [{orient.x:.3f}, {orient.y:.3f}, {orient.z:.3f}, {orient.w:.3f}] | "
                        f"Total distance moved: {self.total_distance_moved:.3f}m"
                    )
                    self.last_pose_update_time = current_time

            self.latest_robot_pose = new_pose

        except Exception as e:
            self.logger.error(f"Error processing robot pose: {e}")

    def joint_state_callback(self, msg: JointState):
        """Callback for joint state messages."""
        try:
            new_joints = np.array(msg.position, dtype=np.float64)

            if self.latest_joint_states is not None:
                joint_delta = np.linalg.norm(new_joints - self.latest_joint_states)
                self.logger.debug(f"[JOINT STATE] Delta: {joint_delta:.4f}")

            self.latest_joint_states = new_joints

        except Exception as e:
            self.logger.error(f"Error processing joint states: {e}")

    def initialize_policy(self):
        """Initialize the policy and preprocessors."""
        try:
            # Ensure policy config is loaded
            assert self.config.policy is not None, "Policy configuration not loaded"
            
            self.logger.info(f"Loading policy from {self.config.policy.pretrained_path}")

            # Wait for camera images to determine shape
            max_wait = 5.0
            wait_start = time.time()
            while not self.camera_images and (time.time() - wait_start) < max_wait:
                rclpy.spin_once(self, timeout_sec=0.1)

            if not self.camera_images:
                raise RuntimeError("No camera images received within timeout")

            # Get camera dimensions from first available camera
            first_cam_key = list(self.camera_images.keys())[0]
            img = self.camera_images[first_cam_key]
            camera_shape = img.shape  # (height, width, channels)
            self.logger.info(f"Using camera shape: {camera_shape} from {first_cam_key}")

            # Load dataset metadata if provided (for normalization stats)
            ds_meta = None
            if self.config.dataset_repo_id or self.config.dataset_root:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
                
                if self.config.dataset_repo_id:
                    self.logger.info(f"Loading dataset metadata from {self.config.dataset_repo_id}")
                    dataset = LeRobotDataset(
                        self.config.dataset_repo_id,
                        root=self.config.dataset_root,
                    )
                    ds_meta = dataset.meta
                    self.logger.info(f"Loaded dataset metadata: fps={ds_meta.fps}, robot_type={ds_meta.robot_type}")
                else:
                    self.logger.warning("dataset_root provided without dataset_repo_id, cannot load metadata")

            # Load policy
            self.policy = make_policy(self.config.policy, ds_meta=ds_meta)
            self.policy.reset()

            # Create preprocessor and postprocessor
            pretrained_path_str = str(self.config.policy.pretrained_path) if self.config.policy.pretrained_path else None
            device_str = self.config.policy.device if self.config.policy.device else "cpu"
            
            # Get dataset stats if available
            dataset_stats = ds_meta.stats if ds_meta else None
            
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=self.config.policy,
                pretrained_path=pretrained_path_str,
                dataset_stats=dataset_stats,
                preprocessor_overrides={
                    "device_processor": {"device": device_str},
                    "rename_observations_processor": {"rename_map": self.config.rename_map},
                },
            )

            self.logger.info("Policy initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize policy: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def _data_ready(self) -> bool:
        """Check if we have all necessary data."""
        cameras_ready = len(self.camera_images) >= len(self.config.camera_topics)
        pose_ready = self.latest_robot_pose is not None

        return cameras_ready and pose_ready

    def control_loop_callback(self):
        """Main control loop callback - executed at control_frequency Hz."""
        try:
            # Check if policy is initialized
            if self.policy is None:
                return
            
            # Ensure policy config exists
            assert self.config.policy is not None, "Policy configuration not available"

            # Check if we have sufficient data
            if not self._data_ready():
                self.logger.debug("Waiting for data...")
                return

            # Check convergence if needed
            if self.config.wait_for_convergence and self.last_published_target is not None and self.latest_robot_pose is not None:
                current_position = self.latest_robot_pose[:3]
                distance = np.linalg.norm(current_position - self.last_published_target)
                if distance > self.config.convergence_threshold:
                    self.logger.debug(
                        f"Waiting for convergence: distance={distance:.4f}m, "
                        f"threshold={self.config.convergence_threshold:.4f}m"
                    )
                    return

            # Build observation frame
            observation_frame = self._build_observation()

            # Run policy inference
            device_str = self.config.policy.device if self.config.policy.device else "cpu"
            
            # Type assertions for safety
            assert self.preprocessor is not None, "Preprocessor not initialized"
            assert self.postprocessor is not None, "Postprocessor not initialized"
            
            action_values = predict_action(
                observation=observation_frame,
                policy=self.policy,
                device=get_safe_torch_device(device_str),
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                use_amp=self.config.policy.use_amp,
                task=self.config.task,
                robot_type="so101",  # TODO: Make this configurable
            )

            # Parse and publish action
            action_data = self._parse_action(action_values)
            self._publish_action(action_data)

            self.logger.info("Published action from policy")

        except Exception as e:
            self.logger.error(f"Error in control loop: {e}")
            self.logger.error(traceback.format_exc())

    def _build_observation(self) -> Dict[str, Any]:
        """Build observation dictionary from current robot state."""
        observation = {}

        # Add camera images as numpy arrays (will be converted to tensors by the policy)
        for key in self.config.camera_keys:
            if key in self.camera_images:
                img = self.camera_images[key]
                # Keep as numpy array with shape (H, W, C) and values in [0, 255]
                observation[f"observation.images.{key}"] = img

        # Add robot state as a single concatenated array: [ee_pos_x, ee_pos_y, ee_pos_z, ee_rot_qx, ee_rot_qy, ee_rot_qz, ee_rot_qw, gripper]
        state_values = []
        
        if self.latest_robot_pose is not None:
            pose = self.latest_robot_pose
            # Add position (x, y, z)
            state_values.extend([pose[0], pose[1], pose[2]])
            # Add rotation quaternion (qw, qx, qy, qz)
            state_values.extend([pose[3], pose[4], pose[5], pose[6]])

        # Add gripper state (from joint states if available, otherwise default to 0)
        if self.latest_joint_states is not None and len(self.latest_joint_states) > 0:
            # Assuming last joint is gripper
            state_values.append(self.latest_joint_states[-1])
        else:
            state_values.append(0.0)

        # Concatenate all state values into a single numpy array
        observation["observation.state"] = np.array(state_values, dtype=np.float32)

        return observation

    def _parse_action(self, action_values: Any) -> Dict[str, Any]:
        """Parse action values into pose, rotation, and gripper components."""
        # Handle case where action_values is a tensor directly
        if isinstance(action_values, torch.Tensor):
            # Squeeze and flatten to get a 1D array
            action_np = action_values.squeeze().cpu().numpy()
        elif isinstance(action_values, dict):
            # Extract action tensor from the action dictionary
            # The key format is typically "action.{feature_name}"
            action_keys = [k for k in action_values.keys() if k.startswith("action.")]
            
            if not action_keys:
                raise ValueError(f"No action keys found in action_values: {list(action_values.keys())}")

            # Concatenate all action components
            action_tensors = [action_values[k] for k in sorted(action_keys)]
            action_tensor = torch.cat([t.flatten() for t in action_tensors])
            action_np = action_tensor.squeeze().cpu().numpy()
        else:
            # Already numpy array
            action_np = np.array(action_values).flatten()

        # Ensure it's a 1D array
        if action_np.ndim > 1:
            action_np = action_np.flatten()

        # Expected action format: [x, y, z, qw, qx, qy, qz, gripper]
        if len(action_np) < 8:
            raise ValueError(
                f"Expected at least 8 action dimensions, got {len(action_np)}. "
                f"Action shape: {action_values.shape if isinstance(action_values, torch.Tensor) else 'N/A'}"
            )

        return {
            "pose": action_np[:3],  # [x, y, z]
            "rotation": action_np[3:7],  # [qw, qx, qy, qz]
            "gripper": float(action_np[7]),
        }

    def _publish_action(self, action_data: Dict[str, Any]):
        """Publish end effector pose and gripper commands."""
        try:
            # Store target for convergence tracking
            self.last_published_target = action_data["pose"].copy()

            # Publish pose
            pose_msg = Pose()
            pose_msg.position = Point(
                x=float(action_data["pose"][0]), y=float(action_data["pose"][1]), z=float(action_data["pose"][2])
            )
            pose_msg.orientation = Quaternion(
                w=float(action_data["rotation"][0]),
                x=float(action_data["rotation"][1]),
                y=float(action_data["rotation"][2]),
                z=float(action_data["rotation"][3]),
            )
            self.ee_pose_pub.publish(pose_msg)

            # Log publisher info
            pub_count = self.ee_pose_pub.get_subscription_count()
            self.logger.info(
                f"📤 PUBLISHED to {self.config.ee_pose_topic} (subscribers: {pub_count}) | "
                f"pose=[{action_data['pose'][0]:.3f}, {action_data['pose'][1]:.3f}, {action_data['pose'][2]:.3f}] | "
                f"quat=[{action_data['rotation'][0]:.3f}, {action_data['rotation'][1]:.3f}, "
                f"{action_data['rotation'][2]:.3f}, {action_data['rotation'][3]:.3f}]"
            )

            # Publish gripper
            gripper_msg = Bool()
            gripper_msg.data = bool(action_data["gripper"] > 0.5)
            self.gripper_pub.publish(gripper_msg)

            self.logger.info(f"📤 PUBLISHED gripper to {self.config.gripper_topic}: {gripper_msg.data}")

        except Exception as e:
            self.logger.error(f"Error publishing action: {e}")

    def stop(self):
        """Stop the executor."""
        self.shutdown_event.set()
        self.logger.info("Executor stopped")


@parser.wrap()
def policy_execute(cfg: PolicyExecuteConfig):
    """Main entry point for policy executor."""
    init_logging()
    logging.info("=" * 80)
    logging.info("Policy Executor Configuration:")
    logging.info("=" * 80)
    logging.info(pformat(asdict(cfg)))
    logging.info("=" * 80)

    # Initialize ROS2
    rclpy.init()

    executor = None
    try:
        # Create executor node
        executor = PolicyExecutor(cfg)

        # Initialize policy
        if not executor.initialize_policy():
            logging.error("Failed to initialize policy")
            return

        # Spin ROS2 node
        executor.logger.info("Starting ROS2 spin loop...")
        rclpy.spin(executor)

    except KeyboardInterrupt:
        logging.info("Shutdown requested by user")
    except Exception as e:
        logging.error(f"Error: {e}")
        logging.error(traceback.format_exc())
    finally:
        # Cleanup
        if executor is not None:
            executor.stop()
            executor.destroy_node()

        rclpy.shutdown()
        logging.info("Policy Executor stopped")


if __name__ == "__main__":
    policy_execute()  # draccus.wrap() handles the config argument
