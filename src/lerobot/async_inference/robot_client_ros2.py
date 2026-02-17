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
ROS2-compatible robot client for async inference with policy server.

This client subscribes to ROS2 topics for camera feeds and robot state,
sends observations to a remote policy server via gRPC, receives action chunks,
and publishes end effector poses and gripper commands via ROS2 topics.

Example command:
```shell
python src/lerobot/async_inference/robot_client_ros2.py \
    --task="Pick up the cube" \
    --server_address=127.0.0.1:8080 \
    --policy_type=groot \
    --pretrained_name_or_path=user/model \
    --policy_device=cuda \
    --actions_per_chunk=8 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --camera_topics="['/so101track_cube/camera/rgb/image_raw', '/so101track_cube/wrist_camera/rgb/image_raw']" \
    --camera_keys="['scene_camera', 'wrist_camera']" \
    --robot_state_topic="/so101track_cube/joint_states" \
    --robot_pose_topic="/so101track_cube/right_arm/end_effector/pose" \
    --ee_pose_topic="/right_hand/pose" \
    --gripper_topic="/right_hand/trigger" \
    --control_frequency=2.0 \
    --fps=30 \
    --debug_visualize_queue_size=False
```
"""

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pprint import pformat
from queue import Queue
from typing import Any, Dict, List

import draccus
import grpc
import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool

from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.async_inference.configs import AGGREGATE_FUNCTIONS, get_aggregate_function
from lerobot.async_inference.constants import DEFAULT_FPS
from lerobot.async_inference.helpers import (
    Action,
    FPSTracker,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    visualize_action_queue_size,
)


@dataclass
class RobotClientROS2Config:
    """Configuration for ROS2-based RobotClient.

    This class defines all configurable parameters for the ROS2 RobotClient,
    including network connection, policy settings, ROS2 topics, and control behavior.
    """

    # Policy configuration
    policy_type: str = field(metadata={"help": "Type of policy to use (e.g., 'groot', 'act', 'pi05')"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})

    # Task instruction for the robot to execute
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})

    # Policies typically output K actions at max, but we can use less to avoid wasting bandwidth
    actions_per_chunk: int = field(default=8, metadata={"help": "Number of actions per chunk"})

    # Action history configuration for temporal conditioning
    n_action_steps: int = field(
        default=1, 
        metadata={"help": "Number of previous actions to include in observations (1 = no history, >1 = action history)"}
    )

    # Control behavior configuration
    chunk_size_threshold: float = field(
        default=0, metadata={"help": "Threshold for chunk size control (0-1)"}
    )
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Observation streaming rate in Hz"})
    control_frequency: float = field(default=2.0, metadata={"help": "Action execution frequency in Hz"})
    grpc_timeout: float = field(
        default=30.0, metadata={"help": "gRPC timeout in seconds for receiving actions from server"}
    )

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
    robot_base_pose_topic: str = field(
        default="/robot_base/pose",
        metadata={"help": "Topic for robot base pose (only used for ACT policy)"},
    )

    # URDF configuration for forward kinematics (only used for ACT policy)
    urdf_path: str = field(
        default="",
        metadata={"help": "Path to URDF file for forward kinematics (required for ACT policy)"},
    )
    end_effector_link: str = field(
        default="gripper_ee",
        metadata={"help": "Name of end effector link in URDF"},
    )
    base_link: str = field(
        default="track_body",
        metadata={"help": "Name of base link in URDF"},
    )

    # Feature keys for policy (these map to the expected state keys in the policy)
    joint_position_keys: List[str] = field(
        default_factory=lambda: ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        metadata={"help": "Keys for joint positions in the policy"},
    )
    gripper_key: str = field(default="gripper", metadata={"help": "Key for gripper state in the policy"})

    # Aggregate function configuration
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"Name of aggregate function. Options: {list(AGGREGATE_FUNCTIONS.keys())}"},
    )

    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize the action queue size"}
    )
    show_images: bool = field(default=False, metadata={"help": "Display camera images for debugging"})

    # Convergence settings
    wait_for_convergence: bool = field(
        default=True, metadata={"help": "Wait for robot to reach target before next action"}
    )
    convergence_threshold: float = field(
        default=0.01, metadata={"help": "Distance threshold for convergence (meters)"}
    )

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
        if not self.server_address:
            raise ValueError("server_address cannot be empty")

        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")

        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.control_frequency <= 0:
            raise ValueError(f"control_frequency must be positive, got {self.control_frequency}")

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if len(self.camera_topics) != len(self.camera_keys):
            raise ValueError(
                f"camera_topics ({len(self.camera_topics)}) and "
                f"camera_keys ({len(self.camera_keys)}) must have same length"
            )

        # Get aggregate function
        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)


class RobotClientROS2(Node):
    """ROS2-based robot client for async inference.

    This client:
    1. Subscribes to camera and robot state ROS2 topics
    2. Sends observations to a remote policy server via gRPC
    3. Receives action chunks from the policy server
    4. Publishes end effector poses and gripper commands via ROS2
    """

    def __init__(self, config: RobotClientROS2Config):
        super().__init__("robot_client_ros2")

        self.config = config
        self.logger = get_logger("robot_client_ros2")

        # Initialize gRPC connection
        self.server_address = config.server_address
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        # Data storage
        self.camera_images = {}
        self.latest_joint_states = None
        self.latest_robot_pose = None
        self.latest_base_pose = None  # Only used for ACT policy

        # URDF and forward kinematics (only for ACT policy)
        self.robot_urdf = None
        self.kinematic_chain = []
        if self.config.policy_type.lower() == "act":
            if not self.config.urdf_path:
                raise ValueError("urdf_path is required when using ACT policy")
            self.load_urdf()
            self.logger.info(f"Forward kinematics ready with {len(self.kinematic_chain)} joints")

        # Action queue management
        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size = []
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        # Control synchronization
        self.shutdown_event = threading.Event()

        # Tracking variables
        self.last_published_target = None
        self.policy_query_count = 0
        self.last_pose_update_time = None
        self.total_distance_moved = 0.0
        self.convergence_wait_count = 0
        self.max_convergence_wait = 5  # Skip convergence check after this many attempts

        # Action history buffer for temporal conditioning
        self.action_history = []
        self.max_action_history = config.n_action_steps
        
        if self.max_action_history > 1:
            self.logger.info(
                f"🕰️  Action history enabled: keeping last {self.max_action_history} actions for temporal conditioning"
            )
        else:
            self.logger.info("⚠️  Action history disabled (n_action_steps=1) - model has no temporal context")

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        # Setup ROS2 subscribers and publishers
        self._setup_ros2(QoSProfile(depth=10))

        # Timers for control loop
        self.policy_timer = self.create_timer(self.config.control_dt, self.control_loop_callback)

        self.logger.info("ROS2 Robot Client initialized and ready")

    def _setup_ros2(self, qos_profile):
        """Setup ROS2 subscribers and publishers."""
        # Camera subscribers
        for i, topic in enumerate(self.config.camera_topics):
            camera_key = self.config.camera_keys[i]
            self.create_subscription(
                Image, topic, lambda msg, key=camera_key: self.camera_callback(msg, key), qos_profile
            )
            self.logger.info(f"Subscribed to camera topic: {topic} -> {camera_key}")

        # Robot state subscriber
        self.create_subscription(JointState, self.config.robot_state_topic, self.joint_state_callback, qos_profile)
        self.logger.info(f"Subscribed to joint state topic: {self.config.robot_state_topic}")

        # Robot pose subscriber
        self.create_subscription(PoseStamped, self.config.robot_pose_topic, self.robot_pose_callback, qos_profile)
        self.logger.info(f"Subscribed to robot pose topic: {self.config.robot_pose_topic}")

        # Robot base pose subscriber (only for ACT policy)
        if self.config.policy_type.lower() == "act":
            self.create_subscription(
                PoseStamped, self.config.robot_base_pose_topic, self.base_pose_callback, qos_profile
            )
            self.logger.info(f"Subscribed to robot base pose topic: {self.config.robot_base_pose_topic}")

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
                cv_image = bgr_image[:, :, ::-1]  # Convert BGR to RGB
            elif msg.encoding == "mono8":
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                gray_image = img_array.reshape((msg.height, msg.width, 1))
                cv_image = np.repeat(gray_image, 3, axis=2)  # Convert grayscale to RGB
            else:
                self.logger.warn(f"Unsupported image encoding: {msg.encoding}")
                return

            self.camera_images[camera_key] = cv_image
            self.logger.debug(
                f"[CAMERA UPDATE] {camera_key}: shape={cv_image.shape}, mean={cv_image.mean():.2f}"
            )

        except Exception as e:
            self.logger.error(f"Error converting camera image: {e}")

    def robot_pose_callback(self, msg: PoseStamped):
        """Callback for robot pose messages."""
        try:
            pos = msg.pose.position
            orient = msg.pose.orientation

            new_pose = np.array([pos.x, pos.y, pos.z, orient.x, orient.y, orient.z, orient.w])

            # Track movement and log periodically
            current_time = time.time()
            if self.latest_robot_pose is not None:
                pose_delta = np.linalg.norm(new_pose[:3] - self.latest_robot_pose[:3])
                if pose_delta > 0.001:
                    self.total_distance_moved += pose_delta
                    self.logger.debug(f"[POSE UPDATE] Moved {pose_delta:.4f}m, total={self.total_distance_moved:.4f}m")
                
                # Log pose updates periodically (every 2 seconds)
                if not hasattr(self, '_last_pose_log') or (current_time - self._last_pose_log) > 2.0:
                    self.logger.info(
                        f"📍 Robot pose: [{new_pose[0]:.3f}, {new_pose[1]:.3f}, {new_pose[2]:.3f}] | "
                        f"quat: [{new_pose[3]:.3f}, {new_pose[4]:.3f}, {new_pose[5]:.3f}, {new_pose[6]:.3f}]"
                    )
                    self._last_pose_log = current_time

            self.latest_robot_pose = new_pose

        except Exception as e:
            self.logger.error(f"Error processing robot pose: {e}")

    def base_pose_callback(self, msg: PoseStamped):
        """Callback for robot base pose messages (only used for ACT policy)."""
        try:
            pos = msg.pose.position
            orient = msg.pose.orientation

            self.latest_base_pose = np.array([pos.x, pos.y, pos.z, orient.x, orient.y, orient.z, orient.w])
            self.logger.debug(
                f"[BASE POSE UPDATE] [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}] | "
                f"quat: [{orient.x:.3f}, {orient.y:.3f}, {orient.z:.3f}, {orient.w:.3f}]"
            )

        except Exception as e:
            self.logger.error(f"Error processing base pose: {e}")

    def load_urdf(self):
        """Load and parse the URDF file for forward kinematics."""
        try:
            from urdf_parser_py.urdf import URDF
            import os

            if not os.path.exists(self.config.urdf_path):
                raise FileNotFoundError(f"URDF file not found: {self.config.urdf_path}")

            self.robot_urdf = URDF.from_xml_file(self.config.urdf_path)
            self.build_kinematic_chain()
            self.logger.info(f"Successfully loaded URDF from {self.config.urdf_path}")

        except ImportError:
            raise ImportError(
                "urdf_parser_py is required for ACT policy. Install with: pip install urdf-parser-py"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load URDF: {e}")

    def build_kinematic_chain(self):
        """Build the kinematic chain from base to end effector."""
        if not self.robot_urdf:
            return

        # Build a map of link -> joint connections
        link_to_joint = {}
        for joint in self.robot_urdf.joints:
            link_to_joint[joint.child] = joint

        # Walk from end effector back to base link to build the chain
        chain = []
        current = self.config.end_effector_link

        while current != self.config.base_link and current in link_to_joint:
            chain.insert(0, link_to_joint[current])
            current = link_to_joint[current].parent

        if current != self.config.base_link:
            raise RuntimeError(
                f"Could not trace {self.config.end_effector_link} back to {self.config.base_link}, "
                f"stopped at {current}"
            )

        self.kinematic_chain = chain
        self.logger.info(f"Built kinematic chain with {len(self.kinematic_chain)} joints")
        for joint in self.kinematic_chain:
            self.logger.debug(f"  - {joint.name} ({joint.type})")

    def joint_state_callback(self, msg: JointState):
        """Callback for joint state messages."""
        try:
            new_joints = np.array(msg.position, dtype=np.float64)

            if self.latest_joint_states is not None:
                joint_delta = np.linalg.norm(new_joints - self.latest_joint_states)
                self.logger.debug(f"[JOINT UPDATE] Delta={joint_delta:.4f} rad")

            self.latest_joint_states = new_joints

        except Exception as e:
            self.logger.error(f"Error processing joint states: {e}")

    def start(self):
        """Start the robot client and connect to the policy server."""
        try:
            # Client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # Prepare lerobot features mapping using hw_to_dataset_features format
            # Build hw_features dict: maps feature names to float (for joints) or tuple (for images)
            from lerobot.datasets.utils import hw_to_dataset_features
            from lerobot.utils.constants import OBS_STR
            
            hw_features = {}
            
            # Model expects 8 state dimensions: 3 position (x,y,z) + 4 rotation (qx,qy,qz,qw) + 1 gripper
            # End effector position (x, y, z)
            hw_features["ee_pos_x"] = float
            hw_features["ee_pos_y"] = float
            hw_features["ee_pos_z"] = float
            
            # End effector rotation (quaternion: qx, qy, qz, qw)
            hw_features["ee_rot_qx"] = float
            hw_features["ee_rot_qy"] = float
            hw_features["ee_rot_qz"] = float
            hw_features["ee_rot_qw"] = float
            
            # Gripper state
            hw_features["gripper"] = float
            
            # Add camera features (mapped to shape tuples - height, width, channels)
            # Wait for at least one camera image to determine actual shape
            max_wait = 5.0  # seconds
            wait_start = time.time()
            while not self.camera_images and (time.time() - wait_start) < max_wait:
                self.logger.info("Waiting for camera images to determine shape...")
                rclpy.spin_once(self, timeout_sec=0.1)
            
            if not self.camera_images:
                raise RuntimeError(f"No camera images received after {max_wait}s. Check camera topics!")
            
            # Get actual camera dimensions from first available camera
            first_cam_key = list(self.camera_images.keys())[0]
            img = self.camera_images[first_cam_key]
            camera_shape = img.shape  # (height, width, channels)
            self.logger.info(f"Using camera shape: {camera_shape} from {first_cam_key}")
            
            for camera_key in self.config.camera_keys:
                hw_features[camera_key] = camera_shape
            
            # Convert to LeRobot dataset features format
            lerobot_features = hw_to_dataset_features(hw_features, OBS_STR, use_video=True)
            
            # Debug: Log the lerobot_features structure
            self.logger.info(f"Constructed lerobot_features: {lerobot_features}")

            # Send policy instructions
            policy_config = RemotePolicyConfig(
                self.config.policy_type,
                self.config.pretrained_name_or_path,
                lerobot_features,
                self.config.actions_per_chunk,
                self.config.policy_device,
            )
            
            # Debug: Log policy config details
            self.logger.info(f"Policy config type: {type(policy_config)}")
            self.logger.info(f"Policy config lerobot_features type: {type(policy_config.lerobot_features)}")
            self.logger.info(f"Policy config lerobot_features keys: {list(policy_config.lerobot_features.keys())}")

            policy_config_bytes = pickle.dumps(policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {policy_config.policy_type} | "
                f"Pretrained: {policy_config.pretrained_name_or_path} | "
                f"Device: {policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)
            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client."""
        self.shutdown_event.set()
        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(self, obs: TimedObservation) -> bool:
        """Send observation to the policy server."""
        if not self.running:
            raise RuntimeError("Client not running. Run start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes, services_pb2.Observation, log_prefix="[CLIENT] Observation", silent=True
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep}")
            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _aggregate_action_queues(
        self, incoming_actions: list[TimedAction], aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
    ):
        """Aggregates actions in the queue using the aggregate function."""
        if aggregate_fn is None:
            aggregate_fn = lambda x1, x2: x2  # Default: take latest

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # Skip old actions
            if new_action.get_timestep() <= latest_action:
                continue

            # Add new action directly if not in queue
            if new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # Aggregate if already in queue
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(current_action_queue[new_action.get_timestep()], new_action.get_action()),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self):
        """Receive actions from the policy server."""
        self.logger.info("✅ Action receiving thread started and ready!")

        action_request_count = 0
        while self.running:
            try:
                action_request_count += 1
                self.logger.info(f"🔄 Requesting actions from server (request #{action_request_count})...")
                
                # Use configurable timeout to prevent infinite blocking
                actions_chunk = self.stub.GetActions(services_pb2.Empty(), timeout=self.config.grpc_timeout)
                
                if len(actions_chunk.data) == 0:
                    self.logger.warning(f"⚠️ Received empty action chunk (request #{action_request_count})")
                    time.sleep(0.1)
                    continue

                receive_time = time.time()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                if len(timed_actions) > 0:
                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    incoming_timesteps = [a.get_timestep() for a in timed_actions]
                    # Log first action details for debugging
                    first_action_data = timed_actions[0].get_action()
                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Incoming: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Latency: {server_to_client_latency:.2f}ms | "
                        f"Action shape: {first_action_data.shape if hasattr(first_action_data, 'shape') else 'N/A'}"
                    )

                # Update action queue
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)

            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    self.logger.error(
                        f"⏰ gRPC timeout after {self.config.grpc_timeout}s waiting for actions. "
                        f"The model inference may be taking too long. "
                        f"Consider: (1) Increasing --grpc_timeout, "
                        f"(2) Reducing --num_inference_steps for diffusion models, "
                        f"or (3) Using a faster model."
                    )
                else:
                    self.logger.error(f"❌ gRPC Error receiving actions: {e}")
                    self.logger.error(f"Error details - Code: {e.code()}, Details: {e.details()}")
                time.sleep(0.1)  # Brief pause before retry
            except Exception as e:
                self.logger.error(f"❌ Unexpected error in receive_actions: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                time.sleep(0.1)

    def _data_ready(self) -> bool:
        """Check if we have all necessary data."""
        cameras_ready = len(self.camera_images) >= len(self.config.camera_topics)
        joints_ready = self.latest_joint_states is not None
        pose_ready = self.latest_robot_pose is not None

        return cameras_ready and joints_ready and pose_ready

    def _ready_to_send_observation(self):
        """Check if ready to send observation based on queue size."""
        with self.action_queue_lock:
            if self.action_chunk_size <= 0:
                return True
            return self.action_queue.qsize() / self.action_chunk_size <= self.config.chunk_size_threshold

    def control_loop_callback(self):
        """Main control loop callback - executed at control_frequency Hz."""
        # Log occasionally to confirm loop is running
        if not hasattr(self, '_control_loop_counter'):
            self._control_loop_counter = 0
        self._control_loop_counter += 1
        if self._control_loop_counter % 20 == 1:  # Log every 20 iterations
            self.logger.info(f"🔄 Control loop iteration #{self._control_loop_counter}")
        
        try:
            # Check if we have sufficient data
            if not self._data_ready():
                cameras_ready = len(self.camera_images) >= len(self.config.camera_topics)
                joints_ready = self.latest_joint_states is not None
                pose_ready = self.latest_robot_pose is not None
                self.logger.info(
                    f"⏳ Waiting for data... | Cameras: {len(self.camera_images)}/{len(self.config.camera_topics)} ready={cameras_ready} | "
                    f"Joints: ready={joints_ready} | Pose: ready={pose_ready}"
                )
                return

            # Execute action if available
            with self.action_queue_lock:
                has_actions = not self.action_queue.empty()
                queue_size = self.action_queue.qsize()

            if has_actions:
                self._execute_action()
            else:
                # Log when waiting for actions (every 10 iterations)
                if self._control_loop_counter % 10 == 1:
                    self.logger.info(f"⏳ Waiting for actions... (queue empty, chunk_size={self.action_chunk_size})")

            # Send observation if ready
            if self._ready_to_send_observation():
                self._send_observation()
            else:
                with self.action_queue_lock:
                    queue_size = self.action_queue.qsize()
                self.logger.info(
                    f"⏸️  Not ready to send observation | Queue: {queue_size}/{self.action_chunk_size} | "
                    f"Threshold: {self.config.chunk_size_threshold}"
                )

        except Exception as e:
            self.logger.error(f"Error in control loop: {e}")

    def _execute_action(self):
        """Execute the next action from the queue."""
        try:
            # Check convergence if needed
            if self.config.wait_for_convergence and self.last_published_target is not None:
                if self.latest_robot_pose is None:
                    self.logger.warning("No robot pose available for convergence check")
                    return
                    
                current_error = np.linalg.norm(self.last_published_target - self.latest_robot_pose[:3])
                if current_error > self.config.convergence_threshold:
                    self.convergence_wait_count += 1
                    
                    # Timeout: skip convergence check after max attempts
                    if self.convergence_wait_count >= self.max_convergence_wait:
                        self.logger.warning(
                            f"⚠️ Convergence timeout after {self.convergence_wait_count} attempts "
                            f"(error={current_error*1000:.1f}mm). Proceeding anyway..."
                        )
                        self.convergence_wait_count = 0  # Reset counter
                    else:
                        # Log periodically (every 2 seconds)
                        if not hasattr(self, '_last_convergence_log') or (time.time() - self._last_convergence_log) > 2.0:
                            self.logger.info(
                                f"⏳ Waiting for convergence ({self.convergence_wait_count}/{self.max_convergence_wait}): "
                                f"error={current_error*1000:.1f}mm (threshold={self.config.convergence_threshold*1000:.0f}mm) | "
                                f"Current pose: [{self.latest_robot_pose[0]:.3f}, {self.latest_robot_pose[1]:.3f}, {self.latest_robot_pose[2]:.3f}] | "
                                f"Target: [{self.last_published_target[0]:.3f}, {self.last_published_target[1]:.3f}, {self.last_published_target[2]:.3f}]"
                            )
                            self._last_convergence_log = time.time()
                        return
                else:
                    # Converged - reset counter
                    if self.convergence_wait_count > 0:
                        self.logger.info(f"✅ Converged after {self.convergence_wait_count} checks")
                    self.convergence_wait_count = 0

            # Get action from queue
            with self.action_queue_lock:
                if self.action_queue.empty():
                    return
                timed_action = self.action_queue.get_nowait()
                current_queue_size = self.action_queue.qsize()
                self.action_queue_size.append(current_queue_size)

            # Parse action data
            action_data = self._parse_action(timed_action.get_action())

            # Add to action history buffer (for temporal conditioning)
            self._add_to_action_history(timed_action.get_action())

            # Publish action
            self._publish_action(action_data)

            with self.latest_action_lock:
                self.latest_action = timed_action.get_timestep()

            self.logger.info(
                f"Executed action #{timed_action.get_timestep()} | Queue size: {current_queue_size}"
            )

        except Exception as e:
            self.logger.error(f"Error executing action: {e}")

    def _compute_forward_kinematics(self, joint_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics from joint positions to get end effector pose.
        
        Args:
            joint_positions: Array of joint angles for the kinematic chain
            
        Returns:
            Tuple of (position [x, y, z], rotation quaternion [qx, qy, qz, qw])
        """
        if not self.robot_urdf or not self.kinematic_chain:
            self.logger.error("URDF not loaded or kinematic chain not built")
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])

        import math

        # Create a mapping of joint names to positions
        joint_position_map = {}
        for i, key in enumerate(self.config.joint_position_keys):
            if i < len(joint_positions):
                joint_position_map[key] = joint_positions[i]

        # Apply all transformations from base to end effector
        transform = np.eye(4)

        for joint in self.kinematic_chain:
            origin = joint.origin
            if not origin:
                continue

            dx, dy, dz = origin.xyz if origin.xyz else [0, 0, 0]
            roll, pitch, yaw = origin.rpy if origin.rpy else [0, 0, 0]

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

                axis = np.array(joint.axis, dtype=np.float64) if joint.axis else np.array([0, 0, 1])
                axis = axis / np.linalg.norm(axis)
                joint_angle = joint_position_map.get(joint.name, 0.0)
                R_joint = self._create_rotation_matrix(axis, joint_angle)[:3, :3]
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

        # Extract position and orientation from transformation matrix
        position = transform[:3, 3].astype(np.float32)
        rotation_matrix = transform[:3, :3]
        quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)

        return position, quaternion

    def _create_rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Create a 4x4 rotation matrix around given axis by given angle."""
        import math

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

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [x, y, z, w]."""
        import math

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

        return np.array([x, y, z, w])

    def _parse_action(self, action_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """Parse action tensor into pose, rotation, and gripper components.
        
        For ACT policy: action is joint positions -> compute FK -> add base pose
        For other policies: action is directly [x, y, z, qx, qy, qz, qw, gripper]
        """
        action_np = action_tensor.cpu().numpy() if isinstance(action_tensor, torch.Tensor) else action_tensor

        # ACT policy outputs joint positions, not end effector pose
        if self.config.policy_type.lower() == "act":
            # Expected ACT action format: [joint1, joint2, ..., jointN, gripper]
            num_joints = len(self.config.joint_position_keys)
            joint_positions = action_np[:num_joints]
            gripper_value = float(action_np[num_joints]) if len(action_np) > num_joints else 0.0
            
            # Compute forward kinematics
            ee_position, ee_rotation = self._compute_forward_kinematics(joint_positions)
            
            # Add robot base pose if available
            if self.latest_base_pose is not None:
                # Transform end effector pose from robot base frame to world frame
                # Base pose: [x, y, z, qx, qy, qz, qw]
                base_pos = self.latest_base_pose[:3]
                base_quat = self.latest_base_pose[3:7]
                
                # Simple position addition (for more accuracy, use proper quaternion rotation)
                # TODO: Implement proper homogeneous transformation if needed
                ee_position = ee_position + base_pos
                
                # For rotation, you may want to compose quaternions
                # For now, keeping EE rotation as-is (relative to base)
                self.logger.debug(
                    f"ACT FK result: pos={ee_position}, quat={ee_rotation}, "
                    f"base_pos={base_pos}, gripper={gripper_value}"
                )
            else:
                self.logger.warning("Base pose not available for ACT policy - using FK result without base transform")
            
            return {
                "pose": ee_position,
                "rotation": ee_rotation,
                "gripper": gripper_value,
            }
        
        # Other policies (e.g., GROOT, PI0) output end effector pose directly
        else:
            return {
                "pose": action_np[:3],  # [x, y, z]
                "rotation": action_np[3:7],  # [qx, qy, qz, qw]
                "gripper": float(action_np[7]) if len(action_np) > 7 else 0.0,
            }

    def _add_to_action_history(self, action: torch.Tensor):
        """Add executed action to history buffer for temporal conditioning."""
        # Convert to numpy for storage
        action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
        
        # Add to history
        self.action_history.append(action_np)
        
        # Keep only the most recent n_action_steps actions
        if len(self.action_history) > self.max_action_history:
            self.action_history = self.action_history[-self.max_action_history:]
        
        self.logger.debug(f"Action history size: {len(self.action_history)}/{self.max_action_history}")

    def _get_action_history_observation(self) -> Dict[str, np.ndarray]:
        """Get action history as observation features.
        
        Returns a dictionary with action history features that will be included
        in the observation sent to the policy. If we don't have enough history yet,
        we pad with zeros.
        """
        obs_dict = {}
        
        # If n_action_steps == 1, no history needed
        if self.max_action_history <= 1:
            return obs_dict
        
        # Pad history with zeros if we don't have enough yet
        history_size = len(self.action_history)
        action_dim = 8  # [x, y, z, qx, qy, qz, qw, gripper]
        
        # Create padded history (oldest to newest)
        padded_history = []
        for i in range(self.max_action_history):
            if i < (self.max_action_history - history_size):
                # Pad with zeros for missing history
                padded_history.append(np.zeros(action_dim, dtype=np.float32))
            else:
                # Use actual history
                hist_idx = i - (self.max_action_history - history_size)
                padded_history.append(self.action_history[hist_idx].astype(np.float32))
        
        # Stack into single array: shape (n_action_steps, action_dim)
        history_array = np.stack(padded_history, axis=0)
        
        # Add to observation dict with a special key that the policy will recognize
        # The policy training code should also use this key for action history
        # Keep as numpy array - server will convert to tensor
        obs_dict["action_history"] = history_array
        
        self.logger.debug(f"Action history observation shape: {history_array.shape}")
        return obs_dict

    def _publish_action(self, action_data: Dict[str, np.ndarray]):
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
                x=float(action_data["rotation"][0]),
                y=float(action_data["rotation"][1]),
                z=float(action_data["rotation"][2]),
                w=float(action_data["rotation"][3]),
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

    def _send_observation(self):
        """Capture and send observation to policy server."""
        try:
            # Build observation dictionary using hardware feature keys (not lerobot keys)
            raw_observation: RawObservation = {}

            # Add camera images (keep as numpy arrays - server will convert to tensors)
            for key in self.config.camera_keys:
                if key in self.camera_images:
                    raw_observation[key] = self.camera_images[key]  # Already numpy array

            # Add robot state: end effector pose + rotation + gripper (8 dimensions total)
            # NOTE: These should be SCALAR values (float), not tensors!
            # build_dataset_frame will convert them to the proper tensor format
            if self.latest_robot_pose is not None:
                # End effector position (x, y, z)
                raw_observation["ee_pos_x"] = float(self.latest_robot_pose[0])
                raw_observation["ee_pos_y"] = float(self.latest_robot_pose[1])
                raw_observation["ee_pos_z"] = float(self.latest_robot_pose[2])
                
                # End effector rotation (quaternion: qx, qy, qz, qw)
                raw_observation["ee_rot_qx"] = float(self.latest_robot_pose[3])
                raw_observation["ee_rot_qy"] = float(self.latest_robot_pose[4])
                raw_observation["ee_rot_qz"] = float(self.latest_robot_pose[5])
                raw_observation["ee_rot_qw"] = float(self.latest_robot_pose[6])
                    
            # Add gripper state from joint states
            if self.latest_joint_states is not None:
                num_arm_joints = len(self.config.joint_position_keys)
                if len(self.latest_joint_states) > num_arm_joints:
                    raw_observation["gripper"] = float(self.latest_joint_states[num_arm_joints])
                else:
                    # Fallback to 0 if no gripper data
                    raw_observation["gripper"] = 0.0

            # Add action history if configured (for temporal conditioning)
            action_history_obs = self._get_action_history_observation()
            for key, value in action_history_obs.items():
                raw_observation[key] = value  # type: ignore

            # Add task (keep as string - will be handled by the observation processor)
            raw_observation["task"] = self.config.task  # type: ignore

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(), observation=raw_observation, timestep=max(latest_action, 0)
            )

            # Set must_go flag - tell server to run inference if action queue is empty
            with self.action_queue_lock:
                action_queue_empty = self.action_queue.empty()
            
            observation.must_go = action_queue_empty
            self.logger.debug(f"Observation must_go={observation.must_go}, queue_empty={action_queue_empty}")

            # Send observation
            _ = self.send_observation(observation)

            # Calculate FPS
            fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())
            self.logger.info(
                f"Sent observation #{observation.get_timestep()} | "
                f"Must go (queue empty): {observation.must_go} | "
                f"Avg FPS: {fps_metrics['avg_fps']:.2f}"
            )

        except Exception as e:
            self.logger.error(f"Error sending observation: {e}")


@draccus.wrap()
def async_client_ros2(cfg: RobotClientROS2Config):
    """Main entry point for ROS2 robot client."""
    logging.info("="*80)
    logging.info("ROS2 Robot Client Configuration:")
    logging.info("="*80)
    logging.info(pformat(asdict(cfg)))
    logging.info("="*80)

    # Initialize ROS2
    rclpy.init()

    try:
        # Create client node
        client = RobotClientROS2(cfg)

        # Start policy connection
        if not client.start():
            client.logger.error("Failed to start client")
            return

        # Start action receiver thread
        client.logger.info("🚀 Starting action receiver thread...")
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()
        client.logger.info(f"📊 Action receiver thread alive: {action_receiver_thread.is_alive()}")

        # Spin ROS2 node (control loop runs via timer callbacks)
        client.logger.info("🔄 Starting ROS2 spin loop (control loop will run via timer callbacks)...")
        rclpy.spin(client)

    except KeyboardInterrupt:
        client.logger.info("Shutdown requested by user")
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        # Cleanup
        if 'client' in locals():
            client.stop()
            if 'action_receiver_thread' in locals():
                action_receiver_thread.join(timeout=2.0)
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.destroy_node()

        rclpy.shutdown()
        logging.info("ROS2 Robot Client stopped")


if __name__ == "__main__":
    async_client_ros2()
