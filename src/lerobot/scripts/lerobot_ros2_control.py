"""
Executes a policy with ROS2 interface.

Example:

```shell
python lerobot/scripts/lerobot_ros2_control.py \
    --robot.camera_topics="['/camera/rgb/image_raw']" \
    --robot.camera_keys="['laptop']" \
    --robot.robot_state_topic="/joint_states" \
    --robot.robot_pose_topic="/ee_pose" \
    --robot.ee_pose_topic="/target_pose" \
    --robot.gripper_topic="/target_gripper" \
    --policy.path=lerobot/test \
    --display_data=true
```
"""

import logging
import threading
import time
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from std_msgs.msg import Bool

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    is_headless,
    predict_action,
)
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class ROS2RobotConfig:
    # ROS2 topic configuration
    camera_topics: List[str] = field(default_factory=lambda: ['/dataset/scene_camera/rgb', '/dataset/wrist_camera/rgb'])
    camera_keys: List[str] = field(default_factory=lambda: ['scene_camera', 'wrist_camera'])
    robot_state_topic: str = "/dataset/joint_states"
    robot_pose_topic: str = "/dataset/right_arm_ee_pose"
    ee_pose_topic: str = "/right_hand/pose"
    gripper_topic: str = "/right_hand/trigger"
    
    # Robot type for policy compatibility
    type: str = "ros2_robot"
    # Mock camera config for compatibility
    cameras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.camera_topics) != len(self.camera_keys):
            raise ValueError("camera_topics and camera_keys must have the same length")


class ROS2Robot(Node):
    def __init__(self, config: ROS2RobotConfig):
        super().__init__('lerobot_ros2_node')
        self.config = config
        self.robot_type = config.type
        self.name = "ros2_robot"
        
        # Data storage
        self.camera_images = {}
        self.latest_joint_states = None
        self.latest_robot_pose = None
        self.connected = False
        
        # Features for dataset/policy compatibility
        self.cameras = {k: {} for k in config.camera_keys}  # Mock cameras
        
        # Define features based on what we expect from ROS2
        # We use the simplified format expected by hw_to_dataset_features
        # Using 8 dimensions (EE pose + gripper) to match policy expectation
        self.observation_features = {
            "x": float,
            "y": float,
            "z": float,
            "qw": float,
            "qx": float,
            "qy": float,
            "qz": float,
            "gripper": float,
        }
        
        for key in config.camera_keys:
            self.observation_features[key] = (480, 640, 3) # Placeholder shape

        self.action_features = {
            "x": float,
            "y": float,
            "z": float,
            "qw": float,
            "qx": float,
            "qy": float,
            "qz": float,
            "gripper": float,
        }

        self._spin_thread = None

    def connect(self):
        if self.connected:
            return

        qos_profile = QoSProfile(depth=10)

        # Subscribers
        for i, topic in enumerate(self.config.camera_topics):
            key = self.config.camera_keys[i]
            self.create_subscription(
                Image,
                topic,
                lambda msg, k=key: self.camera_callback(msg, k),
                qos_profile
            )
            self.get_logger().info(f"Subscribed to camera: {topic} -> {key}")

        self.create_subscription(
            JointState,
            self.config.robot_state_topic,
            self.joint_state_callback,
            qos_profile
        )
        
        self.create_subscription(
            PoseStamped,
            self.config.robot_pose_topic,
            self.robot_pose_callback,
            qos_profile
        )

        # Publishers
        self.ee_pose_pub = self.create_publisher(Pose, self.config.ee_pose_topic, qos_profile)
        self.gripper_pub = self.create_publisher(Bool, self.config.gripper_topic, qos_profile)

        # Start spinning in a thread
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        self._spin_thread.start()
        
        self.connected = True
        self.get_logger().info("ROS2 Robot connected")

    def disconnect(self):
        if self.connected:
            # rclpy.shutdown() is handled globally usually, but we can stop the node
            pass
        self.connected = False

    def camera_callback(self, msg: Image, key: str):
        try:
            # Basic conversion, similar to eval_lerobot_ros2.py
            if msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'bgr8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                img = img[:, :, ::-1]
            else:
                return
            self.camera_images[key] = img
        except Exception as e:
            self.get_logger().error(f"Error in camera callback: {e}")

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_states = np.array(msg.position, dtype=np.float32)

    def robot_pose_callback(self, msg: PoseStamped):
        p = msg.pose.position
        o = msg.pose.orientation
        self.latest_robot_pose = np.array([p.x, p.y, p.z, o.w, o.x, o.y, o.z], dtype=np.float32)

    def get_observation(self) -> RobotObservation:
        # Wait for data if needed, or return latest
        if self.latest_robot_pose is None or self.latest_joint_states is None:
            print("Waiting for robot state and pose...")
            return {}

        # Check if all cameras are available
        if len(self.camera_images) < len(self.config.camera_keys):
            print("Waiting for all camera images...")
            return {}

        obs = {}
        
        # Use EE pose and gripper for state (8 dims) to match policy expectation
        # latest_robot_pose is [x, y, z, qw, qx, qy, qz]
        obs["x"] = float(self.latest_robot_pose[0])
        obs["y"] = float(self.latest_robot_pose[1])
        obs["z"] = float(self.latest_robot_pose[2])
        obs["qw"] = float(self.latest_robot_pose[3])
        obs["qx"] = float(self.latest_robot_pose[4])
        obs["qy"] = float(self.latest_robot_pose[5])
        obs["qz"] = float(self.latest_robot_pose[6])
        
        # Assuming gripper is the last joint
        if len(self.latest_joint_states) > 0:
            obs["gripper"] = float(self.latest_joint_states[-1])
        else:
            obs["gripper"] = 0.0
        
        # Add robot pose if available/needed by policy (some policies use it as state)
        # Note: Standard LeRobot policies often use joint positions as 'state'.
        # If your policy expects EE pose, you might need to map it here.
        
        for key in self.config.camera_keys:
            if key in self.camera_images:
                obs[key] = self.camera_images[key]
        
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        # action is typically a numpy array
        # We assume the action vector contains [x, y, z, qw, qx, qy, qz, gripper] (8 elements)
        
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        if isinstance(action, (np.ndarray, list)):
            action = np.array(action).flatten()
            
            if len(action) >= 7:
                pose_msg = Pose()
                pose_msg.position = Point(x=float(action[0]), y=float(action[1]), z=float(action[2]))
                
                # Rotation
                if len(action) >= 7:
                    # Assuming [x, y, z, qw, qx, qy, qz, gripper]
                    # Note: eval_lerobot_ros2 uses [qw, qx, qy, qz] for rotation
                    pose_msg.orientation = Quaternion(
                        w=float(action[3]),
                        x=float(action[4]),
                        y=float(action[5]),
                        z=float(action[6])
                    )
                
                self.ee_pose_pub.publish(pose_msg)
                
                if len(action) >= 8:
                    gripper_msg = Bool()
                    gripper_msg.data = bool(action[7] > 0.5)
                    self.gripper_pub.publish(gripper_msg)
        
        return action


@dataclass
class DatasetRecordConfig:
    # Dataset identifier.
    repo_id: str = "ros2_debug/test"
    # A short description.
    single_task: str = "ROS2 Control"
    # Root directory.
    root: str | Path | None = None
    # FPS
    fps: int = 3
    # Recording settings
    video: bool = True
    push_to_hub: bool = False
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    rename_map: dict[str, str] = field(default_factory=dict)


@dataclass
class ControlConfig:
    robot: ROS2RobotConfig = field(default_factory=ROS2RobotConfig)
    dataset: DatasetRecordConfig = field(default_factory=DatasetRecordConfig)
    policy: PreTrainedConfig | None = None
    display_data: bool = False
    play_sounds: bool = True

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def control_loop(
    robot: ROS2Robot,
    fps: int,
    robot_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
    dataset: LeRobotDataset | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline | None = None,
    postprocessor: PolicyProcessorPipeline | None = None,
    display_data: bool = False,
):
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    while True:
        start_loop_t = time.perf_counter()

        # Get robot observation
        obs = robot.get_observation()
        if not obs:
            # Wait for data
            time.sleep(0.1)
            continue

        obs_processed = robot_observation_processor(obs)

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from policy
        if policy is not None and preprocessor is not None and postprocessor is not None:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=dataset.meta.task if dataset else "",
                robot_type=robot.robot_type,
            )
            act_processed_policy = make_robot_action(action_values, dataset.features)
        else:
            # No policy, maybe just logging?
            act_processed_policy = None

        if act_processed_policy is not None:
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
            robot.send_action(robot_action_to_send)
            logging.info("Action sent")
            
            if dataset is not None:
                action_frame = build_dataset_frame(dataset.features, act_processed_policy, prefix=ACTION)
                frame = {**observation_frame, **action_frame}
                dataset.add_frame(frame)

        if display_data and act_processed_policy is not None:
            log_rerun_data(observation=obs_processed, action=act_processed_policy)

        dt_s = time.perf_counter() - start_loop_t
        sleep_time = 1 / fps - dt_s
        if sleep_time > 0:
            time.sleep(sleep_time)


@parser.wrap()
def main(cfg: ControlConfig):
    init_logging()
    rclpy.init()
    
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="ros2_control")

    # Initialize ROS2 Robot
    robot = ROS2Robot(cfg.robot)
    robot.connect()

    # Setup processors
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    # Setup Dataset (even if just for features/logging)
    # We need to infer features from the policy if possible, or use defaults
    
    # Load policy
    policy = None
    preprocessor = None
    postprocessor = None
    
    if cfg.policy:
        # Create a dummy dataset to get features/stats if needed, or load from policy
        # For now, let's assume we can load policy without a full dataset if we have the config
        # But make_policy usually requires ds_meta for stats.
        
        # If we are just running inference, we might need to load stats from the policy's dataset_stats.json
        # which is handled by PreTrainedConfig/Policy.
        
        # We need a dummy dataset to hold features for build_dataset_frame
        # We can try to construct it from the policy config if available
        pass

    # For simplicity in this script, we'll create a dataset if we can, or just run if policy allows.
    # But LeRobot structure relies heavily on Dataset object for feature definitions.
    
    # Let's create a temporary dataset to handle features
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        )
    )

    dataset = None
    
    # Determine dataset path to clean up if it exists
    if cfg.dataset.root:
        dataset_path = Path(cfg.dataset.root)
    else:
        dataset_path = Path.home() / ".cache/huggingface/lerobot" / cfg.dataset.repo_id

    if dataset_path.exists():
        logging.warning(f"Dataset path {dataset_path} already exists. Deleting it to start fresh.")
        shutil.rmtree(dataset_path)

    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=cfg.dataset.root,
        robot_type=robot.robot_type,
        features=dataset_features,
        use_videos=cfg.dataset.video,
    )

    if cfg.policy:
        policy = make_policy(cfg.policy, ds_meta=dataset.meta)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
        )

    try:
        control_loop(
            robot=robot,
            fps=cfg.dataset.fps,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            dataset=dataset,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            display_data=cfg.display_data,
        )
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
