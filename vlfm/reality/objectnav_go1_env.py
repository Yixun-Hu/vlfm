# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pyzed.sl as sl
from depth_camera_filtering.depth_camera_filtering import filter_depth

from vlfm.reality.pointnav_go1_env import PointNavEnv
from vlfm.reality.robots.camera_ids import Go1CamIds
from vlfm.utils.geometry_utils import get_fov, wrap_heading
from vlfm.utils.img_utils import reorient_rescale_map, resize_images

# For Go1, we only have one camera (LEFT)
VALUE_MAP_CAMS = []  # Empty list to disable value map

POINT_CLOUD_CAMS = [
    Go1CamIds.LEFT,
]

ALL_CAMS = list(set(VALUE_MAP_CAMS + POINT_CLOUD_CAMS))


class ObjectNavEnv(PointNavEnv):
    """
    Gym environment for doing the ObjectNav task on the Go1 robot with ZED camera in the real world.
    
    Note: Go1 only has one camera (LEFT), no gripper camera. The max_gripper_cam_depth
    parameter is kept for API compatibility but is used for the single camera.
    """

    tf_episodic_to_global: np.ndarray = np.eye(4)  # must be set in reset()
    tf_global_to_episodic: np.ndarray = np.eye(4)  # must be set in reset()
    episodic_start_yaw: float = float("inf")  # must be set in reset()
    target_object: str = ""  # must be set in reset()

    def __init__(self, max_gripper_cam_depth: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # For Go1, we only have one camera (LEFT), so max_gripper_cam_depth is used for it
        # This parameter is kept for API compatibility with Spot version
        self._max_gripper_cam_depth = max_gripper_cam_depth
        # Get the current date and time
        now = datetime.now()
        # Format it into a string in the format MM-DD-HH-MM-SS
        date_string = now.strftime("%m-%d-%H-%M-%S")
        self._vis_dir = f"{date_string}"
        os.makedirs(f"vis/{self._vis_dir}", exist_ok=True)

    def reset(self, goal: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        assert isinstance(goal, str)
        self.target_object = goal
        # Transformation matrix from where the robot started to the global frame
        self.tf_episodic_to_global: np.ndarray = self.robot.get_transform()
        self.tf_episodic_to_global[2, 3] = 0.0  # Make z of the tf 0.0
        self.tf_global_to_episodic = np.linalg.inv(self.tf_episodic_to_global)
        self.episodic_start_yaw = self.robot.xy_yaw[1]
        return self._get_obs()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        # Go1 doesn't have an arm/gripper, so we skip arm control
        # Parent class handles base movement
        vis_imgs = []
        time_id = time.time()
        for k in ["annotated_rgb", "annotated_depth", "obstacle_map", "value_map"]:
            if k in action.get("info", {}):
                img = cv2.cvtColor(action["info"][k], cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"vis/{self._vis_dir}/{time_id}_{k}.png", img)
                if "map" in k:
                    img = reorient_rescale_map(img)
                if k == "annotated_depth" and np.array_equal(img, np.ones_like(img) * 255):
                    # Put text in the middle saying "Target not currently detected"
                    text = "Target not currently detected"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
                    cv2.putText(
                        img,
                        text,
                        (img.shape[1] // 2 - text_size[0] // 2, img.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        1,
                    )
                vis_imgs.append(img)
        
        if vis_imgs:
            vis_img = np.hstack(resize_images(vis_imgs, match_dimension="height"))
            cv2.imwrite(f"vis/{self._vis_dir}/{time_id}.jpg", vis_img)
            if os.environ.get("ZSOS_DISPLAY", "0") == "1":
                cv2.imshow("Visualization", cv2.resize(vis_img, (0, 0), fx=0.5, fy=0.5))
                cv2.waitKey(1)

        # Handle base movement (arm_yaw == -1 means just move base)
        if action.get("arm_yaw", -1) == -1:
            return super().step(action)
        
        # Go1 doesn't have an arm, so if arm_yaw is specified, just move base
        # (This maintains API compatibility but ignores arm commands)
        done = False
        self._num_steps += 1
        return self._get_obs(), 0.0, done, {}

    def _get_obs(self) -> Dict[str, Any]:
        robot_xy, robot_heading = self._get_gps(), self._get_compass()
        nav_depth, obstacle_map_depths, value_map_rgbd, object_map_rgbd = self._get_camera_obs()
        return {
            "nav_depth": nav_depth,
            "robot_xy": robot_xy,
            "robot_heading": robot_heading,
            "objectgoal": self.target_object,
            "obstacle_map_depths": obstacle_map_depths,
            "value_map_rgbd": value_map_rgbd,
            "object_map_rgbd": object_map_rgbd,
        }

    def _get_camera_obs(self) -> Tuple[np.ndarray, List, List, List]:
        """
        Poll all necessary cameras on the robot and return their images, focal lengths,
        and transforms to the global frame.
        
        Note: Go1 only has one camera (LEFT), so all camera data comes from it.
        """
        srcs: List[str] = ALL_CAMS
        cam_data = self.robot.get_camera_data(srcs)
        
        # Get depth map from ZED
        depth_img = self._get_zed_depth()
        
        for src in ALL_CAMS:
            tf = self.tf_global_to_episodic @ cam_data[src]["tf_camera_to_global"]
            # ZED SDK already returns world frame in RIGHT_HANDED_Z_UP_X_FWD convention
            # (X forward, Y left, Z up), which matches our coordinate system.
            # TODO: if the tf_camera_to_global is not in the right convention, we need to rotate it
            # No additional rotation matrix needed for Go1/ZED.
            cam_data[src]["tf_camera_to_global"] = tf  # Remove rotation_matrix multiplication
            
            img = cam_data[src]["image"]

            # For Go1, we use depth from ZED instead of separate depth cameras
            # RGB images are already in cam_data
            if img.dtype == np.uint8:
                if img.ndim == 2 or img.shape[2] == 1:
                    cam_data[src]["image"] = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    cam_data[src]["image"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        min_depth = 0
        # For Go1, we only have one camera (LEFT), no gripper camera
        # max_gripper_cam_depth is used for the single camera depth normalization
        max_depth = self._max_gripper_cam_depth

        # Object map output is a list of (rgb, depth, tf_camera_to_episodic, min_depth,
        # max_depth, fx, fy) for each of the cameras
        src = Go1CamIds.LEFT
        rgb = cam_data[src]["image"]
        # Use ZED depth map
        depth_normalized = self._norm_depth(depth_img, max_depth=max_depth)
        tf = cam_data[src]["tf_camera_to_global"]
        fx, fy = cam_data[src]["fx"], cam_data[src]["fy"]
        object_map_rgbd = [(rgb, depth_normalized, tf, min_depth, max_depth, fx, fy)]

        # Nav depth uses the single LEFT camera depth
        nav_depth = depth_normalized.copy()
        nav_depth = filter_depth(nav_depth, blur_type=None, set_black_value=1.0)

        # Obstacle map output is a list of (depth, tf, min_depth, max_depth, fx, fy,
        # topdown_fov) for each of the cameras
        obstacle_map_depths = []
        src = Go1CamIds.LEFT
        depth = depth_normalized.copy()
        fx, fy = cam_data[src]["fx"], cam_data[src]["fy"]
        tf = cam_data[src]["tf_camera_to_global"]
        fov = get_fov(fx, depth.shape[1])
        src_data = (depth, tf, min_depth, self._max_body_cam_depth, fx, fy, fov)
        obstacle_map_depths.append(src_data)

        # Value map output is a list of (rgb, depth, tf_camera_to_episodic, min_depth,
        # max_depth, fov) for each of the cameras in VALUE_MAP_CAMS
        value_map_rgbd = []
        for src in VALUE_MAP_CAMS:
            rgb = cam_data[src]["image"]
            depth = depth_normalized.copy()
            fx = cam_data[src]["fx"]
            tf = cam_data[src]["tf_camera_to_global"]
            fov = get_fov(fx, rgb.shape[1])
            src_data = (rgb, depth, tf, min_depth, max_depth, fov)
            value_map_rgbd.append(src_data)

        return nav_depth, obstacle_map_depths, value_map_rgbd, object_map_rgbd

    def _get_zed_depth(self) -> np.ndarray:
        """Get depth map from ZED camera.
        
        Returns:
            np.ndarray: Depth map in meters (H, W)
        """
        return self.robot.go1_zed.get_depth()

    def _get_gps(self) -> np.ndarray:
        """
        Get the (x, y) position of the robot's base in the episode frame. x is forward,
        y is left.
        """
        global_xy = self.robot.xy_yaw[0]
        start_xy = self.tf_episodic_to_global[:2, 3]
        offset = global_xy - start_xy
        rotation_matrix = np.array(
            [
                [np.cos(-self.episodic_start_yaw), -np.sin(-self.episodic_start_yaw)],
                [np.sin(-self.episodic_start_yaw), np.cos(-self.episodic_start_yaw)],
            ]
        )
        episodic_xy = rotation_matrix @ offset
        return episodic_xy

    def _get_compass(self) -> float:
        """
        Get the yaw of the robot's base in the episode frame. Yaw is measured in radians
        counterclockwise from the z-axis.
        """
        global_yaw = self.robot.xy_yaw[1]
        episodic_yaw = wrap_heading(global_yaw - self.episodic_start_yaw)
        return episodic_yaw
