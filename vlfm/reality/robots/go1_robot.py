# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, List, Tuple

import numpy as np
from .go1_zed_wrapper.go1_zed import Go1Zed, image_response_to_cv2

from .base_robot import BaseRobot
from .frame_ids import Go1ZedFrameIds

MAX_CMD_DURATION = 5


class Go1Robot(BaseRobot):
    def __init__(self, go1_zed: Go1Zed):
        self.go1_zed = go1_zed

    @property
    def spot(self):
        """
        Spot-like API compatibility layer.
        Returns self to allow robot.spot.set_base_position() calls.
        """
        return self

    def set_base_position(
        self,
        x_pos: float,
        y_pos: float,
        yaw: float,
        end_time: float = 100.0,
        relative: bool = True,
        max_fwd_vel: float = 0.3,
        max_hor_vel: float = 0.2,
        max_ang_vel: float = np.deg2rad(60),
        disable_obstacle_avoidance: bool = False,
        blocking: bool = False,
    ) -> int:
        """Set base position (Spot-like API wrapper)."""
        return self.go1_zed.set_base_position(
            x_pos=x_pos,
            y_pos=y_pos,
            yaw=yaw,
            end_time=end_time,
            relative=relative,
            max_fwd_vel=max_fwd_vel,
            max_hor_vel=max_hor_vel,
            max_ang_vel=max_ang_vel,
            disable_obstacle_avoidance=disable_obstacle_avoidance,
            blocking=blocking,
        )

    def get_cmd_feedback(self, cmd_id: int):
        """Get command feedback (Spot-like API wrapper)."""
        return self.go1_zed.get_cmd_feedback(cmd_id)

    @property
    def xy_yaw(self) -> Tuple[np.ndarray, float]:
        """Returns [x, y], yaw"""
        x, y, yaw = self.go1_zed.get_xy_yaw(use_boot_origin=True)
        return np.array([x, y]), yaw


    def get_camera_images(self, camera_source: List[str]) -> Dict[str, np.ndarray]:
        """Returns a dict of images mapping camera ids to images
        # Currently, we only have one camera

        Args:
            camera_source (List[str]): List of camera ids to get images from

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping camera ids to images
        """
        image_responses = self.go1_zed.get_image_responses(camera_source)
        imgs = {
            source: image_response_to_cv2(image_response, reorient=True)
            for source, image_response in zip(camera_source, image_responses)
        }
        return imgs

    def command_base_velocity(self, ang_vel: float, lin_vel: float) -> None:
        """Commands the base to execute given angular/linear velocities, non-blocking

        Args:
            ang_vel (float): Angular velocity in radians per second
            lin_vel (float): Linear velocity in meters per second
        """
        # Just make the robot stop moving if both velocities are very low
        if np.abs(ang_vel) < 0.01 and np.abs(lin_vel) < 0.01:
            self.go1_zed.stand()
        else:
            self.go1_zed.set_base_velocity(
                lin_vel,
                0.0,  # no horizontal velocity
                ang_vel,
                MAX_CMD_DURATION,
            )

    def get_transform(self) -> np.ndarray:
        """Returns the transformation matrix of the camera frame
        # Currently, we use the pose of the camera as the pose of the base
        # TODO: update this when we have a base frame, and have tf between the base and the camera
        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        return self.go1_zed.get_transform()

    def get_camera_data(self, srcs: List[str]) -> Dict[str, Dict[str, Any]]:
        """Returns a dict that maps each camera id to its image, focal lengths, and
        transform matrix (from camera to global frame).

        Args:
            srcs (List[str]): List of camera ids to get images from

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping camera ids to images
        """
        image_responses = self.go1_zed.get_image_responses(srcs)
        imgs = {
            src: self._camera_response_to_data(image_response) for src, image_response in zip(srcs, image_responses)
        }
        return imgs

    def _camera_response_to_data(self, response: Any) -> Dict[str, Any]:
        image: np.ndarray = image_response_to_cv2(response, reorient=False)
        fx: float = response.source.pinhole.intrinsics.focal_length.x
        fy: float = response.source.pinhole.intrinsics.focal_length.y
        # We don't need a tf_snapshot
        # TODO: update this when we need to use a tf_snapshot and a camera frame to get the tf
        return {
            "image": image,
            "fx": fx,
            "fy": fy,
            "tf_camera_to_global": self.go1_zed.get_transform(),
        }
