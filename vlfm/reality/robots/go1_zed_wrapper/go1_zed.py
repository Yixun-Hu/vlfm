""" Easy-to-use wrapper for properly controlling ZED plus Go1 robot """
import os
import os.path as osp
import time
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np
import pyzed.sl as sl

# Import unitree_legged_sdk
# Assuming the SDK is installed or available in the path
try:
    # Try to import from the local SDK path
    sdk_path = osp.join(osp.dirname(__file__), "..", "unitree_legged_sdk", "lib", "python", "arm64")
    if osp.exists(sdk_path):
        sys.path.insert(0, sdk_path)
    import robot_interface as sdk
    UNITREE_SDK_AVAILABLE = True
except ImportError:
    UNITREE_SDK_AVAILABLE = False
    print("Warning: unitree_legged_sdk not available. Robot control functions will not work.")


@dataclass
class ImageResponse:
    """Wrapper class to mimic Spot API image response structure"""
    source: Any
    shot: Any


@dataclass
class Source:
    """Wrapper for camera source information"""
    name: str
    pinhole: Any


@dataclass
class PinholeIntrinsics:
    """Wrapper for pinhole camera intrinsics"""
    focal_length: Any


@dataclass
class FocalLength:
    """Wrapper for focal length"""
    x: float
    y: float


@dataclass
class Shot:
    """Wrapper for image shot information"""
    image: Any
    frame_name_image_sensor: str
    transforms_snapshot: Optional[Any] = None


@dataclass
class ImageData:
    """Wrapper for image data"""
    data: bytes
    pixel_format: int
    format: int
    rows: int
    cols: int


def image_response_to_cv2(image_response: ImageResponse, reorient: bool = True) -> np.ndarray:
    """
    Convert ZED image response to OpenCV format.
    This mimics the Spot API's image_response_to_cv2 function.
    
    Args:
        image_response: ImageResponse object containing image data
        reorient: Whether to reorient the image (not used for ZED, kept for API compatibility)
    
    Returns:
        np.ndarray: Image as numpy array
    """
    # For ZED, we directly return the image data
    # The image_response.shot.image should be a numpy array already
    img = image_response.shot.image
    if isinstance(img, np.ndarray):
        return img.copy()
    else:
        raise ValueError(f"Unexpected image type: {type(img)}")


class CommandStatus(Enum):
    """Command status enum (mimicking Spot API)"""
    STATUS_UNKNOWN = 0
    STATUS_PROCESSING = 1
    STATUS_COMMAND_OVERRIDDEN = 2
    STATUS_COMMAND_TIMED_OUT = 3
    STATUS_ROBOT_FROZEN = 4
    STATUS_INCOMPATIBLE_HARDWARE = 5
    STATUS_COMMAND_COMPLETE = 1  # Same as PROCESSING for simplicity


@dataclass
class CommandFeedback:
    """Mock command feedback structure (mimicking Spot API)"""
    feedback: Any  # Will be a nested structure
    
    class Feedback:
        class SynchronizedFeedback:
            class MobilityCommandFeedback:
                class SE2TrajectoryFeedback:
                    status: int = CommandStatus.STATUS_PROCESSING.value
                    
                    def __init__(self):
                        self.status = CommandStatus.STATUS_PROCESSING.value
                
                def __init__(self):
                    self.se2_trajectory_feedback = self.SE2TrajectoryFeedback()
            
            def __init__(self):
                self.mobility_command_feedback = self.MobilityCommandFeedback()
        
        def __init__(self):
            self.synchronized_feedback = self.SynchronizedFeedback()
    
    def __init__(self, status: int = CommandStatus.STATUS_PROCESSING.value):
        self.feedback = self.Feedback()
        self.feedback.synchronized_feedback.mobility_command_feedback.se2_trajectory_feedback.status = status


@dataclass
class PositionCommand:
    """Internal structure to track position commands"""
    cmd_id: int
    target_x: float
    target_y: float
    target_yaw: float
    relative: bool
    start_time: float
    start_x: float
    start_y: float
    start_yaw: float
    max_fwd_vel: float
    max_hor_vel: float
    max_ang_vel: float
    status: CommandStatus = CommandStatus.STATUS_PROCESSING
    timeout: float = 30.0  # Default timeout in seconds


class Go1Zed:
    def __init__(
        self,
        resolution: sl.RESOLUTION = sl.RESOLUTION.HD720,
        depth_mode: sl.DEPTH_MODE = sl.DEPTH_MODE.NEURAL,
        coordinate_system: sl.COORDINATE_SYSTEM = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD,
        max_depth_m: float = 8.0,
        use_world_frame: sl.REFERENCE_FRAME = sl.REFERENCE_FRAME.WORLD,
        camera_fps: int = 30,
    ):
        # ----------- initialize ZED -----------
        init_params = sl.InitParameters()
        init_params.depth_mode = depth_mode
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = coordinate_system
        init_params.depth_maximum_distance = max_depth_m
        init_params.camera_resolution = resolution
        init_params.camera_fps = camera_fps
        self._zed = sl.Camera()
        self._image = sl.Mat()
        self._depth = sl.Mat()
        self._pose = sl.Pose()
        self._fx = None
        self._fy = None
        self._image_size = None
        self._runtime_params = sl.RuntimeParameters()
        
        # Initialize Unitree Go1 robot control
        self._unitree_udp = None
        self._unitree_cmd = None
        self._unitree_state = None
        self._unitree_safe = None
        if UNITREE_SDK_AVAILABLE:
            try:
                HIGHLEVEL = 0xee
                # Default Go1 IP: 192.168.123.161, port: 8082
                # You may need to adjust these based on your network configuration
                self._unitree_udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)
                self._unitree_safe = sdk.Safety(sdk.LeggedType.Go1)
                self._unitree_cmd = sdk.HighCmd()
                self._unitree_state = sdk.HighState()
                self._unitree_udp.InitCmdData(self._unitree_cmd)
            except Exception as e:
                print(f"Warning: Failed to initialize Unitree SDK: {e}")
                self._unitree_udp = None
        
        # Open Camera
        status = self._zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"[Go1Zed] Camera open failed: {status}")
        
        # Enable positional tracking
        # Note: ZED SDK automatically sets world frame origin at the first position where
        # motion tracking starts, so no manual boot origin initialization is needed.
        tracking_params = sl.PositionalTrackingParameters()
        err = self._zed.enable_positional_tracking(tracking_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to enable positional tracking: {err}")
        
        # Get camera info
        cam_info = self._zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters.left_cam
        self._fx = float(calib.fx)
        self._fy = float(calib.fy)
        self._image_size = (
            cam_info.camera_configuration.resolution.width,
            cam_info.camera_configuration.resolution.height,
        )

        # Command tracking system
        self._cmd_counter = 0
        self._active_commands: Dict[int, PositionCommand] = {}
        self._cmd_lock = threading.Lock()
        self._control_thread: Optional[threading.Thread] = None
        self._stop_control_thread = False

    def stand(self, timeout_sec: float = 10.0) -> int:
        """
        Command the robot to stand.
        
        Note: Unitree SDK requires continuous command sending (typically every 2ms).
        This function sends the command once. For continuous control, call this
        function repeatedly in your control loop.
        
        Args:
            timeout_sec: Timeout in seconds (not used, kept for API compatibility)
            
        Returns:
            Command ID (always 0 for high-level commands)
        """
        if not UNITREE_SDK_AVAILABLE or self._unitree_udp is None:
            raise RuntimeError("Unitree SDK not available. Cannot control robot.")
        
        # Set mode to 0 (idle/stand) or 1 (force stand)
        # Mode 0: default stand (idle)
        # Mode 1: force stand (controlled by bodyHeight + euler angles)
        self._unitree_cmd.mode = 1  # idle/stand mode
        self._unitree_cmd.gaitType = 0
        self._unitree_cmd.speedLevel = 0
        self._unitree_cmd.footRaiseHeight = 0.0
        self._unitree_cmd.bodyHeight = 0.0
        self._unitree_cmd.euler[0] = 0.0  # roll
        self._unitree_cmd.euler[1] = 0.0  # pitch
        self._unitree_cmd.euler[2] = 0.0  # yaw
        self._unitree_cmd.velocity[0] = 0.0  # forward speed
        self._unitree_cmd.velocity[1] = 0.0  # side speed
        self._unitree_cmd.yawSpeed = 0.0
        
        # Receive current state for safety check
        self._unitree_udp.Recv()
        self._unitree_udp.GetRecv(self._unitree_state)
        # Apply safety checks
        if hasattr(self._unitree_safe, 'PowerProtect'):
            self._unitree_safe.PowerProtect(self._unitree_cmd, self._unitree_state, 1)
        self._unitree_udp.SetSend(self._unitree_cmd)
        self._unitree_udp.Send()
        # Wait for robot to stand
        start = time.time()
        while time.time() - start < timeout_sec:
            self._unitree_udp.Recv()
            self._unitree_udp.GetRecv(self._unitree_state)
            if self._unitree_state.mode == 1:
                return
            time.sleep(0.01)
        raise RuntimeError("Timeout waiting for robot to stand")

    def _update_pose_world(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update and return the current pose in world coordinates.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Translation [x, y, z] and quaternion [w, x, y, z]
        """
        py_translation = sl.Translation()
        py_orientation = sl.Orientation()
        tracking_state = self._zed.get_position(self._pose, sl.REFERENCE_FRAME.WORLD)
        print(f"zed camera tracking_state: {tracking_state}")
        
        if tracking_state != sl.POSITIONAL_TRACKING_STATE.OK:
            raise RuntimeError(f"ZED tracking not OK: {tracking_state}")
        
        translation = np.array(self._pose.get_translation(py_translation).get(), dtype=np.float32)
        orientation = np.array(self._pose.get_orientation(py_orientation).get(), dtype=np.float32)
        # ZED returns quaternion as [x, y, z, w], convert to [w, x, y, z]
        if len(orientation) == 4:
            orientation = np.array([orientation[3], orientation[0], orientation[1], orientation[2]])
        
        return translation, orientation

    @staticmethod
    def _quat_to_rot(quat: np.ndarray) -> np.ndarray:
        """
        Converts a quaternion [w, x, y, z] to a 3x3 rotation matrix.
        
        Args:
            quat: Quaternion as [w, x, y, z] (already converted from ZED's [x, y, z, w] format)
            
        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        if len(quat) != 4:
            raise ValueError(f"Quaternion must have 4 elements, got {len(quat)}")
        
        # Input is already in [w, x, y, z] format (converted in _update_pose_world)
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm < 1e-8:
            raise ValueError("Quaternion norm is too small")
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to rotation matrix
        rot = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return rot

    def _pose_to_tf_matrix(self, translation: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """
        Convert translation and quaternion to 4x4 transformation matrix.
        
        Args:
            translation: [x, y, z] translation vector
            orientation: [w, x, y, z] quaternion
            
        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        rot = self._quat_to_rot(orientation)
        tf = np.eye(4)
        tf[:3, :3] = rot
        tf[:3, 3] = translation
        return tf

    def get_xy_yaw(self, use_boot_origin: bool = True) -> Tuple[float, float, float]:
        """
        Get the robot's position [x, y] and yaw angle.
        
        Args:
            use_boot_origin: This parameter is kept for API compatibility but has no effect.
                            ZED SDK automatically sets world frame origin at the first position
                            where motion tracking starts, so the coordinates are already relative
                            to the boot origin.
            
        Returns:
            Tuple[float, float, float]: x, y, yaw
        """
        translation, orientation = self._update_pose_world()
        
        # Convert quaternion to rotation matrix to extract yaw
        rot = self._quat_to_rot(orientation)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        
        # ZED SDK automatically sets world frame origin at the first position where
        # motion tracking starts, so no additional transformation is needed.
        # The translation is already relative to the boot origin.
        
        return float(translation[0]), float(translation[1]), float(yaw)

    def set_base_velocity(
        self, 
        lin_vel_x: float, 
        lin_vel_y: float, 
        ang_vel: float, 
        timeout_sec: float = 5.0
    ) -> None:
        """
        Set the base velocity for the Go1 robot.
        
        Note: Unitree SDK requires continuous command sending (typically every 2ms).
        This function sends the command once. For continuous control, call this
        function repeatedly in your control loop with the desired velocities.
        
        Args:
            lin_vel_x: Linear velocity in x direction (forward) in m/s
                       Range for trot: -1.1 to 1.5 m/s
                       Range for run: -2.5 to 3.5 m/s
            lin_vel_y: Linear velocity in y direction (sideways) in m/s
                       Range: -1.0 to 1.0 m/s
            ang_vel: Angular velocity (yaw speed) in rad/s
                     Range: -4.0 to 4.0 rad/s
            timeout_sec: Timeout in seconds (not used, kept for API compatibility)
        """
        if not UNITREE_SDK_AVAILABLE or self._unitree_udp is None:
            raise RuntimeError("Unitree SDK not available. Cannot control robot.")
        
        # Convert m/s to normalized [-1, 1] range (matching go1_move.py)
        # Max forward speed ~1.5 m/s, max sideways ~1.0 m/s
        max_fwd_vel = 1.5  # m/s
        max_side_vel = 1.0  # m/s
        
        # Normalize velocities to [-1, 1] range
        ux = np.clip(lin_vel_x / max_fwd_vel, -1.0, 1.0)
        uy = np.clip(lin_vel_y / max_side_vel, -1.0, 1.0)
        
        # For yaw, Unitree SDK expects normalized value too
        # Max angular velocity ~4.0 rad/s
        max_ang_vel = 4.0  # rad/s
        yaw_normalized = np.clip(ang_vel / max_ang_vel, -1.0, 1.0)
        
        # Set mode to 2 (target velocity walking)
        self._unitree_cmd.mode = 2
        self._unitree_cmd.gaitType = 1  # 1: trot, 2: trot running
        self._unitree_cmd.velocity = [ux, uy]  # Normalized [-1, 1]
        self._unitree_cmd.yawSpeed = yaw_normalized  # Normalized
        self._unitree_cmd.bodyHeight = 0.0
        
        # Send command (matching go1_move.py style - direct send)
        self._unitree_udp.SetSend(self._unitree_cmd)
        self._unitree_udp.Send()

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
        """
        Set base position (Spot-like API).
        
        This uses velocity control to simulate position control, similar to perception-guarantees.
        
        Args:
            x_pos: Target x position (m) or displacement if relative=True
            y_pos: Target y position (m) or displacement if relative=True
            yaw: Target yaw (rad) or rotation if relative=True
            end_time: Not used, kept for API compatibility
            relative: If True, treat x_pos/y_pos/yaw as relative to current pose
            max_fwd_vel: Maximum forward velocity (m/s)
            max_hor_vel: Maximum horizontal velocity (m/s)
            max_ang_vel: Maximum angular velocity (rad/s)
            disable_obstacle_avoidance: Not used, kept for API compatibility
            blocking: If True, wait until command completes (not recommended for Go1)
        
        Returns:
            int: Command ID for tracking
        """
        if not UNITREE_SDK_AVAILABLE or self._unitree_udp is None:
            raise RuntimeError("Unitree SDK not available. Cannot control robot.")
        
        # Get current pose
        curr_x, curr_y, curr_yaw = self.get_xy_yaw()
        
        # Calculate target pose
        if relative:
            target_x = curr_x + x_pos
            target_y = curr_y + y_pos
            target_yaw = curr_yaw + yaw
        else:
            target_x = x_pos
            target_y = y_pos
            target_yaw = yaw
        
        # Normalize yaw to [-pi, pi]
        target_yaw = self._wrap_angle(target_yaw)
        
        # Generate command ID
        with self._cmd_lock:
            self._cmd_counter += 1
            cmd_id = self._cmd_counter
            
            # Store command
            cmd = PositionCommand(
                cmd_id=cmd_id,
                target_x=target_x,
                target_y=target_y,
                target_yaw=target_yaw,
                relative=relative,
                start_time=time.time(),
                start_x=curr_x,
                start_y=curr_y,
                start_yaw=curr_yaw,
                max_fwd_vel=max_fwd_vel,
                max_hor_vel=max_hor_vel,
                max_ang_vel=max_ang_vel,
                status=CommandStatus.STATUS_PROCESSING,
                timeout=end_time if end_time < 100 else 30.0,  # Use reasonable timeout
            )
            self._active_commands[cmd_id] = cmd
        
        # Start control thread if not running
        if self._control_thread is None or not self._control_thread.is_alive():
            self._stop_control_thread = False
            self._control_thread = threading.Thread(target=self._position_control_loop, daemon=True)
            self._control_thread.start()
        
        if blocking:
            # Wait for command to complete (not recommended)
            while cmd.status == CommandStatus.STATUS_PROCESSING:
                time.sleep(0.1)
                if time.time() - cmd.start_time > cmd.timeout:
                    break
        
        return cmd_id

    def _position_control_loop(self):
        """Background thread for position control using velocity commands."""
        control_dt = 0.02  # 50 Hz control loop (Unitree SDK requirement)
        
        while not self._stop_control_thread:
            with self._cmd_lock:
                active_cmds = list(self._active_commands.values())
            
            if not active_cmds:
                time.sleep(control_dt)
                continue
            
            # Get current pose
            try:
                curr_x, curr_y, curr_yaw = self.get_xy_yaw()
            except Exception as e:
                print(f"Warning: Failed to get pose in control loop: {e}")
                time.sleep(control_dt)
                continue
            
            # Process each active command
            for cmd in active_cmds:
                # Check timeout
                if time.time() - cmd.start_time > cmd.timeout:
                    cmd.status = CommandStatus.STATUS_COMMAND_TIMED_OUT
                    with self._cmd_lock:
                        if cmd.cmd_id in self._active_commands:
                            del self._active_commands[cmd.cmd_id]
                    self._send_stop_command()
                    continue
                
                # Calculate errors
                dx = cmd.target_x - curr_x
                dy = cmd.target_y - curr_y
                dyaw = self._wrap_angle(cmd.target_yaw - curr_yaw)
                
                # Check if command is complete
                pos_error = np.sqrt(dx**2 + dy**2)
                yaw_error = abs(dyaw)
                
                pos_tolerance = 0.05  # 5cm tolerance
                yaw_tolerance = np.deg2rad(5)  # 5 degrees tolerance
                
                if pos_error < pos_tolerance and yaw_error < yaw_tolerance:
                    cmd.status = CommandStatus.STATUS_COMMAND_COMPLETE
                    with self._cmd_lock:
                        if cmd.cmd_id in self._active_commands:
                            del self._active_commands[cmd.cmd_id]
                    self._send_stop_command()
                    continue
                
                # Calculate desired velocities using perception-guarantees logic
                # Normalize actions
                max_dist = 1.0  # Normalization factor
                ux = np.clip(dx / max_dist, -0.8, 0.8)
                uy = np.clip(dy / max_dist, -0.8, 0.8)
                
                # Dead zone handling (from perception-guarantees)
                if 0.1 < np.abs(ux) < 0.2:
                    ux = np.sign(ux) * 0.15
                if 0.1 < np.abs(uy) < 0.2:
                    uy = np.sign(uy) * 0.15
                
                # Convert to velocities
                lin_vel_x = ux * cmd.max_fwd_vel
                lin_vel_y = uy * cmd.max_hor_vel
                
                # Yaw correction (from perception-guarantees)
                yaw_cmd = -dyaw / 2.0
                yaw_cmd = np.clip(yaw_cmd, -cmd.max_ang_vel, cmd.max_ang_vel)
                
                # Send velocity command
                try:
                    self.set_base_velocity(lin_vel_x, lin_vel_y, yaw_cmd)
                except Exception as e:
                    print(f"Warning: Failed to send velocity command: {e}")
            
            time.sleep(control_dt)

    def _send_stop_command(self):
        """Send stop command to robot."""
        if not UNITREE_SDK_AVAILABLE or self._unitree_udp is None:
            return
        
        try:
            # Set mode to 0 (idle/stand)
            self._unitree_cmd.mode = 0
            self._unitree_cmd.velocity[0] = 0.0
            self._unitree_cmd.velocity[1] = 0.0
            self._unitree_cmd.yawSpeed = 0.0
            
            self._unitree_udp.Recv()
            self._unitree_udp.GetRecv(self._unitree_state)
            
            if hasattr(self._unitree_safe, 'PowerProtect'):
                self._unitree_safe.PowerProtect(self._unitree_cmd, self._unitree_state, 1)
            
            self._unitree_udp.SetSend(self._unitree_cmd)
            self._unitree_udp.Send()
        except Exception as e:
            print(f"Warning: Failed to send stop command: {e}")

    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def get_cmd_feedback(self, cmd_id: int) -> CommandFeedback:
        """
        Get command feedback (Spot-like API).
        
        Args:
            cmd_id: Command ID returned by set_base_position()
        
        Returns:
            CommandFeedback: Feedback object with status
        """
        with self._cmd_lock:
            if cmd_id not in self._active_commands:
                # Command completed or doesn't exist
                return CommandFeedback(status=CommandStatus.STATUS_COMMAND_COMPLETE.value)
            
            cmd = self._active_commands[cmd_id]
            status = cmd.status.value
        
        return CommandFeedback(status=status)

    def get_transform(
        self,
    ) -> np.ndarray:
        """
        Get the transformation matrix from the camera frame (LEFT) to world frame.
        
        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        translation, orientation = self._update_pose_world()
        return self._pose_to_tf_matrix(translation, orientation)

    def get_image_responses(self, sources: List[str], quality: Optional[int] = None) -> List[ImageResponse]:
        """
        Get image responses for the specified camera sources.
        For ZED, we typically only have one camera (LEFT), but this mimics Spot API.
        
        Args:
            sources: List of camera source names
            quality: Image quality (not used for ZED)
            
        Returns:
            List of ImageResponse objects
        """
        # Grab a frame
        if self._zed.grab(self._runtime_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to grab image from ZED")
        
        responses = []
        for source in sources:
            # For ZED, we retrieve LEFT view
            # In the future, you might want to support RIGHT, DEPTH, etc.
            if "left" in source.lower() or "front" in source.lower():
                self._zed.retrieve_image(self._image, sl.VIEW.LEFT)
                image_np = self._image.get_data()
                image_np = image_np[:, :, :3].copy()  # BGR to RGB if needed, or keep BGR
                
                # Create mock response structure
                focal_length = FocalLength(x=self._fx, y=self._fy)
                pinhole = PinholeIntrinsics(focal_length=focal_length)
                source_obj = Source(name=source, pinhole=pinhole)
                
                image_data = ImageData(
                    data=b"",  # Not used, we have numpy array directly
                    pixel_format=0,  # RGB
                    format=0,  # RAW
                    rows=image_np.shape[0],
                    cols=image_np.shape[1]
                )
                
                shot = Shot(
                    image=image_np,
                    frame_name_image_sensor=source,
                    transforms_snapshot=None
                )
                
                response = ImageResponse(source=source_obj, shot=shot)
                responses.append(response)
            else:
                raise ValueError(f"Unsupported camera source: {source}")
        
        return responses

    def get_camera_data(self, srcs: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dict that maps each camera id to its image, focal lengths, and
        transform matrix (from camera to global frame).
        
        Args:
            srcs: List of camera source names
            
        Returns:
            Dict mapping camera ids to dictionaries containing:
                - image: np.ndarray
                - fx: float
                - fy: float
                - tf_camera_to_global: np.ndarray (4x4 transformation matrix)
        """
        imgs: Dict[str, Dict[str, Any]] = {}
        
        # Grab a frame
        if self._zed.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to grab image from ZED")
        
        # Get current pose for transform
        translation, orientation = self._update_pose_world()
        tf_camera_to_global = self._pose_to_tf_matrix(translation, orientation)
        
        for src in srcs:
            if "left" in src.lower() or "front" in src.lower():
                # Retrieve RGB image
                self._zed.retrieve_image(self._image, sl.VIEW.LEFT)
                image_np = self._image.get_data()
                image_np = image_np[:, :, :3].copy()
                
                imgs[src] = {
                    "image": image_np,
                    "fx": self._fx,
                    "fy": self._fy,
                    "tf_camera_to_global": tf_camera_to_global.copy(),
                }
            else:
                raise ValueError(f"Unsupported camera source: {src}")
        
        return imgs

    def get_depth(self) -> np.ndarray:
        """
        Get the depth map from the ZED camera.
        
        Returns:
            np.ndarray: Depth map in meters (H, W)
        """
        if self._zed.grab(self._runtime_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to grab depth map from ZED")
        self._zed.retrieve_measure(self._depth, sl.MEASURE.DEPTH)
        
        depth_np = self._depth.get_data()
        return depth_np.astype(np.float32)
