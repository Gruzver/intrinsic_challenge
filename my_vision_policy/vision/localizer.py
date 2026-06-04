"""
Stereo triangulator: converts two 2D port detections (left + right cameras)
into a 3D position in the base_link frame.

Math overview
─────────────
Each camera has a projection matrix  P = K · [R | t]  (3×4)
where R, t transform a point from world (base_link) to camera frame.

Given a point (u, v) in image space and projection matrix P:
    λ · [u, v, 1]ᵀ = P · [X, Y, Z, 1]ᵀ

With two cameras we get an over-determined linear system; cv2.triangulatePoints
solves it via DLT and returns the 3D point in homogeneous coordinates.

Camera frames used
──────────────────
"left_camera/optical"   — left camera optical frame
"right_camera/optical"  — right camera optical frame

TF lookup_transform("left_camera/optical", "base_link") gives T_cam←world,
i.e. how to express a base_link point in the camera frame:
    p_cam = R_cam←world · p_world + t_cam←world

This is exactly the extrinsic matrix [R | t] we need to build P.

Coordinate convention
─────────────────────
The optical frame follows the ROS/camera convention:
    +X right, +Y down, +Z forward (into the scene)
All K matrices from sensor_msgs/CameraInfo use this convention.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from geometry_msgs.msg import Point
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer
from transforms3d.quaternions import quat2mat

from my_vision_policy.vision.detector import Detection


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class LocalizationResult:
    position: Point        # 3D position in base_link [m]
    reprojection_err: float  # mean reprojection error [px] — lower is better


# ── Utility ───────────────────────────────────────────────────────────────────

def _ros_image_to_numpy(msg) -> np.ndarray:
    """Convert sensor_msgs/Image to H×W×3 uint8 numpy array (RGB)."""
    img = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = img.reshape(msg.height, msg.width, -1)
    # Gazebo publishes 'rgb8'; handle bgr8 just in case
    if msg.encoding == "bgr8":
        img = img[:, :, ::-1].copy()
    else:
        img = img.copy()
    return img


def _camera_info_to_K(info: CameraInfo) -> np.ndarray:
    """Extract 3×3 intrinsic matrix from CameraInfo."""
    return np.array(info.k, dtype=np.float64).reshape(3, 3)


def _build_projection(K: np.ndarray, transform) -> np.ndarray:
    """Build 3×4 projection matrix P = K · [R | t] for world→image.

    transform: geometry_msgs/Transform representing camera frame in world
               (i.e. result of lookup_transform("base_link", "camera/optical")).
               We invert it to get world-in-camera (extrinsic matrix).
    """
    r = transform.rotation
    t_vec = transform.translation

    # R_cw: rotation of camera frame expressed in world frame
    # quat2mat expects (w, x, y, z)
    R_cw = quat2mat([r.w, r.x, r.y, r.z])
    t_cw = np.array([t_vec.x, t_vec.y, t_vec.z])

    # Invert: world-to-camera
    R_wc = R_cw.T
    t_wc = -R_cw.T @ t_cw

    E = np.hstack([R_wc, t_wc.reshape(3, 1)])  # 3×4
    return (K @ E).astype(np.float64)


def _reprojection_error(P: np.ndarray, pt3d: np.ndarray, u: float, v: float) -> float:
    """Reproject 3D point through P and compute pixel distance from (u, v)."""
    h = P @ np.append(pt3d, 1.0)
    if abs(h[2]) < 1e-9:
        return 9999.0
    pu, pv = h[0] / h[2], h[1] / h[2]
    return float(np.sqrt((pu - u) ** 2 + (pv - v) ** 2))


# ── Localizer ─────────────────────────────────────────────────────────────────

# Optical frame names as published by the robot URDF
FRAME_LEFT   = "left_camera/optical"
FRAME_CENTER = "center_camera/optical"
FRAME_RIGHT  = "right_camera/optical"

# Camera pairs ordered by preference: largest baseline first for depth accuracy,
# but any pair that yields valid detections is tried.
CAMERA_PAIRS = [
    ("left+right",   FRAME_LEFT,   FRAME_RIGHT),   # baseline 0.186 m — most precise
    ("left+center",  FRAME_LEFT,   FRAME_CENTER),  # baseline 0.093 m
    ("center+right", FRAME_CENTER, FRAME_RIGHT),   # baseline 0.093 m
]


class StereoLocalizer:
    """Triangulates a 3D port position from detections in any camera pair.

    Tries all available pairs (left+right, left+center, center+right) and
    returns the result with the lowest reprojection error.  Falls back
    gracefully when a camera cannot see the port.
    """

    def __init__(self, tf_buffer: Buffer):
        self._tf_buffer = tf_buffer

    # ── Public API ────────────────────────────────────────────────────────────

    def localize_pair(
        self,
        det_a:   Detection,
        det_b:   Detection,
        info_a:  CameraInfo,
        info_b:  CameraInfo,
        frame_a: str,
        frame_b: str,
    ) -> Optional[LocalizationResult]:
        """Triangulate port centre from a single camera pair.

        Returns LocalizationResult (position in base_link + reprojection error),
        or None if TF is unavailable or result is geometrically inconsistent.
        """
        try:
            tf_a = self._tf_buffer.lookup_transform("base_link", frame_a, Time())
            tf_b = self._tf_buffer.lookup_transform("base_link", frame_b, Time())
        except Exception:
            return None

        K_a = _camera_info_to_K(info_a)
        K_b = _camera_info_to_K(info_b)

        P_a = _build_projection(K_a, tf_a.transform)
        P_b = _build_projection(K_b, tf_b.transform)

        pts_a = np.float32([[det_a.u], [det_a.v]])
        pts_b = np.float32([[det_b.u], [det_b.v]])

        X_hom = cv2.triangulatePoints(P_a, P_b, pts_a, pts_b)  # 4×N
        w = X_hom[3, 0]
        if abs(w) < 1e-9:
            return None

        pt3d = (X_hom[:3, 0] / w).astype(float)  # [X, Y, Z] in base_link

        # Sanity check: point must be in a plausible workspace region
        if np.linalg.norm(pt3d) > 1.5 or pt3d[2] < 0.5:
            return None

        err_a = _reprojection_error(P_a, pt3d, det_a.u, det_a.v)
        err_b = _reprojection_error(P_b, pt3d, det_b.u, det_b.v)
        mean_err = (err_a + err_b) / 2.0

        if mean_err > 30.0:
            return None

        return LocalizationResult(
            position=Point(x=float(pt3d[0]), y=float(pt3d[1]), z=float(pt3d[2])),
            reprojection_err=mean_err,
        )

    def localize_best(
        self,
        detections: dict,   # frame_name → Detection (or None)
        camera_infos: dict, # frame_name → CameraInfo
    ) -> tuple[Optional[LocalizationResult], str]:
        """Try all camera pairs and return (best_result, pair_name).

        detections:   {FRAME_LEFT: det_l, FRAME_CENTER: det_c, FRAME_RIGHT: det_r}
        camera_infos: {FRAME_LEFT: info_l, ...}
        Returns (None, "") if no pair succeeds.
        """
        best_result: Optional[LocalizationResult] = None
        best_name = ""

        for pair_name, frame_a, frame_b in CAMERA_PAIRS:
            det_a = detections.get(frame_a)
            det_b = detections.get(frame_b)
            if det_a is None or det_b is None:
                continue
            info_a = camera_infos.get(frame_a)
            info_b = camera_infos.get(frame_b)
            if info_a is None or info_b is None:
                continue

            result = self.localize_pair(det_a, det_b, info_a, info_b, frame_a, frame_b)
            if result is None:
                continue

            if best_result is None or result.reprojection_err < best_result.reprojection_err:
                best_result = result
                best_name = pair_name

        return best_result, best_name

    @staticmethod
    def ros_image_to_numpy(msg) -> np.ndarray:
        """Convenience wrapper — decodes a ROS Image message to numpy."""
        return _ros_image_to_numpy(msg)
