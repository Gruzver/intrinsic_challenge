"""
Port pose estimator via solvePnP.

For each camera that has a valid YOLO detection (KP0-3 keypoints),
runs cv2.solvePnP to recover the full 6-DOF pose of the port entrance.

Port frame convention (matches the _link_entrance TF frame):
  +X  right  (width direction)
  +Y  up     (height direction)
  +Z  outward toward robot  (normal to entrance face)

3D model points are the 4 entrance corners in this frame at Z=0:
  KP0 TL = (-w/2, +h/2, 0)
  KP1 TR = (+w/2, +h/2, 0)
  KP2 BR = (+w/2, -h/2, 0)
  KP3 BL = (-w/2, -h/2, 0)

Returns PortPose: position in base_link + rotation matrix R_base_port.
  R_base_port[:, 2]  =  port outward normal in base_link (approach direction).
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer
from transforms3d.quaternions import quat2mat

from my_vision_policy.vision.yolo_detector import YoloPoseDetector
from my_vision_policy.vision.localizer import FRAME_LEFT, FRAME_CENTER, FRAME_RIGHT


# ── 3D model points (entrance corners, port frame, meters) ────────────────────

_PORT_MODEL: dict[str, np.ndarray] = {}
for _ptype, _w, _h in [("sfp", 0.0135, 0.0088), ("sc", 0.0258, 0.0108)]:
    _w2, _h2 = _w / 2, _h / 2
    _PORT_MODEL[_ptype] = np.array([
        [-_w2,  _h2, 0.0],   # KP0 TL
        [ _w2,  _h2, 0.0],   # KP1 TR
        [ _w2, -_h2, 0.0],   # KP2 BR
        [-_w2, -_h2, 0.0],   # KP3 BL
    ], dtype=np.float64)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PortPose:
    position:              np.ndarray   # shape (3,) in base_link [m]
    R:                     np.ndarray   # 3×3 rotation, port frame → base_link
    reprojection_err:      float        # mean reprojection error [px]
    inter_camera_spread_mm: float = float('inf')  # max pairwise distance between camera estimates [mm]; inf if only 1 camera


# ── Estimator ─────────────────────────────────────────────────────────────────

# Visibility threshold: YOLO outputs 0 (occluded) or 2 (visible)
_VIS_MIN = 0.3

# Maximum accepted reprojection error per camera [px]
_MAX_REPROJ_PX = 15.0

# Maximum plausible distance from camera to port [m]
_MAX_DEPTH_M = 0.8


class PortPoseEstimator:
    """Estimates full 6-DOF port pose from YOLO-Pose keypoints via solvePnP.

    Works per-camera and then fuses all valid estimates (weighted by
    reprojection error) into a single pose in base_link.

    port_number: for SFP, 0 selects the detection with max base_link X
                 (closer to robot), 1 selects min base_link X.
                 None (default) picks the highest-confidence detection.
    """

    def __init__(self, plug_type: str, port_number: Optional[int] = None):
        if plug_type not in _PORT_MODEL:
            raise ValueError(f"Unknown plug_type '{plug_type}'")
        self._model_pts   = _PORT_MODEL[plug_type]  # shape (4, 3)
        self._port_number = port_number

    # ── Per-camera estimate ───────────────────────────────────────────────────

    def estimate_single(
        self,
        dets: list,    # list[Detection] sorted by confidence desc (all target-class hits)
        camera_info: CameraInfo,
        tf_buffer: Buffer,
        cam_frame: str,
    ) -> Optional[PortPose]:
        """solvePnP from KP0-3 for each detection in dets.

        When port_number is set (SFP multi-port case):
          port_0 → candidate with max base_link X (closer to robot)
          port_1 → candidate with min base_link X
        Otherwise returns the first valid solvePnP result (highest confidence).
        """
        if not dets:
            return None

        K = np.array(camera_info.k, dtype=np.float64).reshape(3, 3)
        D = np.array(camera_info.d, dtype=np.float64)

        try:
            tf_stamped = tf_buffer.lookup_transform("base_link", cam_frame, Time())
        except Exception:
            return None
        T_base_cam = _tf_to_4x4(tf_stamped.transform)
        R_base_cam = T_base_cam[:3, :3]

        # Run solvePnP on every detection; collect valid PortPoses
        candidates: list[PortPose] = []
        for det in dets:
            pose = self._solve_single(det, K, D, T_base_cam, R_base_cam)
            if pose is not None:
                candidates.append(pose)

        if not candidates:
            return None

        # Single result or no port_number constraint → highest-confidence (first valid)
        if self._port_number is None or len(candidates) == 1:
            return candidates[0]

        # SFP multi-port selection by base_link X:
        #   port_0 = most positive X (closer to robot)
        #   port_1 = least positive X (further from robot)
        if self._port_number == 0:
            return max(candidates, key=lambda p: p.position[0])
        else:
            return min(candidates, key=lambda p: p.position[0])

    def _solve_single(
        self,
        det,
        K: np.ndarray,
        D: np.ndarray,
        T_base_cam: np.ndarray,
        R_base_cam: np.ndarray,
    ) -> Optional[PortPose]:
        """Run solvePnP for one detection. Returns PortPose or None."""
        if det is None or det.keypoints is None or det.keypoints_vis is None:
            return None

        kp_entrance  = det.keypoints[:4]
        vis_entrance = det.keypoints_vis[:4]
        if np.any(vis_entrance < _VIS_MIN):
            return None

        pts_2d = kp_entrance.astype(np.float64)
        pts_3d = self._model_pts

        retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            pts_3d, pts_2d, K, D, flags=cv2.SOLVEPNP_IPPE,
        )
        if retval == 0 or rvecs is None:
            return None

        # Pick IPPE solution with most-downward port Z in base (ports open upward)
        best_idx, best_z = 0, float('inf')
        for i in range(retval):
            R_i, _ = cv2.Rodrigues(rvecs[i])
            z_base = float((R_base_cam @ R_i[:, 2])[2])
            if z_base < best_z:
                best_z, best_idx = z_base, i

        rvec, tvec = rvecs[best_idx], tvecs[best_idx]

        depth = float(np.linalg.norm(tvec))
        if depth > _MAX_DEPTH_M or depth < 0.01:
            return None

        R_cam_port, _ = cv2.Rodrigues(rvec)
        proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, D)
        err = float(np.mean(np.linalg.norm(pts_2d - proj.reshape(-1, 2), axis=1)))
        if err > _MAX_REPROJ_PX:
            return None

        T_cam_port = np.eye(4)
        T_cam_port[:3, :3] = R_cam_port
        T_cam_port[:3,  3] = tvec.ravel()
        T_base_port = T_base_cam @ T_cam_port
        return PortPose(
            position=T_base_port[:3, 3].copy(),
            R=T_base_port[:3, :3].copy(),
            reprojection_err=err,
        )

    # ── Multi-camera fused estimate ───────────────────────────────────────────

    def estimate_fused(
        self,
        detections: dict,    # frame → list[Detection]
        camera_infos: dict,  # frame → CameraInfo
        tf_buffer: Buffer,
    ) -> Optional[PortPose]:
        """Estimate port pose from all cameras and return weighted average.

        Position and rotation are averaged over valid cameras, weighted by
        inverse reprojection error.  Returns None if no camera succeeds.
        """
        estimates: list[PortPose] = []
        for frame in (FRAME_LEFT, FRAME_CENTER, FRAME_RIGHT):
            dets = detections.get(frame)
            info = camera_infos.get(frame)
            if not dets or info is None:
                continue
            pose = self.estimate_single(dets, info, tf_buffer, frame)
            if pose is not None:
                estimates.append(pose)

        if not estimates:
            return None

        if len(estimates) == 1:
            return estimates[0]

        # Weighted average: weight = 1 / reprojection_err
        weights = np.array([1.0 / max(e.reprojection_err, 0.1) for e in estimates])
        weights /= weights.sum()

        pos_avg = sum(w * e.position for w, e in zip(weights, estimates))

        # Rotation: weighted average via SVD (Procrustes)
        R_avg = _average_rotations([e.R for e in estimates], weights)
        mean_err = float(sum(w * e.reprojection_err for w, e in zip(weights, estimates)))

        # Inter-camera spread: max pairwise distance between per-camera position estimates
        positions = [e.position for e in estimates]
        spread_mm = float(max(
            np.linalg.norm(positions[i] - positions[j]) * 1000.0
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
        ))

        return PortPose(position=pos_avg, R=R_avg, reprojection_err=mean_err,
                        inter_camera_spread_mm=spread_mm)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tf_to_4x4(transform) -> np.ndarray:
    r = transform.rotation
    t = transform.translation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat2mat([r.w, r.x, r.y, r.z])
    T[:3,  3] = [t.x, t.y, t.z]
    return T


def _average_rotations(Rs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted geodesic average of rotation matrices via SVD."""
    M = sum(w * R for w, R in zip(weights, Rs))
    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg
