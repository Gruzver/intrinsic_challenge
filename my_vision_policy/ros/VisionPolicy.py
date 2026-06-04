"""
Vision-based cable insertion policy.

Pipeline
────────
Phase A  (implemented):
  1. Go to home joint position.
  2. Detect port via YOLO-Pose in all 3 cameras.
  3. solvePnP → full 6-DOF port pose (position + orientation) in base_link.
  4. Compute target TCP pose: connector tip aligned with port entrance (no collision).
  5. Move smoothly to standoff pose (5 cm in front of port, fully aligned).

Phase B  (implemented):
  IBVS descent: detect port in centre camera each step, back-project pixel
  to 3D, correct TCP XY to align connector tip with port, descend 2 mm/step.

Phase C  (implemented):
  Expanding spiral along port axis guided by wrist wrench until connector snaps in.
"""

import math
import os
import re
from typing import Optional

import cv2
import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from rclpy.time import Time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from transforms3d.quaternions import quat2mat

from my_vision_policy.vision.trajectory import interpolate_pose
from my_vision_policy.vision.yolo_detector import YoloPoseDetector
from my_vision_policy.vision.port_pose import PortPoseEstimator, PortPose, _average_rotations
from my_vision_policy.vision.connector import compute_approach_pose
from my_vision_policy.vision.localizer import (
    StereoLocalizer,
    FRAME_LEFT, FRAME_CENTER, FRAME_RIGHT,
)


# ── Constants ─────────────────────────────────────────────────────────────────

HOME_JOINTS = [-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110]

# Approach: responsive but smooth — good Tier-2 jerk score
APPROACH_STIFFNESS = [120.0, 120.0, 120.0, 60.0, 60.0, 60.0]
APPROACH_DAMPING   = [ 60.0,  60.0,  60.0, 30.0, 30.0, 30.0]

# Insertion stiffness — SFP works with 200 N/m (avoids 21 N snap spike at tier_2).
# SC needs 300 N/m: arm is highly extended, Jacobian unfavorable → needs more force.
INSERT_STIFFNESS_SFP = [200.0, 200.0, 200.0,  80.0,  80.0,  80.0]
INSERT_DAMPING_SFP   = [ 65.0,  65.0,  65.0,  25.0,  25.0,  25.0]
INSERT_STIFFNESS_SC  = [300.0, 300.0, 300.0, 100.0, 100.0, 100.0]
INSERT_DAMPING_SC    = [ 80.0,  80.0,  80.0,  30.0,  30.0,  30.0]
# Default alias used by descent phase (same for both — descent has no snap spike)
INSERT_STIFFNESS = INSERT_STIFFNESS_SC
INSERT_DAMPING   = INSERT_DAMPING_SC

CONTACT_FZ_N  = 3.0    # Fz to detect contact
MAX_FZ_N      = 15.0   # Safety cutoff (penalty triggers at >20 N for >1 s)
INSERTED_FZ_N = 1.0    # Fz drops below this after contact → connector in

STEP_S = 0.05          # 20 Hz control loop

# Debug: save annotated detection images to this folder for inspection
DEBUG_DETECT_DIR = "/tmp/aic_detect"

# Phase A: how far in front of the port to stop (pre-connection standoff)
STANDOFF_M = 0.05      # 5 cm

# Maximum retries for stereo detection per trial
MAX_DETECT_ATTEMPTS = 3

# Continuous approach: solvePnP every N steps while arm moves to standoff
APPROACH_DETECT_EVERY = 4     # steps between solvePnP calls (~200 ms at 20 Hz)
APPROACH_CONVERGE_M   = 0.012 # stop approach when TCP is within 12 mm of standoff target
APPROACH_MAX_STEPS    = 300   # hard limit 15 s
EMA_ALPHA_FAR         = 0.25  # conservative smoothing when > 10 cm away
EMA_ALPHA_CLOSE       = 0.50  # more responsive when ≤ 10 cm away

# Continuous descent: reduce standoff from STANDOFF_M → 0 while refining estimate
DESCENT_DETECT_EVERY = 4      # steps between solvePnP during descent
DESCENT_M            = 0.002  # 2 mm standoff reduction per step → ~40 mm/s
DESCENT_MAX_STEPS    = 300    # hard limit 15 s

# SC scan: when initial spread is too large to trust, move to a closer scan pose first
APPROACH_SPREAD_UNRELIABLE_MM = 100.0

# Outlier rejection: ignore solvePnP update if position jumps more than this from current EMA
APPROACH_MAX_JUMP_M = 0.08   # 80 mm — catches bad solutions without filtering valid updates

# Descent EMA: use lower alpha (smoother) and only update XY (Z controlled by standoff)
DESCENT_EMA_ALPHA = 0.25

# Pose refinement at standoff: accumulate N static shots to reduce solvePnP noise
REFINE_SHOTS         = 20     # measurements while arm is stationary at standoff
REFINE_MAX_REPROJ_PX = 10.0   # tighter reproj filter vs global 15 px

# Phase C — spiral insertion
INSERT_DEPTH_STEP_M  = 0.001    # 1 mm/step along insertion axis → ~20 mm/s
SPIRAL_ANGLE_STEP    = 0.4      # ~23° per step (smooth spiral)
SPIRAL_R_INCREMENT   = 0.00025  # radius grows 0.25 mm/step
SPIRAL_MAX_R_M       = 0.008    # max 8 mm radius (covers expected ~12 mm XY error)
INSERT_MAX_DEPTH_SFP = 0.055    # SFP port depth ~45.8 mm + 9 mm margin
INSERT_MAX_DEPTH_SC  = 0.040    # SC  port depth ~15.6 mm + plug gap ~7 mm + margin
INSERT_MAX_STEPS     = 500      # 25 s hard limit

# Pre-defined scan pose for SC port (base_link): ~20 cm above the SC port entrance (GT Z≈0.03 m)
# TCP orientation matches home: w=1 (identity), arm approaches from above
SC_SCAN_POSE = Pose(
    position=Point(x=-0.49, y=0.29, z=0.23),
    orientation=Quaternion(w=0.0, x=1.0, y=0.0, z=0.0),
)


# ── Policy ────────────────────────────────────────────────────────────────────

class VisionPolicy(Policy):
    """Cable insertion: vision (Phase A) + solvePnP descent (B) + spiral insertion (C)."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, parent_node)
        self._detector = YoloPoseDetector("sfp")  # preload model during configure; plug type set per trial
        self._detect_count = 0
        os.makedirs(DEBUG_DETECT_DIR, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # T1 — Movement helpers  (IMPLEMENTED)
    # ═══════════════════════════════════════════════════════════════════════════

    def _go_home(self, move_robot: MoveRobotCallback, duration_sec: float = 4.0) -> None:
        """Drive all joints to the home configuration."""
        cmd = JointMotionUpdate(
            target_stiffness=[80.0, 80.0, 80.0, 40.0, 40.0, 40.0],
            target_damping  =[50.0, 50.0, 50.0, 25.0, 25.0, 25.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION
            ),
        )
        for _ in range(max(1, int(duration_sec / STEP_S))):
            cmd.target_state.positions = HOME_JOINTS
            move_robot(joint_motion_update=cmd)
            self.sleep_for(STEP_S)

    def _move_to_pose(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        target: Pose,
        duration_sec: float = 3.0,
        stiffness: list = None,
        damping: list = None,
    ) -> None:
        """Smooth Cartesian move from current TCP to target using smoothstep."""
        start = get_observation().controller_state.tcp_pose
        steps = max(1, int(duration_sec / STEP_S))
        for i in range(steps + 1):
            self.set_pose_target(
                move_robot=move_robot,
                pose=interpolate_pose(start, target, i / steps),
                stiffness=stiffness or APPROACH_STIFFNESS,
                damping=damping or APPROACH_DAMPING,
            )
            self.sleep_for(STEP_S)

    def _hold_pose(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        duration_sec: float,
        stiffness: list = None,
        damping: list = None,
    ) -> None:
        """Hold a Cartesian pose for duration_sec (keep sending at 20 Hz)."""
        for _ in range(max(1, int(duration_sec / STEP_S))):
            self.set_pose_target(
                move_robot=move_robot,
                pose=pose,
                stiffness=stiffness or APPROACH_STIFFNESS,
                damping=damping or APPROACH_DAMPING,
            )
            self.sleep_for(STEP_S)

    def _log_state(self, get_observation: GetObservationCallback, label: str = "") -> None:
        obs = get_observation()
        if obs is None:
            return
        p = obs.controller_state.tcp_pose.position
        f = obs.wrist_wrench.wrench.force
        prefix = f"[{label}] " if label else ""
        self.get_logger().info(
            f"{prefix}TCP=({p.x:.4f},{p.y:.4f},{p.z:.4f})  "
            f"F=({f.x:.2f},{f.y:.2f},{f.z:.2f}) N"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase A — YOLO detection + solvePnP 6-DOF pose  (IMPLEMENTED)
    # ═══════════════════════════════════════════════════════════════════════════

    def _detect_port_pose(
        self,
        detector: YoloPoseDetector,
        estimator: PortPoseEstimator,
        get_observation: GetObservationCallback,
    ) -> Optional[PortPose]:
        """Run YOLO-Pose on all 3 cameras and fuse solvePnP estimates.

        Returns full 6-DOF PortPose in base_link, or None on failure.
        """
        obs = get_observation()
        if obs is None:
            return None

        img_left   = StereoLocalizer.ros_image_to_numpy(obs.left_image)
        img_center = StereoLocalizer.ros_image_to_numpy(obs.center_image)
        img_right  = StereoLocalizer.ros_image_to_numpy(obs.right_image)

        dets_left   = detector.detect(img_left)
        dets_center = detector.detect(img_center)
        dets_right  = detector.detect(img_right)

        def _fmt(dets):
            if not dets:
                return "FAIL"
            d = dets[0]
            return f"OK s={d.score:.2f} u={d.u} v={d.v}" + (f" +{len(dets)-1}" if len(dets) > 1 else "")
        self.get_logger().info(
            f"Detection — left:{_fmt(dets_left)} | center:{_fmt(dets_center)} | right:{_fmt(dets_right)}"
        )

        # Save annotated detection images — all boxes drawn, selected box in green
        self._detect_count += 1
        n = self._detect_count
        for cam_name, img_rgb, dets in [
            ("left",   img_left,   dets_left),
            ("center", img_center, dets_center),
            ("right",  img_right,  dets_right),
        ]:
            vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            for idx, det in enumerate(dets):
                color = (0, 255, 0) if idx == 0 else (0, 165, 255)
                x1 = det.u - det.w // 2
                y1 = det.v - det.h // 2
                x2 = det.u + det.w // 2
                y2 = det.v + det.h // 2
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"s={det.score:.2f}", (x1, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if idx == 0:
                    kp = getattr(det, "keypoints", None)
                    if kp is not None:
                        colors = [(255,0,0),(0,0,255),(0,128,255),(255,128,0)]
                        for ki in range(min(4, len(kp))):
                            cx, cy = int(kp[ki][0]), int(kp[ki][1])
                            cv2.circle(vis, (cx, cy), 4, colors[ki % 4], -1)
                            cv2.putText(vis, str(ki), (cx + 4, cy - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[ki % 4], 1)
            path = os.path.join(DEBUG_DETECT_DIR, f"{n:04d}_{cam_name}.png")
            cv2.imwrite(path, vis)

        detections = {
            FRAME_LEFT:   dets_left,
            FRAME_CENTER: dets_center,
            FRAME_RIGHT:  dets_right,
        }
        camera_infos = {
            FRAME_LEFT:   obs.left_camera_info,
            FRAME_CENTER: obs.center_camera_info,
            FRAME_RIGHT:  obs.right_camera_info,
        }

        port_pose = estimator.estimate_fused(detections, camera_infos, self._tf_buffer)

        if port_pose is None:
            self.get_logger().warn("solvePnP failed on all cameras")
            return None

        p  = port_pose.position
        nz = port_pose.R[:, 2]   # outward normal
        self.get_logger().info(
            f"Port pose: pos=({p[0]:.4f},{p[1]:.4f},{p[2]:.4f}) m | "
            f"normal=({nz[0]:.3f},{nz[1]:.3f},{nz[2]:.3f}) | "
            f"reproj={port_pose.reprojection_err:.1f} px"
        )
        return port_pose

    def _phase_a_localize(
        self,
        detector: YoloPoseDetector,
        estimator: PortPoseEstimator,
        get_observation: GetObservationCallback,
        send_feedback: SendFeedbackCallback,
    ) -> Optional[PortPose]:
        """Run detection + solvePnP up to MAX_DETECT_ATTEMPTS times."""
        for attempt in range(1, MAX_DETECT_ATTEMPTS + 1):
            send_feedback(f"phase_a solvePnP attempt {attempt}/{MAX_DETECT_ATTEMPTS}")
            port_pose = self._detect_port_pose(detector, estimator, get_observation)
            if port_pose is not None:
                return port_pose
            self.sleep_for(0.1)
        return None

    def _refine_port_pose(
        self,
        detector: YoloPoseDetector,
        estimator: PortPoseEstimator,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        current_pose: PortPose,
        plug_type: str,
    ) -> PortPose:
        """Take REFINE_SHOTS static measurements; return filtered weighted-average pose.

        Reduces random solvePnP noise by ~√N vs a single estimate.
        Sends hold commands at 20 Hz so the arm stays at standoff while measuring.
        Only accepts measurements with reprojection_err ≤ REFINE_MAX_REPROJ_PX.
        Falls back to current_pose if fewer than 3 valid shots are collected.
        """
        hold_target = compute_approach_pose(current_pose, plug_type, STANDOFF_M)
        poses = []
        for _ in range(REFINE_SHOTS):
            self.set_pose_target(
                move_robot=move_robot,
                pose=hold_target,
                stiffness=APPROACH_STIFFNESS,
                damping=APPROACH_DAMPING,
            )
            obs = get_observation()
            dets = {
                FRAME_LEFT:   detector.detect(StereoLocalizer.ros_image_to_numpy(obs.left_image)),
                FRAME_CENTER: detector.detect(StereoLocalizer.ros_image_to_numpy(obs.center_image)),
                FRAME_RIGHT:  detector.detect(StereoLocalizer.ros_image_to_numpy(obs.right_image)),
            }
            infos = {
                FRAME_LEFT:   obs.left_camera_info,
                FRAME_CENTER: obs.center_camera_info,
                FRAME_RIGHT:  obs.right_camera_info,
            }
            pose = estimator.estimate_fused(dets, infos, self._tf_buffer)
            if pose is not None and pose.reprojection_err <= REFINE_MAX_REPROJ_PX:
                poses.append(pose)
            self.sleep_for(STEP_S)

        if len(poses) < 3:
            self.get_logger().warn(
                f"[refine] only {len(poses)}/{REFINE_SHOTS} valid shots — keeping original estimate"
            )
            return current_pose

        weights = np.array([1.0 / max(p.reprojection_err, 0.5) for p in poses])
        weights /= weights.sum()
        pos_r = sum(w * p.position for w, p in zip(weights, poses))
        R_r   = _average_rotations([p.R for p in poses], weights)
        err_r = float(sum(w * p.reprojection_err for w, p in zip(weights, poses)))
        orig  = current_pose.position
        self.get_logger().info(
            f"[refine] {len(poses)}/{REFINE_SHOTS} shots accepted  reproj={err_r:.1f} px  "
            f"Δ=({(pos_r[0]-orig[0])*1e3:+.1f},{(pos_r[1]-orig[1])*1e3:+.1f},"
            f"{(pos_r[2]-orig[2])*1e3:+.1f}) mm"
        )
        return PortPose(position=pos_r, R=R_r, reprojection_err=err_r)

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase A — Continuous approach with live solvePnP refinement (IMPLEMENTED)
    # ═══════════════════════════════════════════════════════════════════════════

    def _continuous_approach(
        self,
        detector: YoloPoseDetector,
        estimator: PortPoseEstimator,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        plug_type: str,
        standoff_m: float = STANDOFF_M,
    ) -> Optional[tuple]:
        """Move arm to standoff while continuously refining port estimate via solvePnP.

        Runs set_pose_target at 20 Hz.  Every APPROACH_DETECT_EVERY steps (~200 ms),
        re-runs YOLO + solvePnP and blends the new estimate into a running EMA.
        The approach target updates smoothly — no discrete stops or waypoints.

        Returns (port_pose_ema, standoff_target) or None if initial detection fails.
        """
        # ── Initial detection ─────────────────────────────────────────────────
        port_pose = self._phase_a_localize(detector, estimator, get_observation, send_feedback)
        if port_pose is None:
            return None

        # SC from home often gives unreliable estimates — move to scan pose first
        if plug_type == "sc" and port_pose.inter_camera_spread_mm > APPROACH_SPREAD_UNRELIABLE_MM:
            self.get_logger().warn(
                f"[SC scan] spread={port_pose.inter_camera_spread_mm:.1f}mm — moving to scan pose"
            )
            send_feedback("sc: moving to scan position")
            obs_now = get_observation()
            tcp_now = obs_now.controller_state.tcp_pose
            tcp_arr = np.array([tcp_now.position.x, tcp_now.position.y, tcp_now.position.z])
            scan_arr = np.array([SC_SCAN_POSE.position.x, SC_SCAN_POSE.position.y, SC_SCAN_POSE.position.z])
            scan_dur = float(np.clip(np.linalg.norm(scan_arr - tcp_arr) * 12.0, 2.0, 5.0))
            self._move_to_pose(get_observation, move_robot, SC_SCAN_POSE, scan_dur)
            new_pose = self._phase_a_localize(detector, estimator, get_observation, send_feedback)
            if new_pose is not None:
                port_pose = new_pose
            else:
                self.get_logger().warn("[SC scan] re-detection failed — keeping rough estimate")

        # ── EMA state ─────────────────────────────────────────────────────────
        pos_ema = port_pose.position.copy()
        R_ema   = port_pose.R.copy()

        p0 = port_pose.position
        R0 = port_pose.R
        self.get_logger().info(
            f"[approach 0] pos=({p0[0]:.4f},{p0[1]:.4f},{p0[2]:.4f})  "
            f"spread={port_pose.inter_camera_spread_mm:.1f}mm  reproj={port_pose.reprojection_err:.1f}px\n"
            f"  port Z=({R0[0,2]:.3f},{R0[1,2]:.3f},{R0[2,2]:.3f})"
        )

        # ── Continuous approach loop ───────────────────────────────────────────
        for step in range(APPROACH_MAX_STEPS):

            # Re-detect every APPROACH_DETECT_EVERY steps (skip step 0 — just detected)
            if step > 0 and step % APPROACH_DETECT_EVERY == 0:
                new_pose = self._detect_port_pose(detector, estimator, get_observation)
                if new_pose is not None:
                    jump = float(np.linalg.norm(new_pose.position - pos_ema))
                    if jump > APPROACH_MAX_JUMP_M:
                        self.get_logger().warn(
                            f"[approach step {step}] solvePnP outlier rejected "
                            f"(jump={jump*1000:.1f}mm > {APPROACH_MAX_JUMP_M*1000:.0f}mm)"
                        )
                    else:
                        obs_now  = get_observation()
                        tcp_now  = obs_now.controller_state.tcp_pose
                        dist_est = float(np.linalg.norm(
                            pos_ema - np.array([tcp_now.position.x, tcp_now.position.y, tcp_now.position.z])
                        ))
                        alpha = EMA_ALPHA_CLOSE if dist_est < 0.10 else EMA_ALPHA_FAR
                        pos_ema = alpha * new_pose.position + (1.0 - alpha) * pos_ema
                        R_ema   = _average_rotations([new_pose.R, R_ema], np.array([alpha, 1.0 - alpha]))
                        if step % (APPROACH_DETECT_EVERY * 5) == 0:
                            self.get_logger().info(
                                f"[approach step {step}] "
                                f"pos=({pos_ema[0]:.4f},{pos_ema[1]:.4f},{pos_ema[2]:.4f})  "
                                f"reproj={new_pose.reprojection_err:.1f}px  "
                                f"spread={new_pose.inter_camera_spread_mm:.1f}mm  α={alpha:.2f}"
                            )

            # Current target from smoothed estimate
            port_pose_ema = PortPose(position=pos_ema, R=R_ema, reprojection_err=0)
            target = compute_approach_pose(port_pose_ema, plug_type, standoff_m)

            self.set_pose_target(
                move_robot=move_robot,
                pose=target,
                stiffness=APPROACH_STIFFNESS,
                damping=APPROACH_DAMPING,
            )
            self.sleep_for(STEP_S)

            # Convergence: TCP within APPROACH_CONVERGE_M of current target
            obs = get_observation()
            tcp = obs.controller_state.tcp_pose
            dist = float(np.linalg.norm([
                tcp.position.x - target.position.x,
                tcp.position.y - target.position.y,
                tcp.position.z - target.position.z,
            ]))
            if dist < APPROACH_CONVERGE_M:
                self.get_logger().info(
                    f"[approach step {step}] converged dist={dist * 1000:.1f} mm  "
                    f"pos_ema=({pos_ema[0]:.4f},{pos_ema[1]:.4f},{pos_ema[2]:.4f})"
                )
                break

        final_port_pose = PortPose(position=pos_ema, R=R_ema, reprojection_err=0)
        final_target    = compute_approach_pose(final_port_pose, plug_type, standoff_m)
        return final_port_pose, final_target

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase B — Continuous descent with live solvePnP refinement (IMPLEMENTED)
    # ═══════════════════════════════════════════════════════════════════════════

    def _continuous_descent(
        self,
        detector: YoloPoseDetector,
        estimator: PortPoseEstimator,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        port_pose: PortPose,
        plug_type: str,
    ) -> tuple:
        """Descend from standoff to port entrance while refining estimate via solvePnP.

        Reduces standoff from STANDOFF_M → 0 at DESCENT_M per step.  Every
        DESCENT_DETECT_EVERY steps, re-runs YOLO + solvePnP and blends the
        result into a running EMA.  compute_approach_pose with the updated
        estimate and shrinking standoff naturally corrects XY and orientation
        without any pixel-level IBVS logic.

        Stops on contact (Fz delta > CONTACT_FZ_N) or standoff ≤ 0.
        Returns final Pose reached.
        """
        obs         = get_observation()
        fz_baseline = float(obs.wrist_wrench.wrench.force.z)
        self.get_logger().info(f"[descent] Fz baseline={fz_baseline:.1f} N")

        pos_ema = port_pose.position.copy()
        R_ema   = port_pose.R.copy()
        standoff = STANDOFF_M

        for step in range(DESCENT_MAX_STEPS):
            obs = get_observation()
            fz  = obs.wrist_wrench.wrench.force.z

            # ── Contact stop ──────────────────────────────────────────────
            if (fz - fz_baseline) > CONTACT_FZ_N:
                self.get_logger().info(
                    f"[descent step {step}] contact Fz={fz:.1f} N "
                    f"(delta={fz - fz_baseline:+.1f} N) — stopping"
                )
                break

            if standoff <= 0:
                self.get_logger().info(f"[descent step {step}] standoff=0 — at port entrance")
                break

            # ── Refine estimate every DESCENT_DETECT_EVERY steps ─────────
            if step % DESCENT_DETECT_EVERY == 0:
                new_pose = self._detect_port_pose(detector, estimator, get_observation)
                if new_pose is not None:
                    jump = float(np.linalg.norm(new_pose.position - pos_ema))
                    if jump <= APPROACH_MAX_JUMP_M:
                        # SC: freeze XY — solvePnP drifts 16mm during close-range descent
                        # SFP: update XY normally with EMA
                        if plug_type != "sc":
                            blended = DESCENT_EMA_ALPHA * new_pose.position + (1.0 - DESCENT_EMA_ALPHA) * pos_ema
                            pos_ema[0] = blended[0]
                            pos_ema[1] = blended[1]
                        R_ema = _average_rotations(
                            [new_pose.R, R_ema],
                            np.array([DESCENT_EMA_ALPHA, 1.0 - DESCENT_EMA_ALPHA])
                        )

            # ── Descend: reduce standoff, send updated approach target ────
            standoff = max(0.0, standoff - DESCENT_M)
            port_pose_ema = PortPose(position=pos_ema, R=R_ema, reprojection_err=0)
            target = compute_approach_pose(port_pose_ema, plug_type, standoff)

            self.set_pose_target(
                move_robot=move_robot,
                pose=target,
                stiffness=INSERT_STIFFNESS,
                damping=INSERT_DAMPING,
            )
            self.sleep_for(STEP_S)

            if step % 20 == 0:
                self.get_logger().info(
                    f"[descent step {step}] standoff={standoff * 1000:.1f} mm  "
                    f"Fz={fz:.1f} N  delta={fz - fz_baseline:+.1f} N"
                )

        final_port_pose = PortPose(position=pos_ema, R=R_ema, reprojection_err=0)
        return final_port_pose, compute_approach_pose(final_port_pose, plug_type, standoff)

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase C — F/T guided insertion
    # ═══════════════════════════════════════════════════════════════════════════

    def _phase_c_insert(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        port_pose: PortPose,
        plug_type: str,
    ) -> bool:
        """Push connector into port with expanding spiral until snap or max depth.

        Moves the plug tip along the port axis (negative standoff = inside port)
        while adding a small spiral offset in the port XY plane to handle any
        remaining alignment error.

        Stops on:
          - Snap: Fz delta was > CONTACT_FZ_N, then drops below INSERTED_FZ_N
          - Max depth reached: INSERT_MAX_DEPTH_SFP / INSERT_MAX_DEPTH_SC
          - Safety: Fz delta > MAX_FZ_N
        """
        obs = get_observation()
        fz_baseline = float(obs.wrist_wrench.wrench.force.z)
        tcp0 = obs.controller_state.tcp_pose.position
        self.get_logger().info(
            f"[insert] Fz baseline={fz_baseline:.1f} N  "
            f"TCP=({tcp0.x:.4f},{tcp0.y:.4f},{tcp0.z:.4f})\n"
            f"  port_pos=({port_pose.position[0]:.4f},{port_pose.position[1]:.4f},{port_pose.position[2]:.4f})\n"
            f"  port_x=({port_pose.R[0,0]:.3f},{port_pose.R[1,0]:.3f},{port_pose.R[2,0]:.3f})  "
            f"port_y=({port_pose.R[0,1]:.3f},{port_pose.R[1,1]:.3f},{port_pose.R[2,1]:.3f})  "
            f"port_z=({port_pose.R[0,2]:.3f},{port_pose.R[1,2]:.3f},{port_pose.R[2,2]:.3f})"
        )

        port_x = port_pose.R[:, 0]
        port_y = port_pose.R[:, 1]
        insert_max = INSERT_MAX_DEPTH_SFP if plug_type == "sfp" else INSERT_MAX_DEPTH_SC
        ins_stiff  = INSERT_STIFFNESS_SFP if plug_type == "sfp" else INSERT_STIFFNESS_SC
        ins_damp   = INSERT_DAMPING_SFP   if plug_type == "sfp" else INSERT_DAMPING_SC

        insertion_depth = 0.0
        spiral_angle    = 0.0
        spiral_r        = 0.0
        was_in_contact  = False

        for step in range(INSERT_MAX_STEPS):
            obs      = get_observation()
            fz       = float(obs.wrist_wrench.wrench.force.z)
            fz_delta = fz - fz_baseline

            # Safety
            if fz_delta > MAX_FZ_N:
                self.get_logger().warn(
                    f"[insert step {step}] SAFETY fz_delta={fz_delta:.1f} N — abort"
                )
                return False

            # Contact/snap sign depends on TCP orientation at the port:
            #   SFP: TCP Z mostly downward → contact gives fz_delta > 0 (port resists)
            #   SC:  TCP Z oriented differently → contact gives fz_delta < 0 (port resists)
            if plug_type == "sc":
                in_contact   = fz_delta < -CONTACT_FZ_N
                snap_reached = was_in_contact and fz_delta > -INSERTED_FZ_N
            else:
                in_contact   = fz_delta > CONTACT_FZ_N
                snap_reached = was_in_contact and fz_delta < INSERTED_FZ_N

            if snap_reached:
                self.get_logger().info(
                    f"[insert step {step}] SNAP  fz_delta={fz_delta:.1f} N  "
                    f"depth={insertion_depth * 1000:.1f} mm"
                )
                return True

            was_in_contact = in_contact

            # Max depth fallback (scoring detects insertion geometrically)
            if insertion_depth >= insert_max:
                self.get_logger().info(
                    f"[insert] max depth {insert_max * 1000:.0f} mm reached"
                )
                return True

            # Advance depth and spiral
            insertion_depth += INSERT_DEPTH_STEP_M
            spiral_angle    += SPIRAL_ANGLE_STEP
            spiral_r         = min(SPIRAL_MAX_R_M, spiral_r + SPIRAL_R_INCREMENT)

            spiral_offset = spiral_r * (
                math.cos(spiral_angle) * port_x + math.sin(spiral_angle) * port_y
            )
            shifted_port = PortPose(
                position=port_pose.position + spiral_offset,
                R=port_pose.R,
                reprojection_err=0,
            )
            target = compute_approach_pose(shifted_port, plug_type, standoff_m=-insertion_depth)

            self.set_pose_target(
                move_robot=move_robot,
                pose=target,
                stiffness=ins_stiff,
                damping=ins_damp,
            )
            self.sleep_for(STEP_S)

            if step % 10 == 0:
                tcp = obs.controller_state.tcp_pose.position
                fx  = float(obs.wrist_wrench.wrench.force.x)
                fy  = float(obs.wrist_wrench.wrench.force.y)
                off = spiral_r * (
                    math.cos(spiral_angle) * port_x + math.sin(spiral_angle) * port_y
                )
                self.get_logger().info(
                    f"[insert step {step}] depth={insertion_depth * 1000:.1f} mm  "
                    f"r={spiral_r * 1000:.1f} mm  "
                    f"F=({fx:+.1f},{fy:+.1f},{fz_delta:+.1f}) N  "
                    f"TCP=({tcp.x:.4f},{tcp.y:.4f},{tcp.z:.4f})  "
                    f"offset=({off[0]*1000:+.1f},{off[1]*1000:+.1f},{off[2]*1000:+.1f}) mm"
                )

        self.get_logger().warn("[insert] max steps reached")
        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # Main pipeline
    # ═══════════════════════════════════════════════════════════════════════════

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(
            f"VisionPolicy plug={task.plug_type} port={task.port_name}"
        )

        # ── Initialise per-trial objects ───────────────────────────────────────
        self._detector.set_plug_type(task.plug_type)
        detector  = self._detector
        # Extract numeric port index from port_name (e.g. "sfp_port_0" → 0, "sc_port_base" → None)
        m = re.search(r'_(\d+)$', task.port_name)
        port_number = int(m.group(1)) if m else None
        estimator = PortPoseEstimator(task.plug_type, port_number=port_number)

        # ── Step 1: go home ───────────────────────────────────────────────────
        send_feedback("going home")
        self._log_state(get_observation, "start")
        self._go_home(move_robot)
        self._log_state(get_observation, "home")

        # ── Phase A: continuous approach to standoff ──────────────────────────
        send_feedback("phase_a: continuous approach")
        result = self._continuous_approach(
            detector, estimator, get_observation, move_robot, send_feedback, task.plug_type
        )
        if result is None:
            self.get_logger().error("Phase A failed: initial detection unsuccessful")
            return False

        port_pose, standoff_pose = result
        self._log_gt_comparison(task, port_pose)
        self._log_state(get_observation, "standoff")

        # ── Refine at standoff: N static measurements → averaged pose ─────────
        send_feedback("refining pose at standoff")
        port_pose = self._refine_port_pose(
            detector, estimator, get_observation, move_robot, port_pose, task.plug_type
        )
        self._log_gt_comparison(task, port_pose)

        # ── Phase B: continuous descent to port entrance ──────────────────────
        send_feedback("phase_b: continuous descent")
        port_pose_final, final_tcp_pose = self._continuous_descent(
            detector, estimator, get_observation, move_robot, port_pose, task.plug_type
        )

        self._log_state(get_observation, "at-port")

        # ── Phase C: spiral insertion ──────────────────────────────────────────
        send_feedback("phase_c: spiral insertion")
        return self._phase_c_insert(
            get_observation, move_robot, port_pose_final, task.plug_type
        )

    def _log_gt_comparison(self, task, port_pose) -> None:
        """Compare solvePnP estimate with ground truth TF (soft — skips if GT unavailable)."""
        gt_frame = f"task_board/{task.target_module_name}/{task.port_name}_link_entrance"
        try:
            gt_tf  = self._tf_buffer.lookup_transform("base_link", gt_frame, Time())
            t      = gt_tf.transform.translation
            q      = gt_tf.transform.rotation
            gt_pos = np.array([t.x, t.y, t.z])
            gt_R   = quat2mat([q.w, q.x, q.y, q.z])
        except TransformException:
            return  # not running with ground_truth:=true — skip silently

        gt_normal = gt_R[:, 2]

        est_pos    = port_pose.position
        est_R      = port_pose.R
        est_normal = est_R[:, 2]
        pos_err_mm    = np.linalg.norm(est_pos - gt_pos) * 1000.0
        angle_err_deg = float(np.degrees(np.arccos(
            float(np.clip(np.dot(est_normal, gt_normal), -1.0, 1.0))
        )))
        self.get_logger().info(
            f"[GT cmp] pos err={pos_err_mm:.1f} mm  normal err={angle_err_deg:.1f} deg\n"
            f"  GT  pos=({gt_pos[0]:.4f},{gt_pos[1]:.4f},{gt_pos[2]:.4f})  "
            f"X=({gt_R[0,0]:.3f},{gt_R[1,0]:.3f},{gt_R[2,0]:.3f})  "
            f"Y=({gt_R[0,1]:.3f},{gt_R[1,1]:.3f},{gt_R[2,1]:.3f})  "
            f"Z=({gt_R[0,2]:.3f},{gt_R[1,2]:.3f},{gt_R[2,2]:.3f})\n"
            f"  EST pos=({est_pos[0]:.4f},{est_pos[1]:.4f},{est_pos[2]:.4f})  "
            f"X=({est_R[0,0]:.3f},{est_R[1,0]:.3f},{est_R[2,0]:.3f})  "
            f"Y=({est_R[0,1]:.3f},{est_R[1,1]:.3f},{est_R[2,1]:.3f})  "
            f"Z=({est_R[0,2]:.3f},{est_R[1,2]:.3f},{est_R[2,2]:.3f})"
        )
