"""
DebugGroundTruth — logs all GT TF frames and runs CheatCode-style approach to standoff.

Run with:
  distrobox: /entrypoint.sh ground_truth:=true start_aic_engine:=true
  pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true -p policy:=my_vision_policy.ros.DebugGroundTruth
"""

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
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
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp


def _unpack(tf_stamped):
    """Return (pos np(3,), R np(3,3), q tuple(w,x,y,z)) from a stamped transform."""
    t = tf_stamped.transform.translation
    qr = tf_stamped.transform.rotation
    pos = np.array([t.x, t.y, t.z])
    q   = (qr.w, qr.x, qr.y, qr.z)
    R   = quat2mat(q)
    return pos, R, q


class DebugGroundTruth(Policy):
    """Log GT TF frames + CheatCode approach to standoff (no insertion)."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, parent_node)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _lookup(self, frame: str):
        """Lookup base_link → frame. Returns (pos, R, q) or (None, None, None)."""
        try:
            tf = self._tf_buffer.lookup_transform("base_link", frame, Time())
            return _unpack(tf)
        except TransformException as ex:
            self.get_logger().warn(f"TF '{frame}': {ex}")
            return None, None, None

    def _log_frame(self, label: str, pos, R):
        if pos is None:
            self.get_logger().info(f"  [{label}]: NOT AVAILABLE")
            return
        self.get_logger().info(
            f"  [{label}]\n"
            f"    pos = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})\n"
            f"    X   = ({R[0,0]:.3f}, {R[1,0]:.3f}, {R[2,0]:.3f})\n"
            f"    Y   = ({R[0,1]:.3f}, {R[1,1]:.3f}, {R[2,1]:.3f})\n"
            f"    Z   = ({R[0,2]:.3f}, {R[1,2]:.3f}, {R[2,2]:.3f})"
        )

    # ── main ──────────────────────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(
            f"DebugGroundTruth plug={task.plug_type} port={task.port_name} "
            f"module={task.target_module_name} cable={task.cable_name} plug={task.plug_name}"
        )

        # Frame names (same pattern as CheatCode)
        f_entrance = f"task_board/{task.target_module_name}/{task.port_name}_link_entrance"
        f_back     = f"task_board/{task.target_module_name}/{task.port_name}_link"
        f_plug     = f"{task.cable_name}/{task.plug_name}_link"
        f_tcp      = "gripper/tcp"

        # ── STEP 1: log all TF frames ─────────────────────────────────────────
        self.get_logger().info("=== STEP 1: TF FRAMES ===")

        pos_ent, R_ent, q_ent = self._lookup(f_entrance)
        self._log_frame("port _link_entrance", pos_ent, R_ent)

        pos_back, R_back, _ = self._lookup(f_back)
        self._log_frame("port _link (back face)", pos_back, R_back)

        pos_plug, R_plug, q_plug = self._lookup(f_plug)
        self._log_frame("plug tip (_link)", pos_plug, R_plug)

        pos_tcp_tf, R_tcp_tf, q_tcp_tf = self._lookup(f_tcp)
        self._log_frame("gripper/tcp (TF)", pos_tcp_tf, R_tcp_tf)

        obs = get_observation()
        tcp = obs.controller_state.tcp_pose
        self.get_logger().info(
            f"  [TCP obs] pos=({tcp.position.x:.4f},{tcp.position.y:.4f},{tcp.position.z:.4f})"
        )

        if pos_ent is None or q_plug is None or q_tcp_tf is None:
            self.get_logger().error(
                "Missing required TF frames. Run with ground_truth:=true"
            )
            return False

        # ── STEP 2: analyse approach geometry ────────────────────────────────
        self.get_logger().info("=== STEP 2: APPROACH GEOMETRY ===")

        if pos_back is not None:
            depth_vec = pos_ent - pos_back
            depth_m   = float(np.linalg.norm(depth_vec))
            self.get_logger().info(
                f"  entrance→back vec: ({depth_vec[0]:.4f},{depth_vec[1]:.4f},{depth_vec[2]:.4f}), "
                f"depth={depth_m*1000:.1f} mm"
            )

        # CheatCode approach: orientation diff between port and plug, applied to gripper
        q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
        q_diff     = quaternion_multiply(q_ent, q_plug_inv)
        q_tcp_target = quaternion_multiply(q_diff, q_tcp_tf)

        standoff_z = 0.10  # 10 cm above port in world Z (CheatCode style)
        target_pos = np.array([pos_ent[0], pos_ent[1], pos_ent[2] + standoff_z])
        # Adjust for plug-tip to TCP offset (same as CheatCode)
        if pos_tcp_tf is not None and pos_plug is not None:
            offset = pos_tcp_tf - pos_plug
            target_pos -= np.array([offset[0], offset[1], offset[2]])

        self.get_logger().info(
            f"  standoff target TCP: ({target_pos[0]:.4f},{target_pos[1]:.4f},{target_pos[2]:.4f})"
        )
        self.get_logger().info(
            f"  target orientation q: w={q_tcp_target[0]:.3f} x={q_tcp_target[1]:.3f} "
            f"y={q_tcp_target[2]:.3f} z={q_tcp_target[3]:.3f}"
        )

        # ── STEP 3: move to standoff (CheatCode-style interpolation) ─────────
        self.get_logger().info(f"=== STEP 3: moving to standoff ({standoff_z*100:.0f} cm above port) ===")
        send_feedback("moving to standoff")

        start_pos = np.array([tcp.position.x, tcp.position.y, tcp.position.z])
        q_start   = (tcp.orientation.w, tcp.orientation.x, tcp.orientation.y, tcp.orientation.z)

        steps = 100
        for i in range(steps + 1):
            frac = i / steps
            p = start_pos + frac * (target_pos - start_pos)
            q = quaternion_slerp(q_start, q_tcp_target, frac)
            self.set_pose_target(
                move_robot=move_robot,
                pose=Pose(
                    position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                    orientation=Quaternion(w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3])),
                ),
            )
            self.sleep_for(0.05)

        # ── STEP 4: log final state ───────────────────────────────────────────
        self.get_logger().info("=== STEP 4: FINAL STATE AT STANDOFF ===")
        obs = get_observation()
        tcp_f = obs.controller_state.tcp_pose
        f     = obs.wrist_wrench.wrench.force
        self.get_logger().info(
            f"  TCP final: pos=({tcp_f.position.x:.4f},{tcp_f.position.y:.4f},{tcp_f.position.z:.4f})\n"
            f"  Force:     F=({f.x:.2f},{f.y:.2f},{f.z:.2f}) N"
        )

        # Re-lookup plug and entrance to see actual alignment
        pos_ent2, _, _ = self._lookup(f_entrance)
        pos_plug2, _, _ = self._lookup(f_plug)
        if pos_ent2 is not None and pos_plug2 is not None:
            plug_to_port = pos_ent2 - pos_plug2
            self.get_logger().info(
                f"  plug_tip→entrance: ({plug_to_port[0]*1000:.1f}, {plug_to_port[1]*1000:.1f}, {plug_to_port[2]*1000:.1f}) mm"
            )

        self.get_logger().info("=== DebugGroundTruth DONE — holding standoff ===")
        self._hold_standoff(move_robot, tcp_f)
        return True

    def _hold_standoff(self, move_robot, pose, duration_sec: float = 2.0):
        steps = int(duration_sec / 0.05)
        for _ in range(steps):
            self.set_pose_target(move_robot=move_robot, pose=pose)
            self.sleep_for(0.05)
