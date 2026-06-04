"""
Connector tip geometry and TCP approach pose calculator.

T_tcp_plug: fixed transform from TCP frame to connector tip frame.

Values derived from ground-truth TF lookups at home position
(DebugGroundTruth policy, gripper/tcp and cable/plug_link frames):

  At home: TCP  X=(1,0,0)  Y=(0,-1,0)  Z=(0,0,-1)   [R_base_tcp = diag(1,-1,-1)]

  R_tcp_plug = R_base_tcp.T @ R_base_plug
  t_tcp_plug = R_base_tcp.T @ (plug_pos_base - tcp_pos_base)

  SFP (cable_0/sfp_tip_link):
    plug frame at home — X=(0.998,0.052,0.020)  Y=(0.056,-0.935,-0.350)  Z=(0,0.350,-0.937)
    plug pos  = (-0.3718, 0.2148, 0.2653),  tcp pos = (-0.3718, 0.1942, 0.3195)

  SC (cable_1/sc_tip_link):
    plug frame at home — X=(-0.020,-0.894,-0.448)  Y=(-1,0.015,0.013)  Z=(-0.005,0.448,-0.894)
    plug pos  = (-0.3721, 0.2075, 0.3038),  tcp pos = (-0.3717, 0.1944, 0.3203)

Approach formula
────────────────
Port Z axis points INTO the port (downward). Standoff is in the -Z direction:
  p_plug_target = p_port - standoff * R_port[:, 2]   # above port entrance

Target TCP pose (aligns plug frame with port frame):
  R_tcp_target = R_base_port @ R_tcp_plug.T
  p_tcp_target = p_plug_target - R_tcp_target @ t_tcp_plug
"""

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from transforms3d.quaternions import mat2quat

from my_vision_policy.vision.port_pose import PortPose


# ── T_tcp_plug — derived from TF ground truth at home ─────────────────────────
# R_base_tcp_home = diag(1,-1,-1)  →  R_base_tcp.T = diag(1,-1,-1)

# SFP: R_tcp_plug = diag(1,-1,-1) @ R_base_plug_sfp
_R_TCP_PLUG = {
    "sfp": np.array([
        [ 0.998,  0.056,  0.000],
        [-0.052,  0.935, -0.350],
        [-0.020,  0.350,  0.937],
    ]),
    "sc": np.array([
        [-0.020, -1.000, -0.005],
        [ 0.894, -0.015, -0.448],
        [ 0.448, -0.013,  0.894],
    ]),
}

# t_tcp_plug = diag(1,-1,-1) @ (plug_pos - tcp_pos)  [m, in TCP frame]
_T_TCP_PLUG = {
    "sfp": np.array([ 0.000, -0.0206,  0.0542]),
    "sc":  np.array([-0.0004, -0.0131,  0.0165]),
}


# ── Approach pose ──────────────────────────────────────────────────────────────

def compute_approach_pose(
    port_pose: PortPose,
    plug_type: str,
    standoff_m: float = 0.05,
) -> Pose:
    """Compute TCP pose that places the connector tip directly above port entrance.

    The connector's insertion axis matches the port axis, so the plug can descend
    straight in without collision.

    Args:
        port_pose:  6-DOF port pose from solvePnP (position + R in base_link).
        plug_type:  'sfp' or 'sc'.
        standoff_m: distance to maintain between connector tip and port face [m].

    Returns:
        Pose (position + quaternion) for the TCP in base_link.
    """
    if plug_type not in _R_TCP_PLUG:
        raise ValueError(f"Unknown plug_type '{plug_type}'. Expected: {list(_R_TCP_PLUG)}")

    R_port = port_pose.R
    p_port = port_pose.position

    # Port Z axis points INTO the port (downward). Standoff goes in -Z direction.
    n_into = R_port[:, 2]
    p_plug_target = p_port - standoff_m * n_into

    # TCP orientation: align plug frame with port frame
    R_tcp_plug = _R_TCP_PLUG[plug_type]
    R_tcp_target = R_port @ R_tcp_plug.T

    # TCP position: back-project plug tip offset
    t_plug = _T_TCP_PLUG[plug_type]
    p_tcp_target = p_plug_target - R_tcp_target @ t_plug

    q_wxyz = mat2quat(R_tcp_target)

    return Pose(
        position=Point(
            x=float(p_tcp_target[0]),
            y=float(p_tcp_target[1]),
            z=float(p_tcp_target[2]),
        ),
        orientation=Quaternion(
            w=float(q_wxyz[0]),
            x=float(q_wxyz[1]),
            y=float(q_wxyz[2]),
            z=float(q_wxyz[3]),
        ),
    )
