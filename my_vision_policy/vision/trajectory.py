import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from transforms3d._gohlketransforms import quaternion_slerp


def smoothstep(t: float) -> float:
    # t²(3-2t): velocity = 0 at t=0 and t=1, minimizes jerk at endpoints
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def interpolate_pose(start: Pose, end: Pose, alpha: float) -> Pose:
    """Interpolate position linearly and orientation spherically (slerp).

    alpha in [0, 1]. Applies smoothstep so velocity is zero at both endpoints.
    """
    a = smoothstep(alpha)

    pos = Point(
        x=start.position.x + a * (end.position.x - start.position.x),
        y=start.position.y + a * (end.position.y - start.position.y),
        z=start.position.z + a * (end.position.z - start.position.z),
    )

    # transforms3d convention: (w, x, y, z);  ROS convention: (x, y, z, w)
    q0 = (start.orientation.w, start.orientation.x, start.orientation.y, start.orientation.z)
    q1 = (end.orientation.w, end.orientation.x, end.orientation.y, end.orientation.z)
    q = quaternion_slerp(q0, q1, a)

    return Pose(
        position=pos,
        orientation=Quaternion(w=q[0], x=q[1], y=q[2], z=q[3]),
    )
