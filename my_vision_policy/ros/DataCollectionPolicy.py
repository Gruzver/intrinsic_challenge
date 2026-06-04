"""
DataCollectionPolicy — política mínima para recolección de datos.

Solo hace tres cosas:
  1. Va a home (posición conocida, buena vista de los puertos).
  2. Loguea qué frames TF de puertos están disponibles (confirma ground_truth:=true).
  3. Duerme HOLD_SEC segundos → el teleop toma el control del robot durante ese tiempo.
  4. Retorna True para que el engine avance al siguiente trial.

Uso:
  pixi reinstall ros-kilted-my-vision-policy
  pixi run ros2 run aic_model aic_model --ros-args \\
    -p use_sim_time:=true \\
    -p policy:=my_vision_policy.ros.DataCollectionPolicy

Simultáneamente:
  pixi run ros2 run aic_teleoperation cartesian_keyboard_teleop
  pixi run python tools/collect_dataset.py --captures 50 --interval 1.0
"""

from rclpy.time import Time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

# ── Configuración ──────────────────────────────────────────────────────────────

HOME_JOINTS = [-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110]

# Tiempo que la política mantiene el trial abierto para teleoperar y capturar datos.
# Aumentar si se necesita más tiempo por trial.
HOLD_SEC = 360.0   # 6 minutos por trial → ~360 capturas a 1/s

STEP_S = 0.05      # 20 Hz

# Todos los frames TF posibles de puertos (para validar que ground_truth funciona)
_PORT_FRAMES: list[tuple[str, str]] = []
for _i in range(5):
    for _j in range(2):
        _base = f"task_board/nic_card_mount_{_i}/sfp_port_{_j}"
        _PORT_FRAMES.append(("sfp", f"{_base}_link_entrance"))
        _PORT_FRAMES.append(("sfp", f"{_base}_link"))
for _i in range(2):
    _base = f"task_board/sc_port_{_i}/sc_port_base"
    _PORT_FRAMES.append(("sc", f"{_base}_link_entrance"))
    _PORT_FRAMES.append(("sc", f"{_base}_link"))


class DataCollectionPolicy(Policy):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, parent_node)
        self._trial_count = 0

    # ──────────────────────────────────────────────────────────────────────────

    def insert_cable(
        self,
        task:            Task,
        get_observation: GetObservationCallback,
        move_robot:      MoveRobotCallback,
        send_feedback:   SendFeedbackCallback,
    ) -> bool:
        self._trial_count += 1
        log = self.get_logger()
        log.info(
            f"=== DataCollectionPolicy — trial {self._trial_count} "
            f"| plug={task.plug_type} port={task.port_name} "
            f"module={task.target_module_name} ==="
        )
        send_feedback(
            f"DataCollection trial {self._trial_count}: {task.plug_type}/{task.port_name}"
        )

        # 1. Ir a home para tener buena vista del puerto
        log.info("Yendo a home...")
        self._go_home(move_robot)
        log.info("Home alcanzado.")

        # 2. Log de TF disponibles (confirma ground_truth:=true y qué puertos hay)
        self._log_available_tf(task)

        # 3. Dormir HOLD_SEC segundos — el teleop puede mover el robot libremente
        log.info(
            f"Manteniendo trial abierto {HOLD_SEC:.0f}s. "
            "Usa cartesian_keyboard_teleop para mover el robot "
            "y collect_dataset.py para capturar imágenes."
        )
        steps = int(HOLD_SEC / STEP_S)
        for i in range(steps):
            # Solo dormimos, no enviamos comandos — el teleop tiene el control
            self.sleep_for(STEP_S)
            # Log de progreso cada 30 segundos
            elapsed = (i + 1) * STEP_S
            if abs(elapsed % 30.0) < STEP_S:
                remaining = HOLD_SEC - elapsed
                log.info(f"  {elapsed:.0f}s / {HOLD_SEC:.0f}s — faltan {remaining:.0f}s")
                send_feedback(f"Capturando datos... {elapsed:.0f}s / {HOLD_SEC:.0f}s")

        log.info(f"Trial {self._trial_count} completado.")
        return True

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _go_home(self, move_robot: MoveRobotCallback, duration_sec: float = 4.0) -> None:
        cmd = JointMotionUpdate(
            target_stiffness=[80.0, 80.0, 80.0, 40.0, 40.0, 40.0],
            target_damping   =[50.0, 50.0, 50.0, 25.0, 25.0, 25.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION
            ),
        )
        steps = max(1, int(duration_sec / STEP_S))
        for _ in range(steps):
            cmd.target_state.positions = HOME_JOINTS
            move_robot(joint_motion_update=cmd)
            self.sleep_for(STEP_S)

    def _log_available_tf(self, task: Task) -> None:
        log = self.get_logger()

        # Frame específico del trial actual (el más importante)
        trial_frame = (
            f"task_board/{task.target_module_name}/{task.port_name}_link_entrance"
        )
        try:
            tf = self._tf_buffer.lookup_transform("base_link", trial_frame, Time())
            p  = tf.transform.translation
            log.info(
                f"[TF OK] {trial_frame} → "
                f"base_link: x={p.x:.4f} y={p.y:.4f} z={p.z:.4f}"
            )
        except TransformException:
            log.warn(
                f"[TF FAIL] {trial_frame} no disponible. "
                "¿Está corriendo con ground_truth:=true?"
            )

        # Escanear todos los frames disponibles
        found = []
        for port_type, frame in _PORT_FRAMES:
            try:
                self._tf_buffer.lookup_transform("base_link", frame, Time())
                found.append(frame)
            except TransformException:
                pass

        if found:
            log.info(f"Frames TF de puertos disponibles ({len(found)}):")
            for f in found:
                log.info(f"  ✓ {f}")
        else:
            log.warn("No se encontró ningún frame TF de puertos.")
