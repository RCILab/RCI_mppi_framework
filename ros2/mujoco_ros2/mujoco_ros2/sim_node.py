import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.parameter import Parameter

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

import mujoco
from mujoco import viewer

from ament_index_python.packages import get_package_share_directory


def _id2name(model: mujoco.MjModel, objtype: mujoco.mjtObj, idx: int, fallback: str) -> str:
    name = mujoco.mj_id2name(model, objtype, idx)
    return name if name else fallback


def resolve_model_path(path: str) -> str:
    if path.startswith("package://"):
        pkg, rel = path.replace("package://", "").split("/", 1)
        return os.path.join(get_package_share_directory(pkg), rel)
    return path


class MujocoRos2Bridge(Node):
    """
    Robust MuJoCo <-> ROS2 bridge (generic):
      - Loads model XML
      - Steps simulation with configurable wall-rate and sim substeps
      - Publishes:
          * sensor_msgs/JointState on `state_topic` (names match published joints)
      - Subscribes commands via:
          * Float64MultiArray (actuator-order) on `cmd_actuator_topic`
          * JointState (name-based) on `cmd_joint_topic`
    """

    def __init__(self):
        super().__init__("mujoco_ros2_bridge")

        # ---------------- Parameters ----------------
        self.declare_parameter("model_xml", "")
        self.declare_parameter("state_topic", "/robot/joint_states")
        self.declare_parameter("cmd_actuator_topic", "/robot/actuator_cmd")
        self.declare_parameter("cmd_joint_topic", "/robot/joint_cmd")
        self.declare_parameter("wall_hz", 200.0)          # ROS timer frequency
        self.declare_parameter("use_sim_timestep", True)  # use model.opt.timestep
        self.declare_parameter("sim_dt", 0.002)           # only used if use_sim_timestep=False
        self.declare_parameter("substeps", 1)             # mj_step repeats per timer tick
        self.declare_parameter("publish_all_1dof_joints", True)  # publish 1DoF joints excluding free
        self.declare_parameter("publish_joint_names", [''])        # if non-empty, publish only these joints
        self.declare_parameter("zero_ctrl_on_start", True)
        self.declare_parameter("render", True)
        self.declare_parameter("initial_keyframe", "home")

        model_xml = self.get_parameter("model_xml").value
        model_xml = resolve_model_path(model_xml)
        if not model_xml:
            raise RuntimeError("Parameter 'model_xml' is empty. Provide an absolute path to MuJoCo XML.")
        if not os.path.exists(model_xml):
            raise FileNotFoundError(f"MuJoCo XML not found: {model_xml}")

        self.state_topic = self.get_parameter("state_topic").value
        self.cmd_actuator_topic = self.get_parameter("cmd_actuator_topic").value
        self.cmd_joint_topic = self.get_parameter("cmd_joint_topic").value

        wall_hz = float(self.get_parameter("wall_hz").value)
        if wall_hz <= 0.0:
            raise ValueError("wall_hz must be > 0")
        self.wall_dt = 1.0 / wall_hz

        self.substeps = int(self.get_parameter("substeps").value)
        if self.substeps < 1:
            self.substeps = 1

        use_sim_timestep = bool(self.get_parameter("use_sim_timestep").value)
        sim_dt = float(self.get_parameter("sim_dt").value)
        render = bool(self.get_parameter("render").value)
        key_name = self.get_parameter("initial_keyframe").value

        # ---------------- Load MuJoCo ----------------
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.data = mujoco.MjData(self.model)

        # ---------------- Key Pos ----------------
        if key_name:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
            if key_id >= 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
                mujoco.mj_forward(self.model, self.data)
                self.get_logger().info(f"Reset to keyframe '{key_name}'")

        # Simulation timestep (internal)
        self.sim_dt = float(self.model.opt.timestep) if use_sim_timestep else sim_dt

        # ---------------- Names & mappings ----------------
        # Actuator names (for actuator-order cmd)
        self.actuator_names: List[str] = [
            _id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i, f"actuator_{i}")
            for i in range(self.model.nu)
        ]

        # Build actuator -> joint mapping (best-effort)
        # For many actuator types, actuator_trnid[i,0] stores joint id.
        self.actuator_joint_id: List[Optional[int]] = []
        for i in range(self.model.nu):
            j_id = int(self.model.actuator_trnid[i, 0])
            if 0 <= j_id < self.model.njnt:
                self.actuator_joint_id.append(j_id)
            else:
                self.actuator_joint_id.append(None)

        self.joint_names_all: List[str] = [
            _id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j, f"joint_{j}")
            for j in range(self.model.njnt)
        ]

        # Build a list of publish joints
        requested = list(self.get_parameter("publish_joint_names").value)
        requested = [n for n in requested if n.strip() != ""]
        publish_all_1dof = bool(self.get_parameter("publish_all_1dof_joints").value)

        self.pub_joint_ids: List[int] = []
        if requested:
            # name-based selection
            name_to_id = {n: i for i, n in enumerate(self.joint_names_all)}
            for n in requested:
                if n in name_to_id:
                    self.pub_joint_ids.append(name_to_id[n])
                else:
                    self.get_logger().warn(f"publish_joint_names contains unknown joint: {n}")
        else:
            # generic selection: publish 1DoF joints excluding free
            # joint type: mjJNT_FREE has 7 qpos / 6 qvel (floating base)
            # mjJNT_HINGE / mjJNT_SLIDE are 1DoF
            for j in range(self.model.njnt):
                jtype = int(self.model.jnt_type[j])
                if not publish_all_1dof:
                    continue
                if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
                    continue
                if jtype in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    self.pub_joint_ids.append(j)

        # Publish joint names in the same order
        self.pub_joint_names = [self.joint_names_all[j] for j in self.pub_joint_ids]

        # Map joint id -> qpos/qvel address for 1DoF joints
        self.joint_qposadr = self.model.jnt_qposadr.copy()
        self.joint_dofadr = self.model.jnt_dofadr.copy()

        # Command buffers
        self.ctrl = np.zeros(self.model.nu, dtype=np.float64)
        self.last_cmd_time = self.get_clock().now()

        if bool(self.get_parameter("zero_ctrl_on_start").value) and self.model.nu > 0:
            self.data.ctrl[:] = 0.0

        # For joint-name-based command: build joint name -> actuator index best-effort
        # If actuator has a mapped joint id, we can map joint_name to actuator index.
        self.jointname_to_actuator: Dict[str, int] = {}
        for a_idx, j_id in enumerate(self.actuator_joint_id):
            if j_id is None:
                continue
            jn = self.joint_names_all[j_id]
            # if multiple actuators map to same joint, keep first (rare but possible)
            self.jointname_to_actuator.setdefault(jn, a_idx)
        
        # ---------------- Mujoco Viewer ----------------
        self.viewer = None
        if render:
            self.viewer = viewer.launch_passive(self.model, self.data)

        # ---------------- ROS I/O ----------------
        self.pub_state = self.create_publisher(JointState, self.state_topic, 10)

        self.sub_act_cmd = self.create_subscription(
            Float64MultiArray, self.cmd_actuator_topic, self.on_actuator_cmd, 10
        )
        self.sub_joint_cmd = self.create_subscription(
            JointState, self.cmd_joint_topic, self.on_joint_cmd, 10
        )
        self.first_cmd_received = False

        self.timer = self.create_timer(self.wall_dt, self.on_timer)

        # ---------------- Logs ----------------
        self.get_logger().info(f"Loaded MuJoCo model: {model_xml}")
        self.get_logger().info(
            f"nu={self.model.nu}, nq={self.model.nq}, nv={self.model.nv}, njnt={self.model.njnt}, timestep={self.model.opt.timestep}"
        )
        self.get_logger().info(f"wall_hz={wall_hz}, sim_dt={self.sim_dt}, substeps={self.substeps}")
        self.get_logger().info(f"Publish JointState: {self.state_topic}")
        self.get_logger().info(f"  joints({len(self.pub_joint_names)}): {self.pub_joint_names[:8]}{'...' if len(self.pub_joint_names)>8 else ''}")
        self.get_logger().info(f"Subscribe actuator cmd: {self.cmd_actuator_topic} (Float64MultiArray, len=nu)")
        self.get_logger().info(f"Subscribe joint cmd: {self.cmd_joint_topic} (JointState name-based)")
        if self.model.nu > 0:
            self.get_logger().info(f"actuators({self.model.nu}): {self.actuator_names[:8]}{'...' if self.model.nu>8 else ''}")

    # -------------- Command callbacks --------------
    def on_actuator_cmd(self, msg: Float64MultiArray):
        if self.model.nu == 0:
            return
        arr = np.asarray(msg.data, dtype=np.float64)
        if arr.size != self.model.nu:
            self.get_logger().warn(f"Actuator cmd size mismatch: got {arr.size}, expected nu={self.model.nu}")
            return
        self.ctrl[:] = arr
        self.last_cmd_time = self.get_clock().now()
        self.first_cmd_received = True

    def on_joint_cmd(self, msg: JointState):
        """
        Name-based command.
        For each msg.name[i], if it maps to some actuator, set ctrl[actuator] = msg.effort[i] or msg.position[i].
        Priority:
          - if msg.effort provided (non-empty): use effort as control
          - else if msg.position provided: use position as control (useful for position actuators)
          - else if msg.velocity provided: use velocity as control
        """
        if self.model.nu == 0:
            return
        if not msg.name:
            return

        # choose source array
        src = None
        if msg.effort:
            src = ("effort", msg.effort)
        elif msg.position:
            src = ("position", msg.position)
        elif msg.velocity:
            src = ("velocity", msg.velocity)
        else:
            self.get_logger().warn("JointState cmd has no effort/position/velocity to use as control.")
            return

        src_name, src_arr = src
        n = min(len(msg.name), len(src_arr))

        updated = 0
        for i in range(n):
            jn = msg.name[i]
            if jn not in self.jointname_to_actuator:
                continue
            a_idx = self.jointname_to_actuator[jn]
            self.ctrl[a_idx] = float(src_arr[i])
            updated += 1

        if updated == 0:
            self.get_logger().warn("JointState cmd: no joint names matched any actuator mapping.")
        self.last_cmd_time = self.get_clock().now()
        self.first_cmd_received = True

    # -------------- Main loop --------------
    def on_timer(self):
        # 1) If in HOLD, just skip step
        if not self.first_cmd_received:
            pass
        else:
            # 2) In normal mode, apply ctrl and step
            if self.model.nu > 0:
                self.data.ctrl[:] = self.ctrl

            for _ in range(self.substeps):
                mujoco.mj_step(self.model, self.data)

        # Publish selected joint states
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.pub_joint_names

        pos = []
        vel = []
        for j_id in self.pub_joint_ids:
            # 1DoF joint: qpos at joint_qposadr[j], qvel at joint_dofadr[j]
            qadr = int(self.joint_qposadr[j_id])
            dadr = int(self.joint_dofadr[j_id])

            pos.append(float(self.data.qpos[qadr]))
            vel.append(float(self.data.qvel[dadr]))

        js.position = pos
        js.velocity = vel
        self.pub_state.publish(js)

        # ---- Viewer sync ----
        if self.viewer is not None:
            self.viewer.sync()


def main():
    rclpy.init()
    node = MujocoRos2Bridge()
    try:
        rclpy.spin(node)
    finally:
        if getattr(node, "viewer", None) is not None:
            try:
                node.viewer.close()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
