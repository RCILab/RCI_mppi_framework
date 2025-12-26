# mppi_ros2/mppi_node.py
import numpy as np
import torch
import json
import time
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import ParameterDescriptor

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped

from mppi_framework import build_controller
from mppi_framework.defaults.mppi import MPPIConfig
from mppi_framework.utils.franka_dh_fk import FrankaDHFK

# assuming the structure where robot-related modules are grouped under robot/
from mppi_ros2.robot import (
    RobotState, RobotStateConfig,
    RobotWrapper, RobotWrapperConfig,
)

class MppiRos2Node(Node):
    def __init__(self):
        super().__init__("mppi_ros2_node")

        # ----------------------------------------
        # Parameter descriptors
        # ----------------------------------------
        dyn_desc = ParameterDescriptor(dynamic_typing=True)

        # ----------------------------------------
        # 1) flat params
        # ----------------------------------------
        self.declare_parameter("state_topic", "/robot/joint_states")
        self.declare_parameter("cmd_actuator_topic", "/robot/actuator_cmd")
        self.declare_parameter("goal_topic", "")
        
        # [NEW] Viz data publishing topic
        self.declare_parameter("viz_data_topic", "/mppi/viz_data")

        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("cmd_timeout_sec", 0.2)

        self.declare_parameter("nq", 7)
        self.declare_parameter("nu", 9)

        self.declare_parameter("q_slice", [0, 7])
        self.declare_parameter("qd_slice", [7, 14])

        self.declare_parameter("joint_order", None, dyn_desc)
        self.declare_parameter("u_to_actuator_index", None, dyn_desc)
        self.declare_parameter("default_actuator_cmd", None, dyn_desc)

        # ----------------------------------------
        # 2) MPPI core
        # ----------------------------------------
        self.declare_parameter("device", "cpu")
        self.declare_parameter("dt", 0.02)
        self.declare_parameter("horizon", 20)
        self.declare_parameter("samples", 512)
        self.declare_parameter("lambda_", 1.0)
        self.declare_parameter("gamma", 0.99)
        self.declare_parameter("record_sample", False)

        self.declare_parameter("u_min", None, dyn_desc)
        self.declare_parameter("u_max", None, dyn_desc)

        self.declare_parameter("dynamics_name", "double_integrator")
        self.declare_parameter("sampler_name", "gaussian")
        self.declare_parameter("cost_name", "ee_goal")

        # ----------------------------------------
        # 3) cfg groups
        # ----------------------------------------
        self.declare_parameter("dynamics_cfg.nq", None, dyn_desc)
        self.declare_parameter("dynamics_cfg.dt", None, dyn_desc)

        self.declare_parameter("sampler_cfg.std_init", None, dyn_desc)
        self.declare_parameter("cost_cfg.device", None, dyn_desc)
        self.declare_parameter("cost_cfg.terms_json", "", dyn_desc)

        # ----------------------------------------
        # 4) Robot (Pinocchio)
        # ----------------------------------------
        self.declare_parameter("robot.urdf_path", "")
        self.declare_parameter("robot.mesh_dir", "")
        self.declare_parameter("robot.default_ee_frame", "")
        self.declare_parameter("robot.enable_torque_output", True)

        # ----------------------------------------
        # 5) Visualization Data Options (For publishing raw data)
        # ----------------------------------------
        # Reuse viz_enable as the flag for "whether to send visualization data".
        self.declare_parameter("viz_enable", False)
        # Sending every control step can be heavy; use stride to send every N steps.
        self.declare_parameter("viz_pub_stride", 5) 
        # Number of samples to send out of all samples (to reduce bandwidth).
        self.declare_parameter("viz_num_samples_send", 20)

        # ----------------------------------------
        # read flat
        # ----------------------------------------
        self.state_topic = str(self.get_parameter("state_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_actuator_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.viz_data_topic = str(self.get_parameter("viz_data_topic").value)

        self.control_hz = float(self.get_parameter("control_hz").value)
        self.cmd_timeout_sec = float(self.get_parameter("cmd_timeout_sec").value)

        self.nq = int(self.get_parameter("nq").value)
        self.nu = int(self.get_parameter("nu").value)

        self.q_slice = list(self.get_parameter("q_slice").value)
        self.qd_slice = list(self.get_parameter("qd_slice").value)

        joint_order_val = self.get_parameter("joint_order").value
        self.joint_order = list(joint_order_val) if joint_order_val else []

        idx_val = self.get_parameter("u_to_actuator_index").value
        self.u_to_actuator_index = list(idx_val) if idx_val else list(range(self.nq))
        if len(self.u_to_actuator_index) != self.nq:
            raise ValueError("u_to_actuator_index must have length nq")

        default_cmd_val = self.get_parameter("default_actuator_cmd").value
        if default_cmd_val:
            if len(default_cmd_val) != self.nu:
                raise ValueError(f"default_actuator_cmd must have length nu={self.nu}")
            self.default_cmd = np.asarray(default_cmd_val, dtype=float)
        else:
            self.default_cmd = np.zeros(self.nu, dtype=float)

        self.device = str(self.get_parameter("device").value)
        self.dt = float(self.get_parameter("dt").value)
        self.horizon = int(self.get_parameter("horizon").value)
        self.samples = int(self.get_parameter("samples").value)
        self.lambda_ = float(self.get_parameter("lambda_").value)
        self.gamma = float(self.get_parameter("gamma").value)
        self.record_sample = bool(self.get_parameter("record_sample").value)

        u_min_val = self.get_parameter("u_min").value
        u_max_val = self.get_parameter("u_max").value
        self.u_min = None if (u_min_val is None or len(u_min_val) == 0) else list(u_min_val)
        self.u_max = None if (u_max_val is None or len(u_max_val) == 0) else list(u_max_val)

        self.dynamics_name = str(self.get_parameter("dynamics_name").value)
        self.sampler_name = str(self.get_parameter("sampler_name").value)
        self.cost_name = str(self.get_parameter("cost_name").value)

        # Viz options
        self.viz_enable = bool(self.get_parameter("viz_enable").value)
        self.viz_pub_stride = int(self.get_parameter("viz_pub_stride").value)
        self.viz_num_samples_send = int(self.get_parameter("viz_num_samples_send").value)

        # ----------------------------------------
        # read cfg (flat keys -> dict)
        # ----------------------------------------
        self.dynamics_cfg = self._get_cfg_group("dynamics_cfg")
        self.sampler_cfg = self._get_cfg_group("sampler_cfg")
        self.cost_cfg = self._get_cfg_group("cost_cfg")

        if self.cost_name == "composite":
            tj = self.cost_cfg.get("terms_json", "")
            if isinstance(tj, str) and tj.strip():
                self.cost_cfg["terms"] = json.loads(tj)
                self.cost_cfg.pop("terms_json", None)
            else:
                raise ValueError("cost_name=composite requires cost_cfg.terms_json (JSON string)")

        self.dynamics_cfg.setdefault("dt", self.dt)
        self.dynamics_cfg.setdefault("nq", self.nq)
        self.sampler_cfg.setdefault("device", self.device)

        # ----------------------------------------
        # RobotState init
        # ----------------------------------------
        self.state = RobotState(RobotStateConfig(nq=self.nq, joint_order=self.joint_order))
        
        # ----------------------------------------
        # RobotWrapper init
        # ----------------------------------------
        self.robot_urdf_path = str(self.get_parameter("robot.urdf_path").value)
        self.robot_mesh_dir = str(self.get_parameter("robot.mesh_dir").value)
        self.robot_default_ee = str(self.get_parameter("robot.default_ee_frame").value)
        self.enable_torque_output = bool(self.get_parameter("robot.enable_torque_output").value)

        self.robot = None
        if self.enable_torque_output:
            if not self.robot_urdf_path:
                raise ValueError("robot.enable_torque_output=True but robot.urdf_path is empty.")
            rw_cfg = RobotWrapperConfig(
                urdf_path=self.robot_urdf_path,
                mesh_dir=(self.robot_mesh_dir if self.robot_mesh_dir else None),
                joint_order=(self.joint_order if self.joint_order else None),
                default_ee_frame=(self.robot_default_ee if self.robot_default_ee else None),
            )
            self.robot = RobotWrapper(rw_cfg)

        # ----------------------------------------
        # QoS / ROS I/O
        # ----------------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1, # Depth reduced for latency
        )
        self.sub_state = self.create_subscription(JointState, self.state_topic, self._on_state, qos)
        self.pub_cmd = self.create_publisher(Float64MultiArray, self.cmd_topic, qos)

        self.sub_goal = None
        if self.goal_topic:
            self.sub_goal = self.create_subscription(PoseStamped, self.goal_topic, self._on_goal, qos)

        # [NEW] Publisher for Visualization Data
        if self.viz_enable:
            self.pub_viz_data = self.create_publisher(Float64MultiArray, self.viz_data_topic, 1)

        self._busy = False
        self._viz_count = 0

        # ----------------------------------------
        # build MPPI
        # ----------------------------------------
        self.ctrl = self._build_controller()

        # control timer
        self.timer = self.create_timer(1.0 / self.control_hz, self._on_timer)

        self.get_logger().info(f"Subscribed: {self.state_topic}")
        self.get_logger().info(f"Publishing: {self.cmd_topic} (nu={self.nu}) @ {self.control_hz} Hz")
        if self.goal_topic:
            self.get_logger().info(f"Goal topic: {self.goal_topic}")
        
        if self.viz_enable:
            self.get_logger().info(f"Viz Data Publishing Enabled on: {self.viz_data_topic}")

    def _get_cfg_group(self, prefix: str) -> dict:
        out = {}
        pfx = prefix + "."
        for name, param in self._parameters.items():
            if not name.startswith(pfx):
                continue
            key = name[len(pfx):]
            val = param.value
            if val is None:
                continue
            out[key] = val
        return out

    def _build_controller(self):
        cfg = MPPIConfig(
            horizon=self.horizon,
            samples=self.samples,
            lambda_=self.lambda_,
            gamma=self.gamma,
            u_min=self.u_min,
            u_max=self.u_max,
            device=self.device,
            dtype=torch.float32,
            record_sample=self.record_sample,
        )
        ctrl = build_controller(
            cfg,
            dynamics_name=self.dynamics_name,
            cost_name=self.cost_name,
            sampler_name=self.sampler_name,
            dynamics_cfg=self.dynamics_cfg,
            sampler_cfg=self.sampler_cfg,
            cost_cfg=self.cost_cfg,
        )
        return ctrl

    def _on_state(self, msg: JointState):
        self.state.update_from_joint_state(msg)

    def _find_term_index(self, target_name: str):
        terms = self.cost_cfg.get("terms", [])
        for i, t in enumerate(terms):
            if isinstance(t, dict) and t.get("name") == target_name:
                return i
        return None

    def _on_goal(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        goal_pos = [p.x, p.y, p.z]
        goal_quat_wxyz = [q.w, q.x, q.y, q.z]

        subs = getattr(self.ctrl.c, "_costs", None)
        if not isinstance(subs, list):
            return

        idx = self._find_term_index("ee_goal")
        if idx is None or idx >= len(subs):
            return

        c = subs[idx]
        if hasattr(c, "set_goal_pose"):
            c.set_goal_pose(goal_pos, goal_quat_wxyz)
        else:
            if hasattr(c, "set_goal_pos"):
                c.set_goal_pos(goal_pos)
            if hasattr(c, "set_goal_quat_wxyz"):
                c.set_goal_quat_wxyz(goal_quat_wxyz)

    def _map_u_to_cmd(self, u: np.ndarray) -> np.ndarray:
        cmd = self.default_cmd.copy()
        for i in range(self.nq):
            ai = int(self.u_to_actuator_index[i])
            if 0 <= ai < self.nu:
                cmd[ai] = float(u[i])
        return cmd

    def _publish_cmd(self, cmd: np.ndarray):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in cmd.reshape(-1).tolist()]
        self.pub_cmd.publish(msg)

    def _publish_viz_data(self, Us, Xs):
        """
        [Updated] x_t removed. Only Us and Xs are sent.
        1. [Us] (Optimal Controls): size [T, nu]
        2. [Xs] (Sampled States):   size [N_send, T+1, nq] (already joint positions)
        """
        arr_flat = []
        
        # 1. Us (Optimal Control)
        # Viz node needs this for forward integration in dynamics.
        if Us is not None:
            # Us: [Horizon, nu]
            arr_flat.extend(Us.detach().cpu().numpy().flatten().tolist())
            
        # 2. Xs (Samples)
        # Samples are already states, so we just send them as-is.
        if Xs is not None and Xs.ndim == 3:
            K = Xs.shape[0]
            n_send = min(K, self.viz_num_samples_send)
            if n_send > 0:
                q0, q1 = self.q_slice[0], self.q_slice[1]
                Xs_sel = Xs[:n_send, :, q0:q1] # [n_send, T+1, nq]
                arr_flat.extend(Xs_sel.detach().cpu().numpy().flatten().tolist())

        # Publish
        if len(arr_flat) > 0:
            msg = Float64MultiArray()
            msg.data = arr_flat
            self.pub_viz_data.publish(msg)

    def _on_timer(self):
        if self._busy:
            return

        if not self.state.is_fresh(self.cmd_timeout_sec):
            if self.robot is not None:
                self._publish_cmd(self.robot.nle)
            return

        qqd = self.state.get_q_qd()
        if qqd is None:
            return
        q, qd = qqd

        self._busy = True
        try:
            state_dim = max(self.q_slice[1], self.qd_slice[1])
            x = np.zeros((1, state_dim), dtype=float)
            x[0, self.q_slice[0]:self.q_slice[1]] = q
            x[0, self.qd_slice[0]:self.qd_slice[1]] = qd
            x_t = torch.as_tensor(x, device=self.device, dtype=torch.float32)
            
            stime = time.perf_counter()

            if self.record_sample:
                u, Xs, Us, noise, costs = self.ctrl.step(x_t)
            else:
                u = self.ctrl.step(x_t)
                Xs, Us = None, None
                # If record_sample=False but we want Us for visualization:
                # depending on the MPPI implementation, step() may return only u
                # or (u, Us). Here we recommend enabling record_sample=True
                # when visualization is required. (One could also use
                # ctrl.prev_us, etc., but that is omitted here.)
            print("on time: ", time.perf_counter() - stime)
            qddot = u.detach().cpu().numpy().reshape(-1)

            if qddot.shape[0] != self.nq:
                raise ValueError(f"MPPI output len={qddot.shape[0]} != nq={self.nq}")

            if self.enable_torque_output and (self.robot is not None):
                self.robot.update(q, qd, update_frames=False)
                # v_des = qd + qddot * self.dt
                # q_des = q + v_des * self.dt
                # kp = 700.0
                # kd = 2*math.sqrt(kp)
                # kd = 20.0
                # qddot = kp * (q_des -q) - kd * qd
                tau = self.robot.mass @ qddot + self.robot.nle
                cmd = self._map_u_to_cmd(tau)
            else:
                cmd = self._map_u_to_cmd(qddot)

            self._publish_cmd(cmd)
            

            # [NEW] Publish Viz Data (with stride check)
            if self.viz_enable:
                self._viz_count += 1
                if self._viz_count >= self.viz_pub_stride:
                    self._viz_count = 0
                    # Us must be available to predict trajectories
                    if Us is None:
                        pass
                    else:
                        self._publish_viz_data(Us, Xs)

        except Exception as e:
            self.get_logger().error(f"MPPI step failed: {e}")
            self._publish_cmd(np.zeros(self.nu, dtype=float))
        finally:
            self._busy = False


def main():
    rclpy.init()
    node = MppiRos2Node()
    # Single Threaded Executor is sufficient now
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
