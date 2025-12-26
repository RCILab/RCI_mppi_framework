# mppi_ros2/mppi_viz_node.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import torch
import numpy as np

# 1. FK module
from mppi_framework.utils.franka_dh_fk import FrankaDHFK

# 2. Dynamics module (use built-in DoubleIntegrator from the framework)
from mppi_framework.defaults.dynamics.double_integrator import DoubleIntegrator


class MppiVizNode(Node):
    def __init__(self):
        super().__init__("mppi_viz_node")

        # ----------------------------------------
        # Parameters
        # ----------------------------------------
        self.declare_parameter("viz_data_topic", "/mppi/viz_data")
        self.declare_parameter("state_topic", "/robot/joint_states")  # subscribed to create x0
        self.declare_parameter("viz_topic", "/mppi/viz")
        self.declare_parameter("viz_frame_id", "world")

        self.declare_parameter("nq", 7)
        self.declare_parameter("nu", 9)
        self.declare_parameter("horizon", 20)
        self.declare_parameter("dt", 0.02)

        self.declare_parameter("viz_line_width_opt", 0.01)
        self.declare_parameter("viz_line_width_samp", 0.005)
        self.declare_parameter("viz_alpha_samp", 0.25)
        self.declare_parameter("viz_time_stride", 2)
        
        self.declare_parameter("device", "cpu")

        # Read params
        self.viz_data_topic = str(self.get_parameter("viz_data_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)
        self.viz_topic = str(self.get_parameter("viz_topic").value)
        self.viz_frame_id = str(self.get_parameter("viz_frame_id").value)

        self.nq = int(self.get_parameter("nq").value)
        self.nu = int(self.get_parameter("nu").value)
        self.horizon = int(self.get_parameter("horizon").value)
        self.dt = float(self.get_parameter("dt").value)

        self.viz_line_width_opt = float(self.get_parameter("viz_line_width_opt").value)
        self.viz_line_width_samp = float(self.get_parameter("viz_line_width_samp").value)
        self.viz_alpha_samp = float(self.get_parameter("viz_alpha_samp").value)
        self.viz_time_stride = int(self.get_parameter("viz_time_stride").value)
        
        self.device = str(self.get_parameter("device").value)

        # ----------------------------------------
        # Modules setup
        # ----------------------------------------
        self.fk = FrankaDHFK(device=self.device, dtype=torch.float32)

        # [Core] Initialize dynamics module (for rollout prediction)
        self.dynamics = DoubleIntegrator(
            dt=self.dt,
            nq=self.nq,
            device=self.device,
            dtype=torch.float32
        )

        # ----------------------------------------
        # State Management (variables for building x0)
        # ----------------------------------------
        self.curr_q = None
        self.curr_qd = None

        # ----------------------------------------
        # Pub/Sub
        # ----------------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribe JointState -> continuously update x0 (q, qd)
        self.sub_state = self.create_subscription(
            JointState, self.state_topic, self._on_state, qos
        )

        # Subscribe Viz Data -> receive Us, Xs
        self.sub_data = self.create_subscription(
            Float64MultiArray, self.viz_data_topic, self._on_data, qos
        )
        
        self.pub_viz = self.create_publisher(MarkerArray, self.viz_topic, qos)

        self.get_logger().info("Viz Node Started. Waiting for JointStates...")

    def _on_state(self, msg: JointState):
        """
        Continuously updates the current robot state.
        This value will be used as x0 when visualization data arrives.
        """
        if len(msg.position) >= self.nq:
            self.curr_q = torch.tensor(msg.position[:self.nq], device=self.device, dtype=torch.float32)
            self.curr_qd = torch.tensor(msg.velocity[:self.nq], device=self.device, dtype=torch.float32)

    def _make_line_strip_marker(self, *, ns, mid, stamp, xyz, frame_id, width, rgba):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = ns
        m.id = int(mid)
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = float(width)
        
        m.color.r = float(rgba[0])
        m.color.g = float(rgba[1])
        m.color.b = float(rgba[2])
        m.color.a = float(rgba[3])

        for t in range(xyz.shape[0]):
            p = Point()
            p.x = float(xyz[t, 0])
            p.y = float(xyz[t, 1])
            p.z = float(xyz[t, 2])
            m.points.append(p)
        return m

    def _rollout_dynamics(self, x0: torch.Tensor, us: torch.Tensor) -> torch.Tensor:
        """
        Predict a trajectory using the DoubleIntegrator module.

        Args:
            x0: [2*nq] (q + qd)
            us: [T, nu]
        Returns:
            q_traj: [T+1, nq]
        """
        T = us.shape[0]
        # Add batch dimension to x0: [state_dim] -> [1, state_dim]
        x = x0.unsqueeze(0)
        
        # Save the first state
        q_list = [x[:, :self.nq]]
        
        with torch.no_grad():
            for t in range(T):
                u_t = us[t].unsqueeze(0)  # [1, nu]
                
                # Dynamics step: x_next = f(x, u)
                x = self.dynamics.f(x, u_t)
                
                # Save only q
                q_list.append(x[:, :self.nq])

        # Merge list: [T+1, 1, nq] -> [T+1, nq]
        q_traj = torch.cat(q_list, dim=0).squeeze(1)
        return q_traj

    def _on_data(self, msg: Float64MultiArray):
        # 1. Check that x0 is ready
        if self.curr_q is None or self.curr_qd is None:
            # Cannot predict if no JointState has been received yet
            return

        raw_data = np.array(msg.data, dtype=np.float32)
        total_len = raw_data.size
        if total_len == 0:
            return
        
        # 2. Parse Us (Optimal Controls)
        # mppi_node sends Us flattened at the beginning
        len_us = self.horizon * self.nu
        if total_len < len_us:
            return

        us_flat = raw_data[:len_us]
        rem_data = raw_data[len_us:]
        
        Us_tensor = torch.as_tensor(us_flat, device=self.device).reshape(self.horizon, self.nu)
        
        # 3. [Core] Build x0 and run rollout
        # Build x0 by concatenating q and qd from JointState
        x0 = torch.cat([self.curr_q, self.curr_qd], dim=0)  # [2*nq]
        
        # Predict using the dynamics module
        q_opt = self._rollout_dynamics(x0, Us_tensor)  # [T+1, nq]
        
        # 4. Parse Xs (Samples)
        # Remaining data from mppi_node corresponds to samples
        steps = self.horizon + 1
        dim = self.nq
        q_samples = None
        rem_len = rem_data.size
        
        if rem_len > 0 and (rem_len % (steps * dim) == 0):
            num_trajs = rem_len // (steps * dim)
            q_samples = torch.as_tensor(
                rem_data.reshape(num_trajs, steps, dim),
                device=self.device
            )
        
        # 5. Visualization
        now = self.get_clock().now().to_msg()
        msg_out = MarkerArray()
        
        # (1) Delete previous markers (ID 0)
        for ns in ["opt", "sample"]:
            d = Marker()
            d.action = Marker.DELETEALL
            d.header.frame_id = self.viz_frame_id
            d.ns = ns
            d.id = 0
            msg_out.markers.append(d)

        # Stride setting
        stride = max(1, self.viz_time_stride)
        
        # (2) Draw Optimal Path (Red)
        q_opt_strided = q_opt[::stride]
        with torch.no_grad():
            ee_opt = self.fk.fk_pos(q_opt_strided.unsqueeze(0)).squeeze(0).cpu().numpy()

        msg_out.markers.append(self._make_line_strip_marker(
            ns="opt", mid=1, stamp=now,
            xyz=ee_opt,
            frame_id=self.viz_frame_id,
            width=self.viz_line_width_opt,
            rgba=(1.0, 0.0, 0.0, 1.0)
        ))
        
        # (3) Draw Samples (Blue)
        if q_samples is not None:
            q_samp_strided = q_samples[:, ::stride, :]
            N, T_sub, _ = q_samp_strided.shape
            q_flat = q_samp_strided.reshape(-1, self.nq)
            
            with torch.no_grad():
                ee_flat = self.fk.fk_pos(q_flat)
            
            ee_samp = ee_flat.view(N, T_sub, 3).cpu().numpy()
            
            for i in range(N):
                msg_out.markers.append(self._make_line_strip_marker(
                    ns="sample", mid=i+1, stamp=now,
                    xyz=ee_samp[i],
                    frame_id=self.viz_frame_id,
                    width=self.viz_line_width_samp,
                    rgba=(0.0, 0.0, 1.0, self.viz_alpha_samp)
                ))

        self.pub_viz.publish(msg_out)


def main():
    rclpy.init()
    node = MppiVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
