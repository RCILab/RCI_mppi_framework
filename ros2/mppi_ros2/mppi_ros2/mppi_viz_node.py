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

# ---------------- [ADDED] Interactive Marker + Goal publish ----------------
from geometry_msgs.msg import PoseStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl
from visualization_msgs.msg import Marker as VisMarker
from visualization_msgs.msg import InteractiveMarkerFeedback
# -------------------------------------------------------------------------


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

        # ---------------- [ADDED] Marker/Goal params ----------------
        # Publish goal PoseStamped for MPPI node
        self.declare_parameter("goal_topic", "/ee_goal")
        # Interactive marker server namespace => topics:
        # /<im_server_name>/update , /<im_server_name>/feedback
        self.declare_parameter("im_server_name", "ee_goal_marker")
        # Frame for interactive marker (best if matches RViz Fixed Frame)
        self.declare_parameter("im_frame_id", "world")
        self.declare_parameter("im_scale", 0.2)
        # ------------------------------------------------------------

        # Read params (기존)
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

        # ---------------- [ADDED] Read marker/goal params ----------------
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.im_server_name = str(self.get_parameter("im_server_name").value)
        self.im_frame_id = str(self.get_parameter("im_frame_id").value)
        self.im_scale = float(self.get_parameter("im_scale").value)
        # ----------------------------------------------------------------

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

        # ---------------- [ADDED] Goal publisher + Interactive Marker server ----------------
        self.pub_goal = self.create_publisher(PoseStamped, self.goal_topic, 10)

        self.im_server = InteractiveMarkerServer(self, self.im_server_name)
        self._create_ee_goal_marker()

        fb_topic = f"/{self.im_server_name}/feedback"
        self.sub_im_feedback = self.create_subscription(
            InteractiveMarkerFeedback,
            fb_topic,
            self._on_im_feedback,
            10
        )
        
        # ----------------------------------------------------------------------------------

        self.get_logger().info("Viz Node Started. Waiting for JointStates...")
        self.get_logger().info(f"Publishing viz MarkerArray: {self.viz_topic}")
        self.get_logger().info(f"InteractiveMarker update topic: /{self.im_server_name}/update")
        self.get_logger().info(f"Publishing goal PoseStamped: {self.goal_topic}")
        self.get_logger().info(f"Subscribing IM feedback: {fb_topic}")

    # ---------------- [ADDED] Interactive marker creation ----------------
    def _create_ee_goal_marker(self):
        im = InteractiveMarker()
        im.header.frame_id = self.im_frame_id
        im.name = "ee_goal"
        im.description = "EE Goal (drag to move/rotate)"
        im.scale = float(self.im_scale)

        im.pose.position.x = 0.4
        im.pose.position.y = 0.0
        im.pose.position.z = 0.8
        im.pose.orientation.w = 1.0

        # Visible sphere
        sphere = VisMarker()
        sphere.type = VisMarker.SPHERE
        sphere.scale.x = 0.05
        sphere.scale.y = 0.05
        sphere.scale.z = 0.05
        sphere.color.r = 1.0
        sphere.color.g = 0.2
        sphere.color.b = 0.2
        sphere.color.a = 1.0

        vis_ctrl = InteractiveMarkerControl()
        vis_ctrl.always_visible = True
        vis_ctrl.markers.append(sphere)
        im.controls.append(vis_ctrl)

        # 6DoF controls
        self._add_6dof_controls(im)

        self.im_server.insert(im)
        self.im_server.applyChanges()

    def _add_6dof_controls(self, im: InteractiveMarker):
        def add(name, x, y, z, mode):
            c = InteractiveMarkerControl()
            c.name = name
            c.orientation.w = 1.0
            c.orientation.x = float(x)
            c.orientation.y = float(y)
            c.orientation.z = float(z)
            c.interaction_mode = mode
            im.controls.append(c)

        add("move_x", 1, 0, 0, InteractiveMarkerControl.MOVE_AXIS)
        add("move_y", 0, 1, 0, InteractiveMarkerControl.MOVE_AXIS)
        add("move_z", 0, 0, 1, InteractiveMarkerControl.MOVE_AXIS)

        add("rot_x", 1, 0, 0, InteractiveMarkerControl.ROTATE_AXIS)
        add("rot_y", 0, 1, 0, InteractiveMarkerControl.ROTATE_AXIS)
        add("rot_z", 0, 0, 1, InteractiveMarkerControl.ROTATE_AXIS)

    def _on_im_feedback(self, fb: InteractiveMarkerFeedback):
        # drag 중 포즈 업데이트만 goal로 씀
        if fb.event_type != InteractiveMarkerFeedback.POSE_UPDATE:
            return

        goal = PoseStamped()
        goal.header = fb.header
        goal.pose = fb.pose
        self.pub_goal.publish(goal)
    # ---------------------------------------------------------------------

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
        """
        T = us.shape[0]
        x = x0.unsqueeze(0)

        q_list = [x[:, :self.nq]]

        with torch.no_grad():
            for t in range(T):
                u_t = us[t].unsqueeze(0)
                x = self.dynamics.f(x, u_t)
                q_list.append(x[:, :self.nq])

        q_traj = torch.cat(q_list, dim=0).squeeze(1)
        return q_traj

    def _on_data(self, msg: Float64MultiArray):
        if self.curr_q is None or self.curr_qd is None:
            return

        raw_data = np.array(msg.data, dtype=np.float32)
        total_len = raw_data.size
        if total_len == 0:
            return

        len_us = self.horizon * self.nu
        if total_len < len_us:
            return

        us_flat = raw_data[:len_us]
        rem_data = raw_data[len_us:]

        Us_tensor = torch.as_tensor(us_flat, device=self.device).reshape(self.horizon, self.nu)

        x0 = torch.cat([self.curr_q, self.curr_qd], dim=0)

        q_opt = self._rollout_dynamics(x0, Us_tensor)

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

        now = self.get_clock().now().to_msg()
        msg_out = MarkerArray()

        for ns in ["opt", "sample"]:
            d = Marker()
            d.action = Marker.DELETEALL
            d.header.frame_id = self.viz_frame_id
            d.ns = ns
            d.id = 0
            msg_out.markers.append(d)

        stride = max(1, self.viz_time_stride)

        q_opt_strided = q_opt[::stride]
        with torch.no_grad():
            T = self.fk.fk_T(q_opt_strided.unsqueeze(0))
            ee_opt = self.fk.fk_pos(T).squeeze(0).cpu().numpy()

        msg_out.markers.append(self._make_line_strip_marker(
            ns="opt", mid=1, stamp=now,
            xyz=ee_opt,
            frame_id=self.viz_frame_id,
            width=self.viz_line_width_opt,
            rgba=(1.0, 0.0, 0.0, 1.0)
        ))

        if q_samples is not None:
            q_samp_strided = q_samples[:, ::stride, :]
            N, T_sub, _ = q_samp_strided.shape
            q_flat = q_samp_strided.reshape(-1, self.nq)
            T_flat = self.fk.fk_T(q_flat)
            with torch.no_grad():
                ee_flat = self.fk.fk_pos(T_flat)

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
