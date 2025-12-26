import torch
from dataclasses import dataclass
from mppi_framework.core.registry import DYNAMICS
from mppi_framework.interfaces.dynamics import BaseDynamics, DynamicsSpec


@DYNAMICS.register("quadrotor3d")
class Quadrotor3DDynamics(BaseDynamics):
    """
    Simple 3D quadrotor model (12 states, 4 inputs)

    State x: [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
        px,py,pz : position in world coordinates [m]
        vx,vy,vz : velocity in world coordinates [m/s]
        roll, pitch, yaw : ZYX Euler angles [rad]
        p,q,r    : body angular velocity [rad/s]

    Input u: [T, tau_phi, tau_theta, tau_psi]
        T         : total thrust (along body z-axis, N)
        tau_phi   : roll torque [N·m]
        tau_theta : pitch torque [N·m]
        tau_psi   : yaw torque [N·m]

    """

    def __init__(
        self,
        dt: float = 0.02,
        mass: float = 1.0,
        Jx: float = 0.02,
        Jy: float = 0.02,
        Jz: float = 0.04,
        g: float = 9.81,
        angle_wrap: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        spec = DynamicsSpec(
            state_dim=12,
            control_dim=4,
            dt=dt,
        )
        super().__init__(spec, device=device, dtype=dtype)

        self.mass = mass
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.g = g
        self.dt = dt

        self.angle_wrap = angle_wrap

        self.J = torch.tensor([Jx, Jy, Jz], device=device, dtype=dtype)  # [3]

    @torch.no_grad()
    def f(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        x: [B,12]
        u: [B,4] = [T, tau_phi, tau_theta, tau_psi]
        return x_next: [B,12]
        """
        device, dtype = self.device, self.dtype
        x = x.to(device=device, dtype=dtype)
        u = u.to(device=device, dtype=dtype)

        B = x.shape[0]
        dt = self.dt

        T = u[..., 0:1]               # [B,1]
        tau = u[..., 1:4]             # [B,3]

        # ----- Decompose state blocks -----
        # pos = [px, py, pz], vel = [vx, vy, vz],
        # ang = [roll, pitch, yaw], omega = [p, q, r]
        pos   = x[..., 0:3]   # [B,3]
        vel   = x[..., 3:6]   # [B,3]
        ang   = x[..., 6:9]   # [B,3]
        omega = x[..., 9:12]  # [B,3]

        roll  = ang[..., 0:1]     # [B,1]
        pitch = ang[..., 1:2]     # [B,1]
        yaw   = ang[..., 2:3]     # [B,1]

        p = omega[..., 0:1]       # [B,1]
        q = omega[..., 1:2]       # [B,1]
        r = omega[..., 2:3]       # [B,1]

        # ----- Rotation matrix R(body->world), ZYX -----
        cr = torch.cos(roll);  sr = torch.sin(roll)
        cp = torch.cos(pitch); sp = torch.sin(pitch)
        cy = torch.cos(yaw);   sy = torch.sin(yaw)

        # R = R_z(yaw) @ R_y(pitch) @ R_x(roll)
        R11 = cy*cp
        R12 = cy*sp*sr - sy*cr
        R13 = cy*sp*cr + sy*sr

        R21 = sy*cp
        R22 = sy*sp*sr + cy*cr
        R23 = sy*sp*cr - cy*sr

        R31 = -sp
        R32 = cp*sr
        R33 = cp*cr

        # How the body z-axis unit vector is seen in world frame: R * e3
        # Since e3 = [0,0,1], we only need the third column.
        # R_e3: [B,3]
        R_e3 = torch.cat([R13, R23, R33], dim=-1)

        # ----- Linear acceleration -----
        # f_world = (T/m) * R_e3 + g
        f_world = (T / self.mass) * R_e3                # [B,3]

        g_vec = torch.tensor([0.0, 0.0, -self.g], device=device, dtype=dtype)
        g_vec = g_vec.view(1,3)                         # [1,3]
        acc = f_world + g_vec                           # [B,3]

        vel_next = vel + dt * acc                       # [B,3]
        pos_next = pos + dt * vel_next                  # [B,3]

        # ----- Angular acceleration -----
        # J * w_dot = tau - w × (J w)
        J = self.J.view(1,3)                            # [1,3]
        Jomega = J * omega                              # [B,3]
        cross = torch.cross(omega, Jomega, dim=-1)      # [B,3]
        omega_dot = (tau - cross) / J                   # [B,3]
        omega_next = omega + dt * omega_dot             # [B,3]

        p_n = omega_next[..., 0:1]
        q_n = omega_next[..., 1:2]
        r_n = omega_next[..., 2:3]

        # ----- Euler angle kinematics -----
        # [roll_dot, pitch_dot, yaw_dot]^T = E(roll,pitch) * [p,q,r]^T
        tan_p = torch.tan(pitch)
        sec_p = 1.0 / torch.cos(pitch)

        roll_dot  = p + q*sr*tan_p + r*cr*tan_p
        pitch_dot = q*cr - r*sr
        yaw_dot   = q*sr*sec_p + r*cr*sec_p

        roll_next  = roll  + dt * roll_dot
        pitch_next = pitch + dt * pitch_dot
        yaw_next   = yaw   + dt * yaw_dot

        if self.angle_wrap:
            pi = torch.pi
            roll_next  = (roll_next  + pi) % (2*pi) - pi
            pitch_next = (pitch_next + pi) % (2*pi) - pi
            yaw_next   = (yaw_next   + pi) % (2*pi) - pi

        ang_next = torch.cat([roll_next, pitch_next, yaw_next], dim=-1)   # [B,3]

        # ----- Concatenate final state -----
        # pos_next, vel_next, ang_next, omega_next are all [B,3]
        x_next = torch.cat([pos_next, vel_next, ang_next, omega_next], dim=-1)  # [B,12]
        return x_next
