import torch
from mppi_framework.core.registry import DYNAMICS
from mppi_framework.interfaces.dynamics import BaseDynamics, DynamicsSpec

# ===== CartPole from here =====
@DYNAMICS.register("cartpole")
class CartPoleDynamics(BaseDynamics):
    """
    State x = [x, x_dot, theta, theta_dot]
    Input u = [force] (N)
    Model: OpenAI Gym CartPole physics vectorized and implemented in Torch.
    Euler integration (not semi-implicit).
    """
    def __init__(
        self,
        dt: float = 0.02,          # (similar to Gym default)
        g: float = 9.8,            # gravity
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 0.5,       # pole half-length (meters)
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        angle_wrap: bool = True,   # whether to wrap angle to -pi~pi
    ):
        spec = DynamicsSpec(state_dim=4, control_dim=1, dt=dt)
        super().__init__(spec, device, dtype)
        self.g = g
        self.mc = masscart
        self.mp = masspole
        self.mt = masscart + masspole
        self.length = length                  # half-length
        self.pml = masspole * length         # polemass_length
        self.angle_wrap = angle_wrap


    @torch.no_grad()
    def f(self, x, u):
        """
        x: [B,4] = [x, x_dot, theta, theta_dot]
        u: [B,1] = force (N)
        return x_next: [B,4]
        """
        # Align device/dtype
        x = x.to(self.device, self.dtype)
        u = u.to(self.device, self.dtype)
        
        # Decompose state
        x_pos   = x[..., 0:1]
        x_dot   = x[..., 1:2]
        theta   = x[..., 2:3]
        th_dot  = x[..., 3:4]

        # Common terms
        sin_th = torch.sin(theta)
        cos_th = torch.cos(theta)

        # temp = (F + pml * th_dot^2 * sin(th)) / mt
        temp = (u + self.pml * (th_dot ** 2) * sin_th) / self.mt

        # theta_acc = (g*sin(th) - cos(th)*temp) / (length*(4/3 - mp*cos^2(th)/mt))
        denom = self.length * (4.0/3.0 - (self.mp * (cos_th ** 2)) / self.mt)
        theta_acc = (self.g * sin_th - cos_th * temp) / denom

        # x_acc = temp - pml * theta_acc * cos(th) / mt
        x_acc = temp - (self.pml * theta_acc * cos_th) / self.mt

        # Euler integration
        dt = self.spec.dt
        x_pos_next  = x_pos  + dt * x_dot
        x_dot_next  = x_dot  + dt * x_acc
        theta_next  = theta  + dt * th_dot
        th_dot_next = th_dot + dt * theta_acc

        if self.angle_wrap:
            # Wrap to -pi ~ pi
            pi = torch.pi
            theta_next = (theta_next + pi) % (2 * pi) - pi

        x_next = torch.cat([x_pos_next, x_dot_next, theta_next, th_dot_next], dim=-1)
        return x_next
