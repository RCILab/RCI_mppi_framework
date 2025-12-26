from typing import Optional
from dataclasses import dataclass
import torch

from mppi_framework.core.registry import DYNAMICS
from mppi_framework.interfaces.dynamics import BaseDynamics, DynamicsSpec

@DYNAMICS.register("mobile2d")
class Mobile2DDynamics(BaseDynamics):
    """
    Unicycle (vehicle) model
    state x = [x, y, theta]
    control u = [v, omega]
    x_{t+1} = f(x_t, u_t):
      x' = x + dt * v * cos(theta)
      y' = y + dt * v * sin(theta)
      th'= theta + dt * omega
    """
    def __init__(self,
                 dt: float = 0.05,
                 angle_wrap: bool = True,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.spec = DynamicsSpec(state_dim=3, control_dim=2, dt=dt)
        self.angle_wrap = angle_wrap

        self.device = device
        self.dtype = dtype

        # Tensorize constants (for convenience during broadcasting in computations)
        self._zero = torch.tensor(0.0, device=device, dtype=dtype)
        self._pi = torch.tensor(torch.pi, device=device, dtype=dtype)

    @torch.no_grad()
    def f(self, x, u):
        """
        x: [B,3] = [x, y, theta]
        u: [B,2] = [v, omega]
        return x_next: [B,3]
        """
        x = x.to(self.device, self.dtype)
        u = u.to(self.device, self.dtype)

        px   = x[..., 0:1]
        py   = x[..., 1:2]
        th   = x[..., 2:3]
        v    = u[..., 0:1]
        omg  = u[..., 1:2]

        dt = self.spec.dt
        cos_th = torch.cos(th)
        sin_th = torch.sin(th)

        px_next  = px + dt * v * cos_th
        py_next  = py + dt * v * sin_th
        th_next  = th + dt * omg

        if self.angle_wrap:
            # Wrap to [-pi, pi]
            th_next = (th_next + self._pi) % (2 * self._pi) - self._pi

        return torch.cat([px_next, py_next, th_next], dim=-1)
