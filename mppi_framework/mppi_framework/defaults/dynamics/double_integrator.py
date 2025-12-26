import torch
from mppi_framework.core.registry import DYNAMICS
from mppi_framework.interfaces.dynamics import BaseDynamics, DynamicsSpec


@DYNAMICS.register("double_integrator")
class DoubleIntegrator(BaseDynamics):
    """
    Generic double integrator:
      state x = [q, qd]  where q,qd in R^nq  -> state_dim = 2*nq
      control u = qdd     in R^nq           -> control_dim = nq

    Supports batched shapes:
      x: (..., 2*nq)
      u: (..., nq)
    """
    def __init__(
        self,
        dt=0.02,
        nq=1,
        device="cpu",
        dtype=torch.float32,
    ):
        self.nq = int(nq)
        spec = DynamicsSpec(state_dim=2 * self.nq, control_dim=self.nq, dt=float(dt))
        super().__init__(spec, device, dtype)


    @torch.no_grad()
    def f(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, self.dtype)
        u = u.to(self.device, self.dtype)

        if x.shape[-1] != 2 * self.nq:
            raise ValueError(f"x last dim must be {2*self.nq}, got {x.shape}")
        if u.shape[-1] != self.nq:
            raise ValueError(f"u last dim must be {self.nq}, got {u.shape}")


        dt = self.spec.dt
        q  = x[..., :self.nq]
        qd = x[..., self.nq:(2*self.nq)]

        q_next  = q  + qd * dt
        qd_next = qd + u  * dt
        return torch.cat([q_next, qd_next], dim=-1)
