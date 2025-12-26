# mppi/defaults/cost.py
import torch
from mppi_framework.core.registry import COSTS
from mppi_framework.interfaces.cost import BaseCost

@COSTS.register("quadratic")
class QuadraticCost(BaseCost):
    def __init__(self, Q=1.0, R=0.01, x_goal=None, device="cpu", dtype=torch.float32):
        self.device, self.dtype = device, dtype
        self.x_goal = None if x_goal is None else torch.as_tensor(x_goal, device=device, dtype=dtype)

        # Q, R: allow scalar or vector (state/control dimensions)
        self.Q = torch.as_tensor(Q, device=device, dtype=dtype)
        self.R = torch.as_tensor(R, device=device, dtype=dtype)

        self.diag_Q = (self.Q.ndim == 1)
        self.diag_R = (self.R.ndim == 1)

    def stage(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        # X:[B,T,S], U:[B,T,U]
        X = X.to(self.device, self.dtype)
        U = U.to(self.device, self.dtype)

        Xerr = X - self.x_goal.view(1,1,-1)             # [B,T,S]

        if self.diag_Q:
            c_x = (Xerr * self.Q.view(1,1,-1) * Xerr).sum(dim=-1)     # [B,T]
        else:
            # full matrix: x^T Q x
            c_x = torch.einsum('bts,sv,btv->bt', Xerr, self.Q, Xerr)

        if self.diag_R:
            c_u = (U * self.R.view(1,1,-1) * U).sum(dim=-1)           # [B,T]
        else:
            c_u = torch.einsum('btu,uv,btv->bt', U, self.R, U)

        return c_x + c_u                                             # [B,T]

    def terminal(self, X_T: torch.Tensor) -> torch.Tensor:
        # Option: terminal penalizes only the state (add extra weight if needed)
        X_T = X_T.to(self.device, self.dtype)
        err = X_T - self.x_goal.view(1,-1)                           # [B,S]
        if self.diag_Q:
            return (err * self.Q.view(1,-1) * err).sum(dim=-1)       # [B]
        else:
            return torch.einsum('bs,sv,bv->b', err, self.Q, err)     # [B]
