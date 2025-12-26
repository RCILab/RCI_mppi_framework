# defaults/costs/obstacle2d.py
from mppi_framework.interfaces.cost import BaseCost
from mppi_framework.core.registry import COSTS
import torch


@COSTS.register("obstacle2d")
class Obstacle2dCost(BaseCost):
    """
    Simple example of an obstacle avoidance cost.
    obstacles: list of dicts with {"x":..., "y":..., "radius":...}
    """

    def __init__(self, obstacles, margin: float = 0.2,
                 device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype  = dtype

        # [N,3] = (x,y,r)
        obs = []
        for o in obstacles:
            obs.append([o["x"], o["y"], o["radius"]])
        self.obstacles = torch.tensor(obs, device=device, dtype=dtype)  # [N,3]
        self.margin = margin

    def stage(self, X, U):
        """
        X: [B,T,S] (S>=2: x,y,...)
        U: [B,T,U]
        return: [B,T]
        """
        pos = X[..., :2]   # [B,T,2]
        B, T, _ = pos.shape
        N = self.obstacles.shape[0]

        # (broadcast)
        obs_xy = self.obstacles[:, :2].view(1,1,N,2)   # [1,1,N,2]
        obs_r  = self.obstacles[:, 2].view(1,1,N,1)    # [1,1,N,1]

        p = pos.view(B, T, 1, 2)                       # [B,T,1,2]
        d = torch.norm(p - obs_xy, dim=-1, keepdim=True)  # [B,T,N,1]

        # Apply penalty when going inside the safety distance
        safe = obs_r + self.margin                     # [1,1,N,1]
        viol = torch.relu(safe - d)                    # [B,T,N,1]
        c = (viol ** 2).sum(dim=2).squeeze(-1)         # [B,T]
        return c

    def terminal(self, X_T):
        """
        X_T: [B,S] (S>=2: x,y,...)
        return: [B]
        """
        pos = X_T[..., :2]                # [B,2]
        B = pos.shape[0]
        N = self.obstacles.shape[0]

        obs_xy = self.obstacles[:, :2].view(1, N, 2)   # [1,N,2]
        obs_r  = self.obstacles[:, 2].view(1, N, 1)    # [1,N,1]

        p = pos.view(B, 1, 2)                          # [B,1,2]
        d = torch.norm(p - obs_xy, dim=-1, keepdim=True)  # [B,N,1]

        safe = obs_r + self.margin                     # [1,N,1]
        viol = torch.relu(safe - d)                    # [B,N,1]

        c = (viol ** 2).sum(dim=1).squeeze(-1)         # [B]
        return c
