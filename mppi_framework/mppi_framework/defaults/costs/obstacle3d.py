# defaults/costs/obstacle3d.py
from __future__ import annotations
from typing import Any, Dict, List
import torch

from mppi_framework.interfaces.cost import BaseCost
from mppi_framework.core.registry import COSTS


@COSTS.register("obstacle3d")
class Obstacle3DCost(BaseCost):
    """
    3D spherical obstacle avoidance cost.

    obstacles: list/tuple/list of dicts
        Example)
        obstacles = [
            {"x": 0.5, "y": 0.2, "z": 1.0, "radius": 0.3},
            {"x": 1.0, "y": 0.0, "z": 1.5, "radius": 0.2},
        ]

    margin: safety distance margin (m)
        The actual radius r is expanded by margin, so entering within r+margin incurs a penalty.

    power: exponent for the violation magnitude (default 2 → squared)
    """

    def __init__(
        self,
        *,
        obstacles,
        margin: float = 0.0,
        power: float = 2.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **_ignored,
    ):
        # BaseCost does not take __init__ arguments, so we do not call super()
        self.device = device
        self.dtype  = dtype

        # obstacles → [N,4] tensor (x,y,z,r)
        obs_list: List[List[float]] = []
        # Assume obstacles is of type list[dict]
        for o in obstacles:
            ox = float(o["x"])
            oy = float(o["y"])
            oz = float(o["z"])
            r  = float(o["radius"])
            obs_list.append([ox, oy, oz, r])

        self.obstacles = torch.tensor(obs_list, device=device, dtype=dtype)  # [N,4]
        self.margin = float(margin)
        self.power  = float(power)

    # === stage: X: [B,T,S], U: [B,T,U] → [B,T] ===
    def stage(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        X: [B,T,S] (S>=3: pos = X[...,:3] = [x,y,z])
        U: [B,T,U] (not used here, just for interface compatibility)
        return: [B,T] obstacle avoidance cost
        """
        device, dtype = self.device, self.dtype
        X = X.to(device=device, dtype=dtype)

        pos = X[..., :3]          # [B,T,3]
        B, T, _ = pos.shape
        N = self.obstacles.shape[0]

        # (broadcast)
        obs_xyz = self.obstacles[:, :3].view(1, 1, N, 3)   # [1,1,N,3]
        obs_r   = self.obstacles[:, 3].view(1, 1, N, 1)    # [1,1,N,1]

        # p: [B,T,1,3]
        p = pos.view(B, T, 1, 3)

        # Distance between each point and obstacle center
        d = torch.norm(p - obs_xyz, dim=-1, keepdim=True)  # [B,T,N,1]

        # Safety radius: r + margin
        safe_r = obs_r + self.margin                       # [1,1,N,1]

        # Amount of penetration inside safe_r (positive only) → violation
        viol = torch.relu(safe_r - d)                      # [B,T,N,1]

        if self.power != 1.0:
            c = (viol ** self.power).sum(dim=2).squeeze(-1)  # [B,T]
        else:
            c = viol.sum(dim=2).squeeze(-1)                  # [B,T]

        return c

    # === terminal: X_T: [B,S] → [B] ===
    def terminal(self, X_T: torch.Tensor) -> torch.Tensor:
        """
        X_T: [B,S] (S>=3: pos = X_T[...,:3])
        → [B] terminal obstacle cost
        """
        device, dtype = self.device, self.dtype
        X_T = X_T.to(device=device, dtype=dtype)

        pos = X_T[..., :3]           # [B,3]
        B = pos.shape[0]
        N = self.obstacles.shape[0]

        obs_xyz = self.obstacles[:, :3].view(1, N, 3)   # [1,N,3]
        obs_r   = self.obstacles[:, 3].view(1, N, 1)    # [1,N,1]

        p = pos.view(B, 1, 3)                           # [B,1,3]

        d = torch.norm(p - obs_xyz, dim=-1, keepdim=True)  # [B,N,1]
        safe_r = obs_r + self.margin                       # [1,N,1]
        viol = torch.relu(safe_r - d)                      # [B,N,1]

        if self.power != 1.0:
            c = (viol ** self.power).sum(dim=1).squeeze(-1)  # [B]
        else:
            c = viol.sum(dim=1).squeeze(-1)                  # [B]

        return c
