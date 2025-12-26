import torch
from mppi_framework.core.registry import COSTS
from mppi_framework.interfaces.cost import BaseCost
from mppi_framework.utils.franka_dh_fk import FrankaDHFK


def _normalize_quat_wxyz(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + eps)


def quat_distance_cost(q_cur_wxyz: torch.Tensor, q_goal_wxyz: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Includes handling of double-cover (q == -q).
    q_cur: (...,4), q_goal: (...,4)
    cost = 1 - |dot|^2  (smooth, 0~1)
    """
    q_cur = _normalize_quat_wxyz(q_cur_wxyz, eps)
    q_goal = _normalize_quat_wxyz(q_goal_wxyz, eps)
    dot = (q_cur * q_goal).sum(dim=-1).abs().clamp(0.0, 1.0)  # (...,)
    return 1.0 - dot * dot


@COSTS.register("ee_goal")
class EEGoalCost(BaseCost):
    """
    EE goal cost (Franka DH FK, batch torch)
    Assumed default state layout:
      X[..., 0:7]   = q
      X[..., 7:14]  = qd

    Orientation goal is quaternion(w,x,y,z).
    """

    def __init__(
        self,
        q_slice=(0, 7),
        qd_slice=(7, 14),

        # weights
        w_pos=1.0,
        w_rot=0.0,   # If 0, ignore orientation
        w_qd=0.0,
        w_u=0.0,

        # goals (with respect to joint7 frame)
        ee_goal_pos=(0.3, 0.0, 0.4),
        ee_goal_quat_wxyz=(1.0, 0.0, 0.0, 0.0),

        device="cpu",
        dtype=torch.float32,

        fk=None,
    ):
        self.device, self.dtype = device, dtype

        self.q0, self.q1 = q_slice
        self.v0, self.v1 = qd_slice

        self.w_pos = float(w_pos)
        self.w_rot = float(w_rot)
        self.w_qd = float(w_qd)
        self.w_u = float(w_u)

        self.fk = fk if fk is not None else FrankaDHFK(device=device, dtype=dtype)

        self._goal_pos = torch.as_tensor(ee_goal_pos, device=device, dtype=dtype).view(1, 1, 3)
        self._goal_quat = _normalize_quat_wxyz(
            torch.as_tensor(ee_goal_quat_wxyz, device=device, dtype=dtype).view(1, 1, 4)
        )

    # ✅ Runtime goal update
    def set_goal_pos(self, ee_goal_pos):
        self._goal_pos = torch.as_tensor(ee_goal_pos, device=self.device, dtype=self.dtype).view(1, 1, 3)

    def set_goal_quat_wxyz(self, ee_goal_quat_wxyz):
        self._goal_quat = _normalize_quat_wxyz(
            torch.as_tensor(ee_goal_quat_wxyz, device=self.device, dtype=self.dtype).view(1, 1, 4)
        )

    def set_goal_pose(self, pos_xyz, quat_wxyz):
        self.set_goal_pos(pos_xyz)
        self.set_goal_quat_wxyz(quat_wxyz)

    def stage(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device, self.dtype)
        U = U.to(self.device, self.dtype)

        q = X[..., self.q0:self.q1]                # [B,T,7]

        # position
        ee_pos = self.fk.fk_pos(q)                 # [B,T,3]
        dp = ee_pos - self._goal_pos
        c = self.w_pos * (dp * dp).sum(dim=-1)     # [B,T]

        # orientation (optional)
        if self.w_rot > 0.0:
            ee_quat = self.fk.fk_quat_wxyz(q)      # [B,T,4]
            c_rot = quat_distance_cost(ee_quat, self._goal_quat)  # [B,T]
            c = c + self.w_rot * c_rot

        # regularizers
        if self.w_qd > 0.0:
            qd = X[..., self.v0:self.v1]
            c = c + self.w_qd * (qd * qd).sum(dim=-1)

        if self.w_u > 0.0:
            c = c + self.w_u * (U * U).sum(dim=-1)

        return c

    def terminal(self, X_T: torch.Tensor) -> torch.Tensor:
        X_T = X_T.to(self.device, self.dtype)
        qT = X_T[:, self.q0:self.q1]               # [B,7]

        ee_pos = self.fk.fk_pos(qT)                # [B,3]
        goal_pos = self._goal_pos.view(1, 3)
        dp = ee_pos - goal_pos
        cT = self.w_pos * (dp * dp).sum(dim=-1)    # [B]

        if self.w_rot > 0.0:
            ee_quat = self.fk.fk_quat_wxyz(qT)     # [B,4]
            goal_quat = self._goal_quat.view(1, 4)
            c_rot = quat_distance_cost(ee_quat, goal_quat)        # [B]
            cT = cT + self.w_rot * c_rot

        return cT
