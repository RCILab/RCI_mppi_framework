import math
import torch

def rotmat_to_quat_wxyz(R: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    R: (...,3,3) -> quat (...,4) in (w,x,y,z)
    안정적인 branch 방식.
    """
    # trace
    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    q = torch.zeros(R.shape[:-2] + (4,), device=R.device, dtype=R.dtype)

    # case 1: trace > 0
    mask = t > 0.0
    if mask.any():
        s = torch.sqrt(t[mask] + 1.0) * 2.0  # s = 4*qw
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / (s + eps)
        q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / (s + eps)
        q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / (s + eps)

    # case 2: trace <= 0 -> pick max diagonal
    mask2 = ~mask
    if mask2.any():
        R2 = R[mask2]
        # indices of max diagonal element
        diag = torch.stack([R2[:, 0, 0], R2[:, 1, 1], R2[:, 2, 2]], dim=1)
        i = torch.argmax(diag, dim=1)

        q2 = torch.zeros((R2.shape[0], 4), device=R.device, dtype=R.dtype)

        # i == 0
        m0 = i == 0
        if m0.any():
            s = torch.sqrt(1.0 + R2[m0, 0, 0] - R2[m0, 1, 1] - R2[m0, 2, 2]) * 2.0
            q2[m0, 0] = (R2[m0, 2, 1] - R2[m0, 1, 2]) / (s + eps)
            q2[m0, 1] = 0.25 * s
            q2[m0, 2] = (R2[m0, 0, 1] + R2[m0, 1, 0]) / (s + eps)
            q2[m0, 3] = (R2[m0, 0, 2] + R2[m0, 2, 0]) / (s + eps)

        # i == 1
        m1 = i == 1
        if m1.any():
            s = torch.sqrt(1.0 + R2[m1, 1, 1] - R2[m1, 0, 0] - R2[m1, 2, 2]) * 2.0
            q2[m1, 0] = (R2[m1, 0, 2] - R2[m1, 2, 0]) / (s + eps)
            q2[m1, 1] = (R2[m1, 0, 1] + R2[m1, 1, 0]) / (s + eps)
            q2[m1, 2] = 0.25 * s
            q2[m1, 3] = (R2[m1, 1, 2] + R2[m1, 2, 1]) / (s + eps)

        # i == 2
        m2 = i == 2
        if m2.any():
            s = torch.sqrt(1.0 + R2[m2, 2, 2] - R2[m2, 0, 0] - R2[m2, 1, 1]) * 2.0
            q2[m2, 0] = (R2[m2, 1, 0] - R2[m2, 0, 1]) / (s + eps)
            q2[m2, 1] = (R2[m2, 0, 2] + R2[m2, 2, 0]) / (s + eps)
            q2[m2, 2] = (R2[m2, 1, 2] + R2[m2, 2, 1]) / (s + eps)
            q2[m2, 3] = 0.25 * s

        q[mask2] = q2

    # normalize
    q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + eps)
    return q


class FrankaDHFK:
    """
    Franka Panda DH 기반 batch FK (너가 주신 변환 규약 그대로).
    - T_offset 없음 (joint7 frame까지)
    - q shape: (..., 7) 지원 (예: [B,T,7] / [B,7])
    """

    def __init__(self, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.nq = 7

        # (7,3): [a, d, alpha]
        self.dhparams = torch.tensor([
            [ 0.0000, 0.3330,      0.0       ],   # Joint 1
            [ 0.0000, 0.0000, -math.pi/2      ],   # Joint 2
            [ 0.0000, 0.3160,  math.pi/2      ],   # Joint 3
            [ 0.0825, 0.0000,  math.pi/2      ],   # Joint 4
            [-0.0825, 0.3840, -math.pi/2      ],   # Joint 5
            [ 0.0000, 0.0000,  math.pi/2      ],   # Joint 6
            [ 0.0880, 0.0000,  math.pi/2      ],   # Joint 7
        ], device=self.device, dtype=self.dtype)

    def fk_T(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: (...,7) -> T_0_7: (...,4,4)
        """
        q = q.to(self.device, self.dtype)
        if q.shape[-1] != 7:
            raise ValueError(f"q last dim must be 7, got {q.shape}")

        a = self.dhparams[:, 0]
        d = self.dhparams[:, 1]
        alpha = self.dhparams[:, 2]

        ct = torch.cos(q)
        st = torch.sin(q)

        # broadcast alpha/a/d to q shape (...,7)
        view_shape = [1] * (q.ndim - 1) + [7]
        ca = torch.cos(alpha).view(*view_shape)
        sa = torch.sin(alpha).view(*view_shape)
        a_ = a.view(*view_shape)
        d_ = d.view(*view_shape)

        T = torch.eye(4, device=self.device, dtype=self.dtype).expand(q.shape[:-1] + (4, 4)).clone()

        for i in range(7):
            Ai = torch.zeros(q.shape[:-1] + (4, 4), device=self.device, dtype=self.dtype)

            # === 너가 올린 코드와 동일한 규약 ===
            Ai[..., 0, 0] = ct[..., i]
            Ai[..., 0, 1] = -st[..., i]
            Ai[..., 0, 2] = 0.0
            Ai[..., 0, 3] = a_[..., i]

            Ai[..., 1, 0] = st[..., i] * ca[..., i]
            Ai[..., 1, 1] = ct[..., i] * ca[..., i]
            Ai[..., 1, 2] = -sa[..., i]
            Ai[..., 1, 3] = -sa[..., i] * d_[..., i]

            Ai[..., 2, 0] = st[..., i] * sa[..., i]
            Ai[..., 2, 1] = ct[..., i] * sa[..., i]
            Ai[..., 2, 2] = ca[..., i]
            Ai[..., 2, 3] = ca[..., i] * d_[..., i]

            Ai[..., 3, 3] = 1.0

            T = T @ Ai

        return T
    

    def fk_rot(self, q: torch.Tensor) -> torch.Tensor:
        """q:(...,7) -> R:(...,3,3)"""
        T = self.fk_T(q)
        return T[..., :3, :3]

    def fk_quat_wxyz(self, q: torch.Tensor) -> torch.Tensor:
        """
        q:(...,7) -> quat:(...,4) in (w,x,y,z)
        """
        R = self.fk_rot(q)
        return rotmat_to_quat_wxyz(R)
    

    def fk_pos(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: (...,7) -> ee_pos: (...,3)
        """
        T = self.fk_T(q)
        return T[..., :3, 3]
