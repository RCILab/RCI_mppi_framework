import math
import torch

def rotmat_to_quat_wxyz(R: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    orig_shape = R.shape[:-2]          # (...,)
    R2 = R.reshape(-1, 3, 3)           # (N,3,3)
    N = R2.shape[0]

    t = R2[:, 0, 0] + R2[:, 1, 1] + R2[:, 2, 2]   # (N,)
    q = torch.zeros((N, 4), device=R.device, dtype=R.dtype)

    mask = t > 0.0
    if mask.any():
        s = torch.sqrt(t[mask] + 1.0) * 2.0
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (R2[mask, 2, 1] - R2[mask, 1, 2]) / (s + eps)
        q[mask, 2] = (R2[mask, 0, 2] - R2[mask, 2, 0]) / (s + eps)
        q[mask, 3] = (R2[mask, 1, 0] - R2[mask, 0, 1]) / (s + eps)

    mask2 = ~mask
    if mask2.any():
        Rm = R2[mask2]  # (M,3,3)
        diag = torch.stack([Rm[:, 0, 0], Rm[:, 1, 1], Rm[:, 2, 2]], dim=1)  # (M,3)
        i = torch.argmax(diag, dim=1)  # (M,)

        q2 = torch.zeros((Rm.shape[0], 4), device=R.device, dtype=R.dtype)

        m0 = i == 0
        if m0.any():
            s = torch.sqrt(1.0 + Rm[m0, 0, 0] - Rm[m0, 1, 1] - Rm[m0, 2, 2]) * 2.0
            q2[m0, 0] = (Rm[m0, 2, 1] - Rm[m0, 1, 2]) / (s + eps)
            q2[m0, 1] = 0.25 * s
            q2[m0, 2] = (Rm[m0, 0, 1] + Rm[m0, 1, 0]) / (s + eps)
            q2[m0, 3] = (Rm[m0, 0, 2] + Rm[m0, 2, 0]) / (s + eps)

        m1 = i == 1
        if m1.any():
            s = torch.sqrt(1.0 + Rm[m1, 1, 1] - Rm[m1, 0, 0] - Rm[m1, 2, 2]) * 2.0
            q2[m1, 0] = (Rm[m1, 0, 2] - Rm[m1, 2, 0]) / (s + eps)
            q2[m1, 1] = (Rm[m1, 0, 1] + Rm[m1, 1, 0]) / (s + eps)
            q2[m1, 2] = 0.25 * s
            q2[m1, 3] = (Rm[m1, 1, 2] + Rm[m1, 2, 1]) / (s + eps)

        m2 = i == 2
        if m2.any():
            s = torch.sqrt(1.0 + Rm[m2, 2, 2] - Rm[m2, 0, 0] - Rm[m2, 1, 1]) * 2.0
            q2[m2, 0] = (Rm[m2, 1, 0] - Rm[m2, 0, 1]) / (s + eps)
            q2[m2, 1] = (Rm[m2, 0, 2] + Rm[m2, 2, 0]) / (s + eps)
            q2[m2, 2] = (Rm[m2, 1, 2] + Rm[m2, 2, 1]) / (s + eps)
            q2[m2, 3] = 0.25 * s

        q[mask2] = q2

    q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + eps)
    return q.reshape(*orig_shape, 4)  # (...,4)


def T_from_R_t(R, t, device="cpu", dtype=torch.float32):
    T = torch.eye(4, device=device, dtype=dtype)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def Rz(a, device="cpu", dtype=torch.float32):
    ca, sa = math.cos(a), math.sin(a)
    return torch.tensor([[ca, -sa, 0.0],
                         [sa,  ca, 0.0],
                         [0.0, 0.0, 1.0]], device=device, dtype=dtype)

def Rx(a, device="cpu", dtype=torch.float32):
    ca, sa = math.cos(a), math.sin(a)
    return torch.tensor([[1.0, 0.0, 0.0],
                         [0.0,  ca, -sa],
                         [0.0,  sa,  ca]], device=device, dtype=dtype)



class FrankaDHFK:

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

        R_7_tool = Rz(-math.pi/4,self.device) @ Rx(math.pi,self.device)
        t_7_tool = torch.tensor([0.0, 0.0, 0.107], dtype=torch.float32,device=self.device)

        self.offset = T_from_R_t(R_7_tool, t_7_tool,device=self.device)

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

        T = T @ self.offset

        return T
    

    def fk_rot(self, T: torch.Tensor) -> torch.Tensor:
        return T[..., :3, :3]

    def fk_quat_wxyz(self, T: torch.Tensor) -> torch.Tensor:
        R = self.fk_rot(T)
        return rotmat_to_quat_wxyz(R)
    

    def fk_pos(self, T: torch.Tensor) -> torch.Tensor:
        return T[..., :3, 3]

