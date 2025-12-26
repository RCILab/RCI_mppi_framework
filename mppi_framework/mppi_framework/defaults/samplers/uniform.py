import torch
from mppi_framework.core.registry import SAMPLERS
from mppi_framework.interfaces.sampler import BaseNoiseSampler


@SAMPLERS.register("uniform")
class UniformSampler(BaseNoiseSampler):
    def __init__(self, control_dim, horizon,
                 std=None,
                 u_min=None, u_max=None, alpha=0.3,
                 range=None,
                 device="cpu", dtype=torch.float32):
        super().__init__(control_dim, horizon)
        self.device, self.dtype = device, dtype

        if range is not None:
            r = torch.as_tensor(range, device=device, dtype=dtype)
            if r.numel() == 1: r = r.expand(control_dim)
            self.half_range = r

        elif std is not None:
            s = torch.as_tensor(std, device=device, dtype=dtype)
            if s.numel() == 1: s = s.expand(control_dim)
            self.half_range = (3.0 ** 0.5) * s

        elif (u_min is not None) and (u_max is not None):
            umin = torch.as_tensor(u_min, device=device, dtype=dtype)
            umax = torch.as_tensor(u_max, device=device, dtype=dtype)
            if umin.numel() == 1: umin = umin.expand(control_dim)
            if umax.numel() == 1: umax = umax.expand(control_dim)
            self.half_range = alpha * 0.5 * (umax - umin)

        else:
            # final fallback (a default like 6.0 that you were using)
            self.half_range = torch.full((control_dim,), 6.0, device=device, dtype=dtype)

    def sample(self, B):
        z = torch.rand((B, self.horizon, self.control_dim), device=self.device, dtype=self.dtype) * 2 - 1
        return z * self.half_range
