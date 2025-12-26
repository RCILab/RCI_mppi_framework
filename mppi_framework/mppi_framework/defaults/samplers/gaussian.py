import torch
from mppi_framework.core.registry import SAMPLERS
from mppi_framework.interfaces.sampler import BaseNoiseSampler

@SAMPLERS.register("gaussian")
class GaussianSampler(BaseNoiseSampler):
    def __init__(self, control_dim, horizon, std_init=0.5, device="cpu", dtype=torch.float32):
        super().__init__(control_dim, horizon)
        self.std = std_init
        self.device = device
        self.dtype = dtype

        # If std_init is a scalar → apply to all controls; if list/tensor → use as-is
        if isinstance(std_init, (float, int)):
            self.std = torch.full((control_dim,), std_init, device=device, dtype=dtype)
        else:
            self.std = torch.tensor(std_init, device=device, dtype=dtype)  # e.g., [0.3, 0.6, 0.9]
            

    def sample(self, B):
        return torch.randn(B, self.horizon, self.control_dim, device=self.device, dtype=self.dtype) * self.std
