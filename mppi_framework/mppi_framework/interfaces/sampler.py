from abc import ABC, abstractmethod

class BaseNoiseSampler(ABC):
    def __init__(self, control_dim: int, horizon: int, **kw):
        self.control_dim = control_dim
        self.horizon = horizon

    @abstractmethod
    def sample(self, batch_size: int):
        """Return noise: [B, T, U] (torch.Tensor)"""
        ...

    def update(self, weights, noise, u_seq):
        """Override if needed (e.g., CEM / variance scheduling)"""
        return None
