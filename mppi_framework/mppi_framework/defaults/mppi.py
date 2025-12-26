import torch
from dataclasses import dataclass
from mppi_framework.core.registry import ALGOS
from mppi_framework.interfaces.dynamics import BaseDynamics
from mppi_framework.interfaces.cost import BaseCost
from mppi_framework.interfaces.sampler import BaseNoiseSampler
from mppi_framework.utils.smoothers import Smoothers
@dataclass
class MPPIConfig:
    horizon: int = 20
    samples: int = 1024
    lambda_: float = 1.0
    gamma: float = 1.0
    u_min: torch.Tensor | None = None
    u_max: torch.Tensor | None = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    record_sample: bool = False

@ALGOS.register("mppi")
class MPPI:
    def __init__(self, dynamics: BaseDynamics, cost: BaseCost, sampler: BaseNoiseSampler, cfg: MPPIConfig):
        self.f = dynamics
        self.c = cost
        self.sampler = sampler
        self.cfg = cfg
        T, U = cfg.horizon, dynamics.spec.control_dim
        self.u_seq = torch.zeros(T, U, device=cfg.device, dtype=cfg.dtype)

        self.u_min = torch.tensor(cfg.u_min, device=cfg.device, dtype=cfg.dtype)
        self.u_max = torch.tensor(cfg.u_max, device=cfg.device, dtype=cfg.dtype)


        if self.cfg.gamma != 1.0:
            self.gamma = torch.pow(
                torch.as_tensor(cfg.gamma, device=cfg.device, dtype=cfg.dtype),
                torch.arange(T, device=cfg.device, dtype=cfg.dtype)
            ).view(1, T)                                  # [1,T]
        else:
            self.gamma = cfg.gamma

        if cfg.record_sample is None:
            self.record_sample = False
        else:
            self.record_sample = cfg.record_sample


    @torch.no_grad()
    def rollout(self, x0):
        B, T, U = self.cfg.samples, self.cfg.horizon, self.f.spec.control_dim
        device, dtype = self.cfg.device, self.cfg.dtype
        S = torch.zeros(B, device=device, dtype=dtype)
        Xs = torch.empty(B, T+1, self.f.spec.state_dim, device=device, dtype=dtype)
        noise = self.sampler.sample(B)                               # [B,T,U]
        noise = Smoothers.savitzky_golay_filter(noise, window_length=10, polyorder=2)
        u = self.u_seq.unsqueeze(0) + noise                          # [B,T,U]
        if self.cfg.u_min is not None:
                u = torch.clamp(u, min=self.u_min, max=self.u_max)
                noise = u - self.u_seq
        x = x0.expand(B, -1).to(device=device, dtype=dtype)          # [B,S]
        Xs[:, 0] = x

        for t in range(T):
            ut = u[:, t, :]
            x = self.f.step(x, ut)
            Xs[:, t+1] = x

        X = Xs[:, :-1]                                      # [B,T,S]
        C_stage = self.c.stage(X, u) * self.gamma               # [B,T]
        S = C_stage.sum(dim=1) + self.c.terminal(Xs[:, -1])
        return S, noise, Xs

    @torch.no_grad()
    def weights(self, costs):
        c_min = torch.min(costs)
        w = torch.exp(-(costs-c_min) / self.cfg.lambda_)
        w = w / (w.sum() + 1e-12)
        return w

    @torch.no_grad()
    def update(self, x0):
        costs, noise, Xs = self.rollout(x0)
        w = self.weights(costs)                         # [B]
        delta = (w.view(-1,1,1) * noise).sum(dim=0)      # [T,U]
        self.u_seq = self.u_seq + delta
        return self.u_seq[0], Xs , self.u_seq ,noise, costs                             # 첫 제어

    @torch.no_grad()
    def step(self, x_now):
        u0, Xs, Us ,noise, costs = self.update(x_now)
        self.u_seq = torch.cat([self.u_seq[1:], self.u_seq[-1:]], dim=0)
        if self.record_sample:
            return u0, Xs, Us ,noise, costs
        else:
            return u0
        
    @torch.no_grad()
    def predict_traj(self, x0, us):
        T = self.cfg.horizon
        S = self.f.spec.state_dim
        device, dtype = self.cfg.device, self.cfg.dtype

        x = x0.to(device=device, dtype=dtype).squeeze(0)  # [S]
        X_plan = torch.empty(T+1, S, device=device, dtype=dtype)
        X_plan[0] = x

        for t in range(T):
            u_t = us[t]          # [U]
            # dynamics는 [B,S], [B,U] 받으니까 batch dim 붙였다가 다시 squeeze
            x = self.f.step(x.unsqueeze(0), u_t.unsqueeze(0)).squeeze(0)
            X_plan[t+1] = x

        return X_plan   # [T+1, S]
