# examples/cartpole_run.py
import argparse
import torch
import numpy as np
from mppi_framework.__init__ import build_controller, build_offline_renderer, build_online_visualizer
from mppi_framework.defaults.mppi import MPPIConfig
from mppi_framework.interfaces.visualization import RolloutLog

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--theta0", type=float, default=3.14)  # initial angle (rad)
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--lambda_", type=float, default=1.0)
    p.add_argument("--u_min", type=float, default=-10.0)
    p.add_argument("--u_max", type=float, default=10.0)
    p.add_argument("--range", type=float, default=8.0)
    p.add_argument("--record_sample", type=bool, default=True)
    p.add_argument("--save", type=str, default="outputs/cartpole_uniform.gif")
    args = p.parse_args()

    # 1) Build controller
    cfg = MPPIConfig(
        horizon=args.horizon,
        samples=args.samples,
        lambda_=args.lambda_,
        gamma=1.0,
        u_min=args.u_min,
        u_max=args.u_max,
        device=args.device,
        dtype=torch.float32,
        record_sample=args.record_sample,
    )

    ctrl = build_controller(
        cfg,
        dynamics_name="cartpole",
        cost_name="quadratic",
        sampler_name="uniform",
        dynamics_cfg={"dt": args.dt, "angle_wrap": True, "device": args.device},
        cost_cfg={"Q": [20.0, 0.1, 5.0, 0.5], "R": [0.01], "x_goal": (0., 0., 0., 0.), "device": args.device},
        sampler_cfg={"range": args.range, "device": args.device},
    )

     # offline renderer (matplotlib gif)
    renderer = build_offline_renderer("cartpole_matplotlib")

    rec = build_online_visualizer("offline_recorder", {
        "renderer": renderer,
        "save_path": args.save,
        "dt": args.dt,
        "renderer_kwargs": {
            "pole_half_length": 0.5,
        },
        "record_Xss": args.record_sample,
        "record_Xopt": args.record_sample,
        "max_rollouts": 300,
    })
    rec.reset()

    # 2) Initial state: [x, x_dot, theta, theta_dot]
    x = torch.tensor([[0.0, 0.0, args.theta0, 0.0]], device=args.device)

    # 3) Log buffer
    T = args.steps

    for t in range(T):
        if args.record_sample:
            u, Xs,Us,noise, costs = ctrl.step(x)
            Xopt = ctrl.predict_traj(x,Us)
            log = RolloutLog(Xs=Xs, noise=noise, Xopt=Xopt, costs=costs)
        else:
            u = ctrl.step(x)
            log = None

        x = ctrl.f.step(x, u.unsqueeze(0))  # [1,3]

        # online-style update (store only)
        rec.update(t, x.squeeze(0), u, log=log)

    out = rec.finalize()
    print(f"[saved] {out.save_path}")
if __name__ == "__main__":
    main()
