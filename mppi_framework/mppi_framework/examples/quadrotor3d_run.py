# examples/quadrotor3d_run.py
import argparse
import numpy as np
import torch

from mppi_framework.__init__ import build_controller, build_offline_renderer, build_online_visualizer
from mppi_framework.defaults.mppi import MPPIConfig
from mppi_framework.interfaces.visualization import RolloutLog


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--dt", type=float, default=0.02)

    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--lambda_", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.99)

    # u = [T, tau_phi, tau_theta, tau_psi]
    p.add_argument("--u_min", type=float, nargs='+', default=[0.0, -0.5, -0.5, -0.5])
    p.add_argument("--u_max", type=float, nargs='+', default=[15.0, 0.5, 0.5, 0.5])
    p.add_argument("--std_init", type=float, nargs='+', default=[2.0, 0.3, 0.3, 0.1])

    # initial / goal state (only care about position + yaw, others start from 0)
    p.add_argument("--x0", type=float, default=-1.0)
    p.add_argument("--y0", type=float, default=-1.0)
    p.add_argument("--z0", type=float, default=1.0)

    p.add_argument("--gx", type=float, default=0.0)
    p.add_argument("--gy", type=float, default=0.0)
    p.add_argument("--gz", type=float, default=2.0)

    p.add_argument("--record_sample", type=bool, default=False)
    p.add_argument("--save", type=str, default="outputs/quad3d.gif")
    args = p.parse_args()

    # 1) MPPI config
    cfg = MPPIConfig(
        horizon=args.horizon,
        samples=args.samples,
        lambda_=args.lambda_,
        gamma=args.gamma,
        u_min=args.u_min,          # [T, tau_phi, tau_theta, tau_psi]
        u_max=args.u_max,         
        device=args.device,
        dtype=torch.float32,
        record_sample=args.record_sample,
    )

    # 2) Build controller
    # state: [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
    x_goal = (
        args.gx, args.gy, args.gz,   # position
        0.0, 0.0, 0.0,               # velocity
        0.0, 0.0, 0.0,               # roll, pitch, yaw
        0.0, 0.0, 0.0,               # angular velocity
    )

    ctrl = build_controller(
        cfg,
        dynamics_name="quadrotor3d",
        cost_name="quadratic",
        sampler_name="gaussian",
        dynamics_cfg={
            "dt": args.dt,
            "mass": 1.0,
            "Jx": 0.02,
            "Jy": 0.02,
            "Jz": 0.04,
            "angle_wrap": True,
            # device/dtype are passed by build_controller
        },
        cost_cfg={
            # Diagonal Q for 12D state
            # [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
            "Q": [8.0, 8.0, 10.0,   # position
                  1.0, 1.0, 1.0,    # velocity
                  2.0, 2.0, 1.0,    # roll, pitch, yaw
                  0.5, 0.5, 0.5],   # angular velocity
            "R": [0.01, 0.01, 0.01, 0.01],  # input penalty
            "x_goal": x_goal,
            "device": args.device,
        },
        sampler_cfg={
            "std_init": args.std_init,
            "device": args.device,
        },
    )

    

    renderer = build_offline_renderer("quadrotor3d_matplotlib")

    rec = build_online_visualizer("offline_recorder", {
        "renderer": renderer,
        "save_path": args.save,
        "dt": args.dt,
        "renderer_kwargs": {
            "goal": (args.gx, args.gy, args.gz),
        },
        "record_Xss": args.record_sample,
        "record_Xopt": args.record_sample,
        "max_rollouts": 300,
        "state_slice": [0, 1, 2, 6, 7, 8],
    })
    rec.reset()

    T = args.steps
    # 3) Initial state: [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
    x0 = torch.tensor(
        [[args.x0, args.y0, args.z0,
          0.0, 0.0, 0.0,
          0.0, 0.0, 0.0,
          0.0, 0.0, 0.0]],
        device=args.device
    )
    x = x0

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

        if (t % 20) == 0:
            x_np = x.squeeze(0).detach().cpu().numpy()
            print(f"t={t:03d}  pos=({x_np[0]:+.2f},{x_np[1]:+.2f})  th={x_np[2]:+.2f}  u=[{u[0].item():+.2f},{u[1].item():+.2f}]")

    print("[done] final state:", x.tolist())
    out = rec.finalize()
    print(f"[saved] {out.save_path}")


if __name__ == "__main__":
    main()
