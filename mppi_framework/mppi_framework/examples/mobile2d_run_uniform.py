# examples/2dmobile_run.py
import argparse
import torch

from mppi_framework.__init__ import build_controller, build_offline_renderer, build_online_visualizer
from mppi_framework.defaults.mppi import MPPIConfig
from mppi_framework.interfaces.visualization import RolloutLog



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--dt", type=float, default=0.02)

    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--lambda_", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.99)

    p.add_argument("--u_min", type=float, nargs='+', default=[-0.3, -3.0])
    p.add_argument("--u_max", type=float, nargs='+', default=[0.7, 3.0])
    p.add_argument("--std_init", type=float, nargs='+', default=[0.1, 2.0])

    # initial / goal states
    p.add_argument("--x0", type=float, default=0.0)
    p.add_argument("--y0", type=float, default=0.0)
    p.add_argument("--th0", type=float, default=1.5708)
    p.add_argument("--gx", type=float, default=2.0)
    p.add_argument("--gy", type=float, default=2.0)
    p.add_argument("--gth", type=float, default=1.5708)

    p.add_argument("--record_sample", type=bool, default=True)
    p.add_argument("--save", type=str, default="outputs/mobile2d_uniform.gif")
    args = p.parse_args()

    # 1) Build controller
    cfg = MPPIConfig(
        horizon=args.horizon,
        samples=args.samples,
        lambda_=args.lambda_,
        gamma=args.gamma,
        u_min=args.u_min,              # v, omega are clipped separately inside dynamics
        u_max=args.u_max,
        device=args.device,
        dtype=torch.float32,
        record_sample=args.record_sample,
    )

    obstacles = [
        {"x": 0.5, "y": 0.5, "radius": 0.2},
        {"x": 0.8, "y": 1.3, "radius": 0.2},
        {"x": 1.3, "y": 0.7, "radius": 0.2},
        {"x": 1.7, "y": 1.4, "radius": 0.2},

    ]

    ctrl = build_controller(
        cfg,
        dynamics_name="mobile2d",
        cost_name="composite",
        sampler_name="uniform",
        dynamics_cfg={
            "dt": args.dt,
            "angle_wrap": True,
            # device/dtype are already passed by build_controller (avoid duplicates)
        },
        cost_cfg={                       # 🔸 composite configuration
        "terms": [
            {
                "name": "quadratic",
                "weight": 2.0,
                "cfg": {
                    "Q": [2.0, 2.0, 0.0],
                    "R": [0.0, 0.0],
                    "x_goal": (args.gx, args.gy, args.gth),
                },
            },
            {
                "name": "obstacle2d",
                "weight": 10000.0,     # how strongly to penalize obstacle avoidance
                "cfg": {
                    "obstacles": obstacles,
                    "margin": 0.15,
                },
            },
        ],
        "device": args.device,      # shared option (passed to CompositeCost init)
    },
        sampler_cfg={
            "std": args.std_init, "device": args.device
        },
    )

    # offline renderer (matplotlib gif)
    renderer = build_offline_renderer("mobile2d_matplotlib")

    rec = build_online_visualizer("offline_recorder", {
        "renderer": renderer,
        "save_path": args.save,
        "dt": args.dt,
        "renderer_kwargs": {
            "body_radius": 0.15,
            "goal": (args.gx, args.gy),
            "obstacles": obstacles,
        },
        "record_Xss": args.record_sample,
        "record_Xopt": args.record_sample,
        "max_rollouts": 300,
        "state_slice": slice(0, 3),
    })
    rec.reset()

    
    T = args.steps
    x = torch.tensor([[args.x0, args.y0, args.th0]], device=args.device)

    for t in range(T):
        if args.record_sample:
            u, Xs,Us,noise, costs = ctrl.step(x)
            Xopt = ctrl.predict_traj(x,Us)
            log = RolloutLog(Xs=Xs[...,:2], noise=noise, Xopt=Xopt, costs=costs)
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
