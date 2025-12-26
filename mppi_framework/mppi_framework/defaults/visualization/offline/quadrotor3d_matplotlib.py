# defaults/visualization/offline/quadrotor3d_matplotlib.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from mppi_framework.core.registry import VIS_RENDERERS
from mppi_framework.interfaces.visualization import VisualizationOutput


@VIS_RENDERERS.register("quadrotor3d_matplotlib")
class Quadrotor3DMatplotlibRenderer:
    """
    Quadrotor3D matplotlib offline renderer (GIF).
    원본 utils/quadrotor3d_renderer.py의 plot 설정/스타일을 그대로 유지.
    """

    def render(
        self,
        *,
        ts,
        xs,
        Xss=None,
        save_path="outputs/quad3d.gif",
        world=None,          # (xmin, xmax, ymin, ymax, zmin, zmax)
        goal=None,           # (gx, gy, gz)
        trail_stride=2,
        obstacles=None,
        **kwargs,
    ) -> VisualizationOutput:
        plt.rcParams.update({
            "font.family": "Calibri",
            "font.weight": "bold",

            "axes.titleweight": "bold",
            "axes.labelweight": "bold",

            "font.size": 120,
            "axes.titlesize": 120,
            "axes.labelsize": 120,
            "xtick.labelsize": 120,
            "ytick.labelsize": 120,
            "legend.fontsize": 80,
        })

        ts = np.asarray(ts)
        xs = np.asarray(xs)
        T = len(ts)
        assert xs.shape == (T, 6), "xs must be (T,6) = [px,py,pz,roll,pitch,yaw]"

        if world is None:
            pad = 1.5
            xmin = float(xs[:, 0].min())
            xmax = float(xs[:, 0].max())
            ymin = float(xs[:, 1].min())
            ymax = float(xs[:, 1].max())
            zmin = float(xs[:, 2].min())
            zmax = float(xs[:, 2].max())

            if Xss is not None:
                Xss = np.asarray(Xss)
                assert Xss.shape[0] == T, "Xss first dim must match len(ts)"
                xmin = min(xmin, float(Xss[..., 0].min()))
                xmax = max(xmax, float(Xss[..., 0].max()))
                ymin = min(ymin, float(Xss[..., 1].min()))
                ymax = max(ymax, float(Xss[..., 1].max()))
                zmin = min(zmin, float(Xss[..., 2].min()))
                zmax = max(zmax, float(Xss[..., 2].max()))

            if obstacles is not None:
                for o in obstacles:
                    ox, oy, oz, r = o["x"], o["y"], o["z"], o["radius"]
                    xmin = min(xmin, ox - r)
                    xmax = max(xmax, ox + r)
                    ymin = min(ymin, oy - r)
                    ymax = max(ymax, oy + r)
                    zmin = min(zmin, oz - r)
                    zmax = max(zmax, oz + r)

            xmin, xmax = xmin - pad, xmax + pad
            ymin, ymax = ymin - pad, ymax + pad
            zmin, zmax = zmin - pad, zmax + pad
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = world

        fig = plt.figure(figsize=(30, 30))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel("X [m]", labelpad=100)
        ax.set_ylabel("Y [m]", labelpad=120)
        ax.set_zlabel("Z [m]", labelpad=80)
        ax.tick_params(axis='x', pad=20)
        ax.tick_params(axis='y', pad=40)
        ax.tick_params(axis='z', pad=30)

        for spine in ax.spines.values():
            spine.set_linewidth(15)

        if goal is not None:
            gx, gy, gz = goal
            ax.plot(
                [gx], [gy], [gz],
                marker='x',
                markersize=80,
                markeredgewidth=15,
                markeredgecolor='red',
                linestyle='None',
            )

        if obstacles is not None:
            u = np.linspace(0, 2 * np.pi, 24)
            v = np.linspace(0, np.pi, 12)
            uu, vv = np.meshgrid(u, v)

            for o in obstacles:
                ox, oy, oz, r = o["x"], o["y"], o["z"], o["radius"]

                x_sphere = ox + r * np.cos(uu) * np.sin(vv)
                y_sphere = oy + r * np.sin(uu) * np.sin(vv)
                z_sphere = oz + r * np.cos(vv)

                ax.plot_wireframe(
                    x_sphere, y_sphere, z_sphere,
                    rstride=2, cstride=2,
                    color='red', linewidth=1.5, alpha=0.6,
                )
                ax.plot_surface(
                    x_sphere, y_sphere, z_sphere,
                    color='red', alpha=0.12, linewidth=0
                )

        trail_line, = ax.plot([], [], [], '-', lw=18, alpha=0.9, color='green')
        point, = ax.plot([], [], [], linestyle='None')

        dummy = np.zeros((1, 2, 3), float)
        drone_lc = Line3DCollection(dummy, colors='blue', linewidths=8)
        ax.add_collection3d(drone_lc)

        prop_scatter = ax.scatter([], [], [], s=400, color='blue',
                                  edgecolors='black', linewidths=5)

        lc = None
        if Xss is not None:
            dummy = np.zeros((1, 2, 3), dtype=float)
            lc = Line3DCollection(dummy, linewidths=0.6, colors='gray', alpha=0.18)
            ax.add_collection3d(lc)

        time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

        def _segments_from_batch(batch_xyz):
            segs = []
            for P in batch_xyz:
                if P.shape[0] >= 2:
                    segs.append(np.stack([P[:-1], P[1:]], axis=1))
            if segs:
                return np.concatenate(segs, axis=0)
            return np.empty((0, 2, 3), dtype=float)

        def init():
            trail_line.set_data([], [])
            trail_line.set_3d_properties([])

            point.set_data([], [])
            point.set_3d_properties([])

            drone_lc.set_segments([])
            prop_scatter._offsets3d = ([], [], [])

            if lc is not None:
                lc.set_segments([])

            time_text.set_text("")

            artists = [trail_line, point, time_text, drone_lc, prop_scatter]
            if lc is not None:
                artists.append(lc)
            return artists

        def animate(i):
            trail_line.set_data(xs[:i + 1:trail_stride, 0], xs[:i + 1:trail_stride, 1])
            trail_line.set_3d_properties(xs[:i + 1:trail_stride, 2])

            px = xs[i, 0]
            py = xs[i, 1]
            pz = xs[i, 2]

            roll = xs[i, 3]
            pitch = xs[i, 4]
            yaw = xs[i, 5]

            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)

            Rz = np.array([[cy, -sy, 0],
                           [sy,  cy, 0],
                           [0,    0, 1]])

            Ry = np.array([[cp, 0, sp],
                           [0,  1, 0],
                           [-sp, 0, cp]])

            Rx = np.array([[1, 0,  0],
                           [0, cr, -sr],
                           [0, sr,  cr]])

            R = Rz @ Ry @ Rx

            arm = 0.15
            local_points = np.array([
                [-arm, -arm, 0],
                [ arm,  arm, 0],
                [-arm,  arm, 0],
                [ arm, -arm, 0],
            ])

            world_points = (R @ local_points.T).T + np.array([px, py, pz])

            drone_lines = np.array([
                [world_points[0], world_points[1]],
                [world_points[2], world_points[3]],
            ])
            drone_lc.set_segments(drone_lines)

            propellers = world_points
            prop_scatter._offsets3d = (
                propellers[:, 0],
                propellers[:, 1],
                propellers[:, 2],
            )

            if lc is not None and Xss is not None:
                batch_xyz = Xss[i]
                segs = _segments_from_batch(batch_xyz)
                lc.set_segments(segs)

            time_text.set_text(f"t = {ts[i]:.2f}s")

            artists = [trail_line, point, time_text, drone_lc, prop_scatter]
            if lc is not None:
                artists.append(lc)
            return artists

        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=T, interval=1000 * (ts[1] - ts[0]), blit=True
        )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        animate(0)
        fig.savefig("outputs/quad3d_t0.png", dpi=200,)

        anim.save(save_path, writer="pillow",
                  fps=int(1.0 / (ts[1] - ts[0])),
                  dpi=150)
        plt.close(fig)

        return VisualizationOutput(save_path=save_path)
