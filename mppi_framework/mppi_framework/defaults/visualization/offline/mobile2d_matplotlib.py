# defaults/visualization/offline/mobile2d_matplotlib.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from pathlib import Path

from mppi_framework.core.registry import VIS_RENDERERS
from mppi_framework.interfaces.visualization import VisualizationOutput


@VIS_RENDERERS.register("mobile2d_matplotlib")
class Mobile2DMatplotlibRenderer:
    """
    Mobile2D matplotlib offline renderer (GIF).
    원본 utils/mobile2d_renderer.py의 plot 설정/스타일을 그대로 유지.
    """

    def render(
        self,
        *,
        ts,
        xs,
        Xss=None,
        Xopt=None,
        save_path="outputs/mobile2d.gif",
        body_radius=0.15,
        world=None,            # (xmin, xmax, ymin, ymax)
        goal=None,             # (gx, gy)
        trail_stride=2,
        show_heading=True,
        heading_scale=1.0,
        obstacles=None,        # [{"x":..,"y":..,"radius":..}, ...]
        **kwargs,
    ) -> VisualizationOutput:
        # ---- 원본 설정 그대로 ----
        plt.rcParams.update({
            "font.family": "Calibri",  # 글꼴
            "font.weight": "bold",             # 전체 bold

            "axes.titleweight": "bold",
            "axes.labelweight": "bold",

            "font.size": 150,
            "axes.titlesize": 150,
            "axes.labelsize": 150,
            "xtick.labelsize": 150,
            "ytick.labelsize": 150,
            "legend.fontsize": 80,
        })

        ts = np.asarray(ts)
        xs = np.asarray(xs)

        T = len(ts)
        assert xs.shape == (T, 3), "xs should be (T,3)=[x,y,theta]"

        # 월드 범위 자동 결정 (샘플이 있으면 포함)
        if world is None:
            pad = max(0.15, body_radius * 1.0)
            xmin = float(xs[:, 0].min())
            xmax = float(xs[:, 0].max())
            ymin = float(xs[:, 1].min())
            ymax = float(xs[:, 1].max())

            if Xss is not None:
                Xss_arr = np.asarray(Xss)
                # ✅ (B,H,3) 들어오면 앞 2개만 사용
                if Xss_arr.shape[-1] >= 2:
                    Xss_xy = Xss_arr[..., :2]
                else:
                    Xss_xy = Xss_arr

                assert Xss_xy.shape[0] == T, "Xss first dim must match len(ts)"
                xmin = min(xmin, float(Xss_xy[..., 0].min()))
                xmax = max(xmax, float(Xss_xy[..., 0].max()))
                ymin = min(ymin, float(Xss_xy[..., 1].min()))
                ymax = max(ymax, float(Xss_xy[..., 1].max()))

            if Xopt is not None:
                Xopt_arr = np.asarray(Xopt)
                assert Xopt_arr.shape[0] == T, "Xopt first dim must match len(ts)"
                xmin = min(xmin, float(Xopt_arr[..., 0].min()))
                xmax = max(xmax, float(Xopt_arr[..., 0].max()))
                ymin = min(ymin, float(Xopt_arr[..., 1].min()))
                ymax = max(ymax, float(Xopt_arr[..., 1].max()))

            if obstacles is not None:
                for o in obstacles:
                    ox, oy, r = float(o["x"]), float(o["y"]), float(o["radius"])
                    xmin = min(xmin, ox - r)
                    xmax = max(xmax, ox + r)
                    ymin = min(ymin, oy - r)
                    ymax = max(ymax, oy + r)

            xmin, xmax = xmin - 0.2, xmax + 0.2
            ymin, ymax = ymin - 0.2, ymax + 0.2
        else:
            xmin, xmax, ymin, ymax = world

        fig, ax = plt.subplots(figsize=(40, 40))
        plt.subplots_adjust(right=0.98, top=0.98)

        ax.set_aspect('equal')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, ls='--', alpha=0.8, lw=4.0)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.tick_params(axis='both', width=2)

        for spine in ax.spines.values():
            spine.set_linewidth(5)

        # 목표점
        if goal is not None:
            gx, gy = goal
            ax.plot([gx], [gy], 'rx', ms=100, mew=20)

        # 🔹 장애물
        obstacle_patches = []
        if obstacles is not None:
            for o in obstacles:
                ox, oy, r = o["x"], o["y"], o["radius"]

                # (1) 회색 채움 (테두리 없음)
                fill_circ = Circle(
                    (ox, oy),
                    radius=r,
                    fill=True,
                    facecolor="0.8",
                    edgecolor="none",
                    alpha=0.4
                )
                ax.add_patch(fill_circ)

                # (2) 점선 테두리만 표시
                outline_circ = Circle(
                    (ox, oy),
                    radius=r,
                    fill=False,
                    edgecolor="k",
                    lw=10,
                    linestyle='--'
                )
                ax.add_patch(outline_circ)

                obstacle_patches.append(outline_circ)

        # 메인 궤적(실행)
        trail_line, = ax.plot([], [], '-', lw=18, alpha=0.8, color='green')

        # 로봇 바디(원)
        cx0, cy0, th0 = xs[0]
        body = Circle((cx0, cy0), radius=body_radius, fill=False, lw=10)
        ax.add_patch(body)

        # heading 표시(원 중심에서 짧은 선)
        if show_heading:
            hd_x = [cx0, cx0 + body_radius * heading_scale * np.cos(th0)]
            hd_y = [cy0, cy0 + body_radius * heading_scale * np.sin(th0)]
            heading_line, = ax.plot(hd_x, hd_y, lw=5)
        else:
            heading_line = None

        # 샘플 롤아웃 라인 컬렉션 (프레임마다 교체)
        lc = None
        if Xss is not None:
            lc = LineCollection([], linewidths=18, colors='gray', alpha=0.18)
            ax.add_collection(lc)

        # 🔹 optimal plan 라인
        opt_line = None
        if Xopt is not None:
            opt_line, = ax.plot([], [], '-', lw=18, alpha=0.9, color='cyan')

        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

        def init():
            trail_line.set_data([], [])
            body.center = (xs[0, 0], xs[0, 1])
            if heading_line is not None:
                x, y, th = xs[0]
                hd_x = [x, x + body_radius * heading_scale * np.cos(th)]
                hd_y = [y, y + body_radius * heading_scale * np.sin(th)]
                heading_line.set_data(hd_x, hd_y)
            if lc is not None:
                lc.set_segments([])
            time_text.set_text("")
            return (trail_line, body, time_text) if heading_line is None and lc is None else \
                   (trail_line, body, heading_line, time_text) if lc is None else \
                   (trail_line, body, time_text, lc) if heading_line is None else \
                   (trail_line, body, heading_line, time_text, lc)

        def _segments_from_batch(batch_xy):
            segs = []
            for P in batch_xy:
                if P.shape[0] >= 2:
                    segs.append(np.stack([P[:-1], P[1:]], axis=1))
            if segs:
                return np.concatenate(segs, axis=0)
            return np.empty((0, 2, 2), dtype=float)

        def animate(i):
            trail_line.set_data(xs[:i + 1:trail_stride, 0], xs[:i + 1:trail_stride, 1])

            x, y, th = xs[i]
            body.center = (x, y)

            if heading_line is not None:
                hd_x = [x, x + body_radius * heading_scale * np.cos(th)]
                hd_y = [y, y + body_radius * heading_scale * np.sin(th)]
                heading_line.set_data(hd_x, hd_y)

            if lc is not None:
                batch_xy = Xss[i]
                segs = _segments_from_batch(batch_xy)
                lc.set_segments(segs)

            if opt_line is not None and Xopt is not None:
                opt_xy = Xopt[i]
                opt_line.set_data(opt_xy[:, 0], opt_xy[:, 1])

            time_text.set_text(f"t = {ts[i]:.2f}s")
            return (trail_line, body, time_text) if heading_line is None and lc is None else \
                   (trail_line, body, heading_line, time_text) if lc is None else \
                   (trail_line, body, time_text, lc) if heading_line is None else \
                   (trail_line, body, heading_line, time_text, lc)

        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=T, interval=1000 * (ts[1] - ts[0]), blit=True
        )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig("outputs/mobile2d_t0.png", dpi=150, bbox_inches='tight')
        anim.save(save_path, writer="pillow", fps=int(1.0 / (ts[1] - ts[0])))
        plt.close(fig)

        return VisualizationOutput(save_path=save_path)
