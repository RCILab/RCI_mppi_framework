# defaults/visualization/offline/cartpole_matplotlib.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from pathlib import Path

from mppi_framework.core.registry import VIS_RENDERERS
from mppi_framework.interfaces.visualization import VisualizationOutput


@VIS_RENDERERS.register("cartpole_matplotlib")
class CartpoleMatplotlibRenderer:
    """
    CartPole matplotlib offline renderer (GIF + optional snapshot).
    원본 utils/cartpole_renderer.py의 설정/스타일 그대로 유지.
    """

    def render(
        self,
        *,
        ts,
        xs,
        pole_half_length=0.5,
        save_path="outputs/cartpole.gif",
        **kwargs,
    ) -> VisualizationOutput:
        ts = np.asarray(ts)
        xs = np.asarray(xs)

        plt.rcParams.update({
            "font.family": "Calibri",
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "font.size": 80,
            "axes.titlesize": 80,
            "axes.labelsize": 80,
            "xtick.labelsize": 80,
            "ytick.labelsize": 80,
            "legend.fontsize": 80,
        })

        cart_width, cart_height = 0.6, 0.3
        pole_len = 2.0 * pole_half_length
        track_half = 1.0

        fig, ax = plt.subplots(figsize=(25, 25))
        ax.set_xlim(-track_half - 0.5, track_half + 0.5)
        ax.set_ylim(-0.6, 1.0)
        ax.set_aspect('equal')
        ax.set_xlabel("X [m]")
        ax.tick_params(axis='both', width=2)
        ax.set_yticks([])

        ax.plot([-track_half - 1, track_half + 1], [0, 0], lw=1.0, color="k")

        for spine in ax.spines.values():
            spine.set_linewidth(3)

        cart = Rectangle(
            (-cart_width / 2, 0.0),
            cart_width,
            cart_height,
            fill=False,
            lw=8.0
        )
        ax.add_patch(cart)

        pole_line, = ax.plot([], [], lw=8.0)
        time_text = ax.text(0.02, 0.85, "", transform=ax.transAxes)

        def init():
            cart.set_xy((-cart_width / 2, 0.0))
            pole_line.set_data([], [])
            time_text.set_text("")
            cart.set_alpha(1.0)
            pole_line.set_alpha(1.0)
            return cart, pole_line, time_text

        def animate(i):
            x_pos = xs[i, 0]
            theta = xs[i, 2]

            cart.set_xy((x_pos - cart_width / 2, 0.0))

            base_x, base_y = x_pos, cart_height
            tip_x = base_x + pole_len * np.sin(theta)
            tip_y = base_y + pole_len * np.cos(theta)

            pole_line.set_data([base_x, tip_x], [base_y, tip_y])

            cart.set_alpha(1.0)
            pole_line.set_alpha(1.0)

            time_text.set_text(f"t = {ts[i]:.2f}s")
            return cart, pole_line, time_text

        T = len(ts)
        base_dir = Path(save_path).parent
        base_dir.mkdir(parents=True, exist_ok=True)

        anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=T,
            interval=1000 * (ts[1] - ts[0]),
            blit=True
        )

        anim.save(save_path, writer="pillow",
                  fps=int(1.0 / (ts[1] - ts[0])))

        plt.close(fig)
        return VisualizationOutput(save_path=save_path)
