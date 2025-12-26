import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path

def _segments_from_batch(batch_xy: np.ndarray) -> np.ndarray:
    """
    batch_xy: (B, H, 2) 또는 (B, H+1, 2)
    반환: (n_segments, 2, 2) for LineCollection
    """
    segs = []
    for P in batch_xy:
        if P.shape[0] >= 2:
            segs.append(np.stack([P[:-1], P[1:]], axis=1))  # (H-1, 2, 2)
    if segs:
        return np.concatenate(segs, axis=0)
    return np.empty((0, 2, 2), dtype=float)

def plot_rollouts_step(
    xss_path: str,
    save_path: str,
    world: str | None = None,   # "xmin,xmax,ymin,ymax" or None(auto)
    pad: float = 0.8,
    lw: float = 0.8,
    alpha: float = 0.20,
    show_start: bool = False,
    show_end: bool = False,
    title: str | None = None,
):
    """
    xss_path: npy 파일 경로 (B,H,2) 또는 (B,H+1,2)
    """
    if not os.path.exists(xss_path):
        print(f"[Skip] not found: {xss_path}")
        return

    batch_xy = np.load(xss_path)  # (B,H,2) or (B,H+1,2)
    if batch_xy.ndim != 3 or batch_xy.shape[-1] != 2:
        raise ValueError(f"Xss must be (B,H,2) or (B,H+1,2), got {batch_xy.shape}")
    if batch_xy.shape[0] == 0:
        print(f"[Skip] empty batch in {xss_path}")
        return

    # world 범위
    if world is None:
        xmin = float(batch_xy[..., 0].min()); xmax = float(batch_xy[..., 0].max())
        ymin = float(batch_xy[..., 1].min()); ymax = float(batch_xy[..., 1].max())
        xmin, xmax = xmin - pad, xmax + pad
        ymin, ymax = ymin - pad, ymax + pad
    else:
        try:
            xmin, xmax, ymin, ymax = map(float, world.split(","))
        except Exception as e:
            raise ValueError(f"--world must be 'xmin,xmax,ymin,ymax', got {world}") from e

    # 그림
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    if title: ax.set_title(title)

    segs = _segments_from_batch(batch_xy)
    lc = LineCollection(segs, linewidths=lw, colors="gray", alpha=alpha)
    ax.add_collection(lc)

    if show_start:
        starts = batch_xy[:, 0, :]
        ax.scatter(starts[:, 0], starts[:, 1], s=16, c="tab:blue", alpha=0.8, label="start")
    if show_end:
        ends = batch_xy[:, -1, :]
        ax.scatter(ends[:, 0], ends[:, 1], s=16, c="tab:red", alpha=0.8, label="end")
    if show_start or show_end:
        ax.legend(loc="lower right")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {save_path}")

def main():
    p = argparse.ArgumentParser(description="Plot single-step rollouts (Xss) for multiple distributions.")
    p.add_argument("--dir", type=str, default="outputs", help="Base directory for npy/png.")
    p.add_argument("--names", type=str, nargs="+",
                   default=["sample_traj_gaussian", "sample_traj_log_nln", "sample_traj_uniform"],
                   help="Distribution name suffixes (file stem).")
    p.add_argument("--xss_tmpl", type=str, default="{name}.npy",
                   help="Template for Xss filename in --dir (use {name}).")
    p.add_argument("--png_tmpl", type=str, default="{name}.png",
                   help="Template for PNG filename in --dir (use {name}).")
    p.add_argument("--world", type=str, default="-0.5,0.5,-0.5,0.5", help="Optional 'xmin,xmax,ymin,ymax'.")
    p.add_argument("--pad", type=float, default=0.8)
    p.add_argument("--lw", type=float, default=0.8)
    p.add_argument("--alpha", type=float, default=0.20)
    p.add_argument("--show_start", action="store_true")
    p.add_argument("--show_end", action="store_true")
    p.add_argument("--title_prefix", type=str, default="Rollouts (single step) — ")
    args = p.parse_args()

    for name in args.names:
        xss_path = os.path.join(args.dir, args.xss_tmpl.format(name=name))
        png_path = os.path.join(args.dir, args.png_tmpl.format(name=name))
        title = args.title_prefix + name
        plot_rollouts_step(
            xss_path=xss_path,
            save_path=png_path,
            world=args.world,
            pad=args.pad,
            lw=args.lw,
            alpha=args.alpha,
            show_start=args.show_start,
            show_end=args.show_end,
            title=title,
        )

if __name__ == "__main__":
    main()
