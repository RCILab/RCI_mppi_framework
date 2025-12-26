# examples/plot_mobile2d_env.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path


def plot_mobile2d_env_snapshot(
    ts,
    xs,
    save_path="outputs/mobile2d_env.png",
    body_radius=0.15,      # 원형 로봇 반경
    world=None,            # (xmin, xmax, ymin, ymax) 없으면 궤적으로 자동 결정
    goal=None,             # (gx, gy)
    show_heading=True,     # 진행방향 선 표시
    heading_scale=1.0,     # heading 선 길이 배율
    frame_idx=0,           # ts, xs 중 몇 번째 상태를 그릴지 (기본 0번째)
    dpi=300,
):
    """
    ts   : (T,)   시간 [s]
    xs   : (T,3)  실행 궤적 상태: [x, y, theta]
    """

    # ---- 폰트 설정 (renderer_mobile2d_gif랑 맞춤) ----
    plt.rcParams.update({
        "font.size": 14,           # 전체 글씨 크기
        "axes.titlesize": 16,      # 제목 크기
        "axes.labelsize": 14,      # x,y 라벨 크기
        "xtick.labelsize": 12,     # x축 숫자 크기
        "ytick.labelsize": 12,     # y축 숫자 크기
        "legend.fontsize": 12      # 범례 글씨 크기
    })

    ts = np.asarray(ts)
    xs = np.asarray(xs)
    T = len(ts)
    assert xs.shape == (T, 3), "xs should be (T,3)=[x,y,theta]"

    # frame_idx 정리
    if frame_idx < 0:
        frame_idx = T + frame_idx
    frame_idx = int(np.clip(frame_idx, 0, T - 1))

    # 월드 범위 자동 결정 (renderer_mobile2d_gif와 동일 로직에서 궤적만 사용)
    if world is None:
        pad = max(0.8, body_radius * 4.0)
        xmin = float(xs[:, 0].min())
        xmax = float(xs[:, 0].max())
        ymin = float(xs[:, 1].min())
        ymax = float(xs[:, 1].max())
        xmin, xmax = xmin - pad, xmax + pad
        ymin, ymax = ymin - pad, ymax + pad
    else:
        xmin, xmax, ymin, ymax = world

    # ---- Figure 생성 ----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, ls='--', alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Mobile2D (unicycle) snapshot")

    # 목표점
    if goal is not None:
        gx, gy = goal
        ax.plot([gx], [gy], 'rx', ms=10, label='goal')

    # 로봇 바디(원) - frame_idx 시점의 상태
    cx, cy, th = xs[frame_idx]
    body = Circle((cx, cy), radius=body_radius, fill=False, lw=2)
    ax.add_patch(body)

    # heading 표시
    if show_heading:
        hd_x = [cx, cx + body_radius * heading_scale * np.cos(th)]
        hd_y = [cy, cy + body_radius * heading_scale * np.sin(th)]
        ax.plot(hd_x, hd_y, lw=2, label='heading')

    # 범례 (goal이나 heading 있으면 나옴)
    if goal is not None or show_heading:
        ax.legend(loc='lower right')

    # ---- 저장 ----
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] {save_path}")


# ===============================================================
# 이 파일을 직접 실행했을 때: 예시 데이터 만들어서 바로 PNG 저장
# ===============================================================
if __name__ == "__main__":
    # 간단한 예시 궤적 (원 궤적)
    T = 100
    ts = np.linspace(0.0, 5.0, T)
    xs = np.zeros((T, 3))
    xs[:, 0] = 2.0 * np.cos(ts)      # x
    xs[:, 1] = 1.5 * np.sin(ts)      # y
    xs[:, 2] = ts                    # theta (그냥 회전)

    # 0번째 프레임 기준 스냅샷 하나 저장
    plot_mobile2d_env_snapshot(
        ts,
        xs,
        save_path="outputs/mobile2d_env.png",
        body_radius=0.15,
        world=None,          # None이면 궤적 기반 자동 범위
        goal=(0.0, 0.0),     # 원하면 목표점 안 쓰려면 None으로
        show_heading=True,
        heading_scale=1.0,
        frame_idx=0,         # 0번째 상태 기준 (원하면 -1로 마지막 상태)
        dpi=300,
    )
