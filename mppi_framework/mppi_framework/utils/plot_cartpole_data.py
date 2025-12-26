import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ----------------------------------------------------
# 🔧 설정 및 데이터 정의
# ----------------------------------------------------
base_dir = "outputs"
dt = 0.02

# 비교할 시나리오 정의 (라벨, 색상, 비용파일, 상태파일)
scenarios = [
    {
        "label": "Gaussian",
        "color": "#D32F2F",       # 진한 빨강
        "cost_file": "min_costs_cartpole.npy",
        "state_file": "cartpole_xs.npy"
    },
    {
        "label": "Lormal Log-Normal",
        "color": "#1976D2",       # 진한 파랑
        "cost_file": "min_costs_log_nln_cartpole.npy",
        "state_file": "cartpole_log_nln_xs.npy"
    },
    {
        "label": "Uniform",
        "color": "#388E3C",       # 진한 초록
        "cost_file": "min_costs_uniform_cartpole.npy",
        "state_file": "cartpole_uniform_xs.npy"
    }
]

# 출력 디렉토리 확인
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# ----------------------------------------------------
# 🎨 스타일 공통 설정
# ----------------------------------------------------
plt.rcParams.update({
    "font.family": "Calibri",
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "font.size": 25,
    "axes.titlesize": 25,
    "axes.labelsize": 25,
    "lines.linewidth": 5,      # 겹쳐서 그리므로 6 -> 5로 살짝 조정 (가독성 위함)
    "legend.fontsize": 20,     # 범례 폰트 크기
    "legend.framealpha": 0.9,  # 범례 배경 불투명도
})

def style_axis(ax):
    """축 스타일 공통 적용 함수"""
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.grid(True, ls="--", alpha=0.5, lw=2)
    ax.tick_params(axis='both', which='major', labelsize=25, width=2)

# ====================================================
# 1. Cost 비교 그래프 (Compare Costs)
# ====================================================
plt.figure(figsize=(12, 6))
ax = plt.gca()

for sc in scenarios:
    path = Path(base_dir) / sc["cost_file"]
    if path.exists():
        data = np.load(path)
        ts = np.arange(len(data)) * dt
        # alpha=0.8로 약간 투명하게 하여 겹치는 부분 보이게 함
        plt.plot(ts, data, color=sc["color"], label=sc["label"], alpha=0.8)
    else:
        print(f"[Skip] {sc['cost_file']} not found.")

style_axis(ax)
plt.xlabel("Time [s]")
plt.ylabel("Cost")
plt.xlim(0, 6)
plt.legend(loc="upper right") # 범례 위치
plt.tight_layout()

save_path = Path(base_dir) / "compare_costs.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[Saved] {save_path}")


# ====================================================
# 2. Position 비교 그래프 (Compare Position)
# ====================================================
plt.figure(figsize=(12, 6))
ax = plt.gca()

for sc in scenarios:
    path = Path(base_dir) / sc["state_file"]
    if path.exists():
        data = np.load(path)
        ts = np.arange(data.shape[0]) * dt
        x_pos = data[:, 0] # 0번 컬럼: 위치
        plt.plot(ts, x_pos, color=sc["color"], label=sc["label"], alpha=0.8)

# 목표 라인 (검은색 점선)
plt.axhline(0, color='black', ls='--', lw=3, alpha=0.6, label='Target')

style_axis(ax)
plt.xlabel("Time [s]")
plt.ylabel("X [m]")
plt.xlim(0, 6)
plt.legend(loc="best")
plt.tight_layout()

save_path = Path(base_dir) / "compare_position.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[Saved] {save_path}")


# ====================================================
# 3. Angle 비교 그래프 (Compare Angle)
# ====================================================
plt.figure(figsize=(12, 6))
ax = plt.gca()

max_angle_val = -np.inf
min_angle_val = np.inf

for sc in scenarios:
    path = Path(base_dir) / sc["state_file"]
    if path.exists():
        data = np.load(path)
        ts = np.arange(data.shape[0]) * dt
        theta = np.unwrap(data[:, 2]) # 2번 컬럼: 각도 + unwrap
        
        # y축 범위 설정을 위해 최대/최소 기록
        max_angle_val = max(max_angle_val, np.max(theta))
        min_angle_val = min(min_angle_val, np.min(theta))
        
        plt.plot(ts, theta, color=sc["color"], label=sc["label"], alpha=0.8)

# 목표 라인 그리기 (데이터 범위 내에 있는 0, 2pi 등)
target_candidates = [0, 2*np.pi, -2*np.pi]
target_drawn = False
for t_val in target_candidates:
    
    # 그래프 범위 근처에 목표값이 있으면 표시
    if (min_angle_val - 1.0) <= t_val <= (max_angle_val + 1.0):
        lbl = "Goal" if not target_drawn else None
        plt.axhline(t_val, color='black', ls='--', lw=3, alpha=0.6, label=lbl)
        target_drawn = True

style_axis(ax)
plt.xlabel("Time [s]")
plt.ylabel(r"$\theta$ [rad]") # LaTeX 스타일
plt.xlim(0, 6)
# plt.legend(loc="best")
plt.tight_layout()

save_path = Path(base_dir) / "compare_angle.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[Saved] {save_path}")