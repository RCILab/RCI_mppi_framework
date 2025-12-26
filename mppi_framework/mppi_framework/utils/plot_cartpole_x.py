import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ----------------------------------------------------
# 🔧 설정 파트
T = 300
dt = 0.02
base_dir = "outputs"

# 출력 디렉토리 확인
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# --- [핵심] 처리할 작업 목록 정의 ---
# 형식: (입력 npy 파일명, 출력 pos 이미지명, 출력 angle 이미지명)
# ※ 파일명이 다르다면 이 리스트의 첫 번째 요소들을 실제 파일명으로 수정하세요.
tasks = [
    ("cartpole_xs.npy",           "cartpole_pos.png",           "cartpole_angle.png"),
    ("cartpole_log_nln_xs.npy",   "cartpole_log_nln_pos.png",   "cartpole_log_nln_angle.png"),
    ("cartpole_uniform_xs.npy",   "cartpole_uniform_pos.png",   "cartpole_uniform_angle.png")
]

# ----------------------------------------------------
# 🎨 스타일 공통 설정 (한 번만 설정하면 됨)
plt.rcParams.update({
    "font.family": "Calibri",
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "font.size": 25,
    "axes.titlesize": 25,
    "axes.labelsize": 25,
})

# ----------------------------------------------------
# 🔄 반복 실행
for xs_name, save_name_pos, save_name_angle in tasks:
    print(f"\nProcessing: {xs_name} ...")

    # 파일 경로 확인
    xs_path = Path(base_dir) / xs_name
    
    if not xs_path.exists():
        print(f"  [Skip] 파일을 찾을 수 없습니다: {xs_path}")
        continue

    # 데이터 로드 및 전처리
    xs = np.load(xs_path)
    
    # 시간 축 길이 조정 (데이터 길이에 맞춤)
    current_T = xs.shape[0]
    ts = np.arange(current_T, dtype=float) * dt
    
    x = xs[:current_T, 0]
    theta_raw = xs[:current_T, 2]
    theta_continuous = np.unwrap(theta_raw)

    # ==========================================
    # 1. Cart Position x(t) 그래프 그리기
    # ==========================================
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()

    # 테두리 굵게
    for spine in ax1.spines.values():
        spine.set_linewidth(2)

    ax1.plot(ts, x, lw=6, color='tab:blue', label='Trajectory')
    ax1.set_ylabel("X [m]")
    ax1.set_xlabel("Time [s]")
    ax1.set_xlim(0, 6)

    # 그리드 및 0점 라인
    ax1.grid(True, ls="--", alpha=0.5, lw=2)
    ax1.axhline(0, color='tab:blue', ls='--', alpha=0.8, lw=3, label='Target')
    ax1.legend(loc='best')
    
    plt.tight_layout()
    
    save_path_pos = Path(base_dir) / save_name_pos
    plt.savefig(save_path_pos, dpi=300, bbox_inches="tight")
    plt.close() # 메모리 해제
    print(f"  -> [Saved] {save_name_pos}")


    # ==========================================
    # 2. Pole Angle θ(t) 그래프 그리기
    # ==========================================
    plt.figure(figsize=(12, 6))
    ax2 = plt.gca()

    # 테두리 굵게
    for spine in ax2.spines.values():
        spine.set_linewidth(2)

    ax2.plot(ts, theta_continuous, lw=6, color='tab:orange', label='Trajectory')
    ax2.set_ylabel(r"$\theta$ [rad]")
    ax2.set_xlabel("Time [s]")
    ax2.set_xlim(0, 6)

    # 그리드
    ax2.grid(True, ls="--", alpha=0.5, lw=2)
    ax2.legend(loc='best')
    
    # 목표 각도 표시 (0, 2pi)
    target_lines = [0, 2*np.pi] 
    label_added = False 

    for t_val in target_lines:
        # 데이터 범위 내에 타겟 값이 있을 때만 선 그리기 (혹은 근처일 때)
        # 데이터가 0~6인데 2pi(6.28)을 그리면 그래프가 눌릴 수 있으니 체크
        min_th, max_th = min(theta_continuous), max(theta_continuous)
        
        # 화면 범위(y축)을 고려하거나, 그냥 항상 그리되 데이터 범위 안에 있을 때만 그릴 수도 있음.
        # 여기서는 "데이터 범위 내에 있거나 근접할 때" 그리는 로직 유지
        if (min_th - 0.5) <= t_val <= (max_th + 0.5): 
            lbl = 'Target' if not label_added else None
            ax2.axhline(t_val, color='tab:orange', ls='--', alpha=0.8, lw=3, label=lbl)
            label_added = True

    plt.tight_layout()
    
    save_path_angle = Path(base_dir) / save_name_angle
    plt.savefig(save_path_angle, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> [Saved] {save_name_angle}")

print("\n모든 작업 완료!")