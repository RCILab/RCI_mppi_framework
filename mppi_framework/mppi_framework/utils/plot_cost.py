import numpy as np
import matplotlib.pyplot as plt
import os

# --- 전역 폰트 설정 ---
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# 저장 경로 설정
save_dir = "outputs"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# --- [핵심 변경] 처리할 파일 목록 정의 (입력 파일명, 저장할 파일명) ---
# 튜플 형태: (불러올 .npy 파일 이름, 저장할 .png 파일 이름)
tasks = [
    ("min_costs_cartpole.npy", "cost_gaussian_graph.png"),
    ("min_costs_log_nln_cartpole.npy", "cost_log_nln_graph.png"),
    ("min_costs_uniform_cartpole.npy", "cost_uniform_graph.png")
]

for input_file, output_file in tasks:
    print(f"Processing: {input_file} -> {output_file} ...")
    
    # 데이터 불러오기
    file_path = os.path.join(save_dir, input_file)
    try:
        costs = np.load(file_path)
        print(f"  - 데이터를 성공적으로 로드했습니다: {len(costs)} points")
    except FileNotFoundError:
        print(f"  - [주의] '{input_file}' 파일을 찾을 수 없어 임시 데이터로 그립니다.")
        costs = np.exp(-np.linspace(0, 5, 100)) # 테스트용 임시 데이터

    # 시간 축 생성
    ts = np.arange(len(costs)) * 0.02

    # --- 그래프 그리기 시작 ---
    plt.figure(figsize=(12, 6))

    # 선 굵게
    plt.plot(ts, costs, linewidth=6, color='red')
    
    ax = plt.gca()

    # 테두리(spine) 굵게
    for spine in ax.spines.values():
        spine.set_linewidth(2) 

    # 라벨 크기 키우기
    plt.xlabel("Time [s]", fontsize=25)
    plt.ylabel("Cost", fontsize=25)

    # tick 글씨 크기 키우기
    plt.xticks(fontsize=25, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')

    # grid 투명하게 + 굵게
    plt.grid(
        True, 
        alpha=0.5,
        linewidth=2,
        ls='--'
    )
    plt.xlim(left=0, right=6)
    
    # 레이아웃 조정
    plt.tight_layout()

    # --- PNG 저장 ---
    save_path = os.path.join(save_dir, output_file)
    
    plt.savefig(
        save_path, 
        dpi=300,              # 논문용 고해상도
        bbox_inches='tight',  # 여백 자동 조절
        pad_inches=0.1
    )
    
    print(f"  - [저장 완료] {save_path}\n")
    
    # 메모리 해제를 위해 현재 figure 닫기 (반복문 사용 시 필수)
    plt.close()

print("모든 작업이 완료되었습니다.")