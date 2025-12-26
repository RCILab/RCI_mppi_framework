import numpy as np
import matplotlib.pyplot as plt
import os

def plot_noise_hist(noise_path="sample_noise_t0.npy", control_idx=1, save_path="noise_hist_w.png"):
    """
    noise_path : npy 파일 경로
    control_idx : 0=velocity(v), 1=angular velocity(w)
    """
    if not os.path.exists(noise_path):
        print(f"[Error] File not found: {noise_path}")
        return
    
    noise = np.load(noise_path)  # shape: [B, horizon, control_dim]
    print(f"Loaded noise shape: {noise.shape}")

    # w(control_idx=1) 값들만 flatten
    w_noise = noise[..., control_idx].reshape(-1)

    plt.figure(figsize=(6,4))
    plt.hist(w_noise, bins=100, density=True, alpha=0.7, color="royalblue", edgecolor="black")
    plt.xlabel("w noise")
    plt.ylabel("Density")
    plt.title("Noise Distribution of w (t=0)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Saved] Histogram figure: {save_path}")

if __name__ == "__main__":
    plot_noise_hist(noise_path="outputs/sample_noise_gaussian.npy", save_path="outputs/sample_noise_gaussian.png")
    plot_noise_hist(noise_path="outputs/sample_noise_log_nln.npy", save_path="outputs/sample_noise_log_nln.png")
    plot_noise_hist(noise_path="outputs/sample_noise_uniform.npy", save_path="outputs/sample_noise_uniform.png")
