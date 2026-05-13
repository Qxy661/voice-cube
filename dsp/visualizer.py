"""
声纹魔方 - 可视化模块
生成波形对比图、频谱对比图、语谱图
用于展示 DSP 处理前后的声学特征变化
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端，适合 Web 服务
import matplotlib.pyplot as plt
import librosa
import librosa.display

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS


# 全局样式
plt.rcParams.update({
    "figure.facecolor": "#0F0E17",
    "axes.facecolor": "#1A1A2E",
    "axes.edgecolor": "#333355",
    "axes.labelcolor": "#EAEAEA",
    "text.color": "#EAEAEA",
    "xtick.color": "#888888",
    "ytick.color": "#888888",
    "grid.color": "#2A2A4A",
    "font.size": 10,
})


def plot_waveform_comparison(
    original: np.ndarray, processed: np.ndarray, sr: int, labels: tuple = ("原声", "处理后")
) -> plt.Figure:
    """
    时域波形对比图

    Parameters
    ----------
    original, processed : np.ndarray
        原始和处理后的音频信号
    sr : int
        采样率
    labels : tuple
        两个子图的标题

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

    t_orig = np.arange(len(original)) / sr
    t_proc = np.arange(len(processed)) / sr

    ax1.plot(t_orig, original, color="#6C63FF", linewidth=0.5, alpha=0.9)
    ax1.set_ylabel("振幅")
    ax1.set_title(labels[0], fontsize=11, fontweight="bold")
    ax1.set_ylim(-1, 1)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_proc, processed, color="#FF6584", linewidth=0.5, alpha=0.9)
    ax2.set_ylabel("振幅")
    ax2.set_xlabel("时间 (s)")
    ax2.set_title(labels[1], fontsize=11, fontweight="bold")
    ax2.set_ylim(-1, 1)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_spectrum_comparison(
    original: np.ndarray, processed: np.ndarray, sr: int, labels: tuple = ("原声频谱", "处理后频谱")
) -> plt.Figure:
    """
    频谱对比图（FFT 幅度谱）

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    for ax, signal, color, label in [
        (ax1, original, "#6C63FF", labels[0]),
        (ax2, processed, "#FF6584", labels[1]),
    ]:
        # FFT
        fft_result = np.fft.rfft(signal)
        magnitude = np.abs(fft_result)
        freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)

        # 转 dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        ax.plot(freqs, magnitude_db, color=color, linewidth=0.8, alpha=0.9)
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("幅度 (dB)")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlim(0, sr / 2)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_spectrogram_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    labels: tuple = ("原声语谱图", "处理后语谱图"),
    third: np.ndarray = None,
    third_label: str = "AI克隆后",
) -> plt.Figure:
    """
    语谱图对比（支持 2 或 3 列并排）

    Parameters
    ----------
    third : np.ndarray, optional
        第三段音频（用于 AI 克隆后的对比）
    third_label : str
        第三列标题

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_cols = 3 if third is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 2:
        axes = [axes[0], axes[1]]

    signals = [original, processed]
    signal_labels = [labels[0], labels[1]]
    colors = ["magma", "magma"]

    if third is not None:
        signals.append(third)
        signal_labels.append(third_label)
        colors.append("magma")

    for ax, sig, label, cmap in zip(axes, signals, signal_labels, colors):
        S = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        S_dB = librosa.power_to_db(S, ref=np.max)

        img = librosa.display.specshow(
            S_dB, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel",
            ax=ax, cmap=cmap,
        )
        ax.set_title(label, fontsize=11, fontweight="bold")
        fig.colorbar(img, ax=ax, format="%+2.0f dB", shrink=0.8)

    fig.tight_layout()
    return fig


def plot_f0_contour(f0: np.ndarray, sr: int, title: str = "基频 (F0) 轮廓线") -> plt.Figure:
    """
    基频轮廓线图

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 2.5))

    times = np.arange(len(f0)) * HOP_LENGTH / sr
    ax.plot(times, f0, color="#6C63FF", linewidth=1.5, alpha=0.9)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("频率 (Hz)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_pipeline_flow(steps: list) -> plt.Figure:
    """
    处理流水线可视化

    Parameters
    ----------
    steps : list of dict
        每个步骤包含 {"name": str, "status": "done"|"active"|"pending"}

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")

    status_colors = {"done": "#4CAF50", "active": "#FF9800", "pending": "#555555"}

    for i, step in enumerate(steps):
        color = status_colors.get(step.get("status", "pending"), "#555555")
        circle = plt.Circle((i + 0.5, 0.5), 0.3, color=color, ec="white", linewidth=1.5)
        ax.add_patch(circle)
        ax.text(i + 0.5, 0.5, step.get("icon", ""), ha="center", va="center", fontsize=16)
        ax.text(i + 0.5, 0.05, step["name"], ha="center", va="center", fontsize=8, color="#EAEAEA")

        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 0.8, 0.5), xytext=(i + 1.2, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5))

    fig.tight_layout()
    return fig
