"""
声纹魔方 - DSP 预处理模块
谱减法降噪：在 AI 处理前去除环境底噪和电流声

算法流程:
  1. 前 N 帧作为噪声估计
  2. 计算噪声功率谱
  3. 对每帧做功率谱减法
  4. 半波整流 + 谱下限
  5. 保留原相位 IFFT 重建
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, NOISE_FRAMES, NOISE_OVERSUBTRACTION, SPECTRAL_FLOOR


def spectral_subtraction(
    audio: np.ndarray,
    sr: int,
    noise_frames: int = NOISE_FRAMES,
    alpha: float = NOISE_OVERSUBTRACTION,
    beta: float = SPECTRAL_FLOOR,
) -> np.ndarray:
    """
    谱减法降噪

    核心公式: |S(ω)|² = |X(ω)|² - α * |N(ω)|²
    半波整流: |S(ω)|² = max(|S(ω)|², β * |X(ω)|²)

    Parameters
    ----------
    audio : np.ndarray
        输入音频信号
    sr : int
        采样率
    noise_frames : int
        用于噪声估计的前导帧数
    alpha : float
        过减因子，越大降噪越强，但可能引入音乐噪声 (推荐 2~4)
    beta : float
        谱下限因子，防止过度抑制 (推荐 0.01~0.1)

    Returns
    -------
    np.ndarray
        降噪后的音频信号
    """
    # STFT 分析
    n_fft = N_FFT
    hop = HOP_LENGTH
    window = np.hanning(n_fft)

    # 分帧
    n_frames = 1 + (len(audio) - n_fft) // hop
    frames = np.zeros((n_frames, n_fft))
    for i in range(n_frames):
        start = i * hop
        frames[i] = audio[start:start + n_fft] * window

    # FFT
    spec = np.fft.rfft(frames, axis=1)
    magnitude = np.abs(spec)
    phase = np.angle(spec)

    # Step 1: 噪声估计（取前 N 帧的平均功率谱）
    actual_noise_frames = min(noise_frames, n_frames)
    noise_power = np.mean(magnitude[:actual_noise_frames] ** 2, axis=0)

    # Step 2: 谱减法
    # |S(ω)|² = |X(ω)|² - α * |N(ω)|²
    clean_magnitude_sq = magnitude ** 2 - alpha * noise_power

    # Step 3: 半波整流 + 谱下限
    # max(|S|², β * |X|²)
    clean_magnitude_sq = np.maximum(clean_magnitude_sq, beta * magnitude ** 2)

    # Step 4: 开方得到幅度谱
    clean_magnitude = np.sqrt(clean_magnitude_sq)

    # Step 5: 保留原相位，IFFT 重建
    clean_spec = clean_magnitude * np.exp(1j * phase)

    # ISTFT（重叠相加法）
    clean_frames = np.fft.irfft(clean_spec, axis=1)
    output = np.zeros(len(audio))
    window_sum = np.zeros(len(audio))

    for i in range(n_frames):
        start = i * hop
        end = start + n_fft
        if end > len(audio):
            break
        output[start:end] += clean_frames[i] * window
        window_sum[start:end] += window ** 2

    # 归一化（消除重叠相加的窗效应）
    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

    # 最终归一化防止削波
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output.astype(np.float32)


def noise_gate(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    简单噪声门：低于阈值的信号直接置零

    Parameters
    ----------
    audio : np.ndarray
        输入音频信号
    threshold : float
        噪声门阈值

    Returns
    -------
    np.ndarray
        处理后的音频
    """
    output = audio.copy()
    output[np.abs(output) < threshold] = 0
    return output.astype(np.float32)
