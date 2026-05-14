"""
声纹魔方 - DSP 预处理模块 (v2)
改进: Wiener滤波平滑, 时间平滑, 更少音乐噪声
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SAMPLE_RATE, N_FFT, HOP_LENGTH,
    NOISE_FRAMES, NOISE_OVERSUBTRACTION, SPECTRAL_FLOOR, SPECTRAL_SMOOTH_FRAMES,
)


def spectral_subtraction(
    audio: np.ndarray,
    sr: int,
    noise_frames: int = NOISE_FRAMES,
    alpha: float = NOISE_OVERSUBTRACTION,
    beta: float = SPECTRAL_FLOOR,
) -> np.ndarray:
    """
    谱减法降噪 (v2): 减少音乐噪声伪影

    改进:
      - α 从 2.0 降到 1.5 (减少过减)
      - β 从 0.01 提到 0.04 (更高的谱下限)
      - 频谱平滑 (相邻帧平均)
      - Wiener 滤波后处理
    """
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

    # 噪声估计 (取前 N 帧中位数，比均值更鲁棒)
    actual_noise_frames = min(noise_frames, n_frames)
    noise_power = np.median(magnitude[:actual_noise_frames] ** 2, axis=0)

    # 频谱平滑 (3帧滑动平均，减少音乐噪声)
    smoothed_mag = np.copy(magnitude)
    for i in range(1, n_frames - 1):
        smoothed_mag[i] = (
            magnitude[i-1] * 0.25 + magnitude[i] * 0.5 + magnitude[i+1] * 0.25
        )

    # 谱减法
    clean_magnitude_sq = smoothed_mag ** 2 - alpha * noise_power

    # 半波整流 + 谱下限
    clean_magnitude_sq = np.maximum(clean_magnitude_sq, beta * smoothed_mag ** 2)

    # Wiener 滤波增益: G = (S² - αN²) / S²
    # 比直接谱减更平滑
    wiener_gain = clean_magnitude_sq / (smoothed_mag ** 2 + 1e-10)
    wiener_gain = np.clip(wiener_gain, beta, 1.0)

    clean_magnitude = magnitude * wiener_gain

    # 保留原相位，IFFT 重建
    clean_spec = clean_magnitude * np.exp(1j * phase)

    # ISTFT
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

    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output.astype(np.float32)


def noise_gate(audio: np.ndarray, threshold: float = 0.01,
               attack_ms: float = 5.0, release_ms: float = 50.0,
               sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    噪声门 (v2): 带 attack/release 的软噪声门

    避免硬切换造成的咔嗒声
    """
    output = audio.copy().astype(np.float64)

    attack_coeff = np.exp(-1.0 / (sr * attack_ms / 1000.0))
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000.0))

    env = 0.0
    gate = 0.0

    for i in range(len(output)):
        level = abs(output[i])

        # 包络检测
        if level > env:
            env = attack_coeff * env + (1 - attack_coeff) * level
        else:
            env = release_coeff * env + (1 - release_coeff) * level

        # 门控增益 (软过渡)
        if env > threshold:
            gate = min(1.0, gate + 0.01)  # 渐开
        else:
            gate = max(0.0, gate - 0.005)  # 渐关

        output[i] *= gate

    return output.astype(np.float32)
