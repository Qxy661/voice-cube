"""
声纹魔方 - 音频特效处理模块
包含环形调制、带通滤波、梳状滤波、混响、气声等经典 DSP 特效

所有特效均为纯 DSP 实现，无深度学习黑盒
"""

import numpy as np
from scipy.signal import firwin, lfilter, butter

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SAMPLE_RATE, FILTER_ORDER,
    TELEPHONE_LOW, TELEPHONE_HIGH,
    REVERB_DELAYS, REVERB_GAINS, ALLPASS_DELAYS, ALLPASS_GAIN,
)


def ring_modulate(audio: np.ndarray, sr: int, depth: float = 0.5, mod_freq: float = 50.0) -> np.ndarray:
    """
    环形调制（幅度调制）- 实现机械音/金属音效果

    公式: y[n] = x[n] * (1 - depth + depth * sin(2π * f_mod * n / fs))

    Parameters
    ----------
    audio : np.ndarray
        输入音频信号
    sr : int
        采样率
    depth : float
        调制深度 0.0 ~ 1.0
    mod_freq : float
        调制频率 (Hz)，推荐 30~300

    Returns
    -------
    np.ndarray
        环形调制后的音频
    """
    if depth == 0:
        return audio.copy()

    depth = np.clip(depth, 0.0, 1.0)
    n = np.arange(len(audio))

    # 生成调制信号
    modulator = np.sin(2.0 * np.pi * mod_freq * n / sr)

    # 幅度调制：原始信号 * (直流偏移 + 调制分量)
    # 直流偏移保证原始信号不被完全抑制
    output = audio * (1.0 - depth + depth * modulator)

    return output.astype(np.float32)


def telephone_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    老式电话带通滤波效果
    使用 FIR 带通滤波器，只保留 300-3400Hz 频率范围

    Parameters
    ----------
    audio : np.ndarray
        输入音频信号
    sr : int
        采样率

    Returns
    -------
    np.ndarray
        带通滤波后的音频
    """
    # 设计 FIR 带通滤波器
    b = firwin(FILTER_ORDER, [TELEPHONE_LOW, TELEPHONE_HIGH], pass_zero=False, fs=sr)
    filtered = lfilter(b, [1.0], audio)

    return filtered.astype(np.float32)


def comb_reverb(audio: np.ndarray, sr: int, decay: float = 0.5) -> np.ndarray:
    """
    Schroeder 混响：4个并行梳状滤波器 + 2个串联全通滤波器

    Parameters
    ----------
    audio : np.ndarray
        输入音频信号
    sr : int
        采样率
    decay : float
        混响强度 0.0 ~ 1.0

    Returns
    -------
    np.ndarray
        添加混响后的音频
    """
    if decay == 0:
        return audio.copy()

    decay = np.clip(decay, 0.0, 1.0)

    # ===== 并行梳状滤波器 =====
    comb_out = np.zeros_like(audio, dtype=np.float64)
    for delay, gain in zip(REVERB_DELAYS, REVERB_GAINS):
        # 调整延迟采样数（按采样率缩放）
        actual_delay = int(delay * sr / 44100)
        delayed = np.zeros_like(audio, dtype=np.float64)
        if actual_delay < len(audio):
            delayed[actual_delay:] = audio[:-actual_delay]
        comb_out += delayed * gain * decay

    # ===== 串联全通滤波器 =====
    result = comb_out
    for delay in ALLPASS_DELAYS:
        actual_delay = int(delay * sr / 44100)
        g = ALLPASS_GAIN * decay
        delayed = np.zeros_like(result)
        if actual_delay < len(result):
            delayed[actual_delay:] = result[:-actual_delay]
        # 全通滤波器: y[n] = -g*x[n] + x[n-D] + g*y[n-D]
        result = -g * result + delayed + g * np.roll(result, actual_delay)

    # 混合原始信号和混响信号
    wet = decay
    dry = 1.0 - decay * 0.3
    output = dry * audio + wet * result

    # 归一化
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output.astype(np.float32)


def add_breathiness(audio: np.ndarray, sr: int, amount: float = 0.5) -> np.ndarray:
    """
    气声效果：在原始信号中混入滤波后的白噪声，模拟气声

    Parameters
    ----------
    audio : np.ndarray
        输入音频信号
    sr : int
        采样率
    amount : float
        气声混合比例 0.0 ~ 1.0

    Returns
    -------
    np.ndarray
        添加气声后的音频
    """
    if amount == 0:
        return audio.copy()

    amount = np.clip(amount, 0.0, 1.0)

    # 生成与信号等长的白噪声
    noise = np.random.randn(len(audio)) * 0.02

    # 低通滤波噪声（气声主要在高频）
    b, a = butter(4, 4000 / (sr / 2), btype="low")
    filtered_noise = lfilter(b, a, noise)

    # 用原始信号的包络调制噪声幅度
    envelope = np.abs(audio)
    # 平滑包络
    from scipy.ndimage import uniform_filter1d
    envelope = uniform_filter1d(envelope, size=int(sr * 0.02))
    modulated_noise = filtered_noise * envelope

    # 混合
    output = (1.0 - amount) * audio + amount * modulated_noise

    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output.astype(np.float32)


def comb_filter_metallic(audio: np.ndarray, sr: int, freq: float = 1000.0) -> np.ndarray:
    """
    梳状滤波器 - 实现金属质感

    公式: y[n] = x[n] + g * x[n - M]
    其中 M = sr / freq

    Parameters
    ----------
    audio : np.ndarray
        输入音频信号
    sr : int
        采样率
    freq : float
        梳状滤波频率 (Hz)

    Returns
    -------
    np.ndarray
        梳状滤波后的音频
    """
    delay_samples = int(sr / freq)
    g = 0.7  # 反馈增益

    delayed = np.zeros_like(audio)
    if delay_samples < len(audio):
        delayed[delay_samples:] = audio[:-delay_samples]

    output = audio + g * delayed

    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output.astype(np.float32)
