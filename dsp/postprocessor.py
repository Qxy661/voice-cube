"""
声纹魔方 - 后置 DSP 处理模块 (v2)
改进: 正确的搁架滤波器, 正确的Freeverb混响, LUFS响度归一化
"""

import numpy as np
from scipy.signal import iirpeak, lfilter, butter, sosfilt, iirnotch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SAMPLE_RATE, REVERB_DELAYS, REVERB_GAINS, ALLPASS_DELAYS, ALLPASS_GAIN,
    TARGET_LUFS, EQ_CROSSOVER_LOW, EQ_CROSSOVER_HIGH,
)


def _low_shelf_coeffs(fc, gain_db, sr, Q=0.707):
    """设计低频搁架滤波器 (biquad)"""
    w0 = 2 * np.pi * fc / sr
    A = 10 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2 * Q)

    b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
    a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])


def _high_shelf_coeffs(fc, gain_db, sr, Q=0.707):
    """设计高频搁架滤波器 (biquad)"""
    w0 = 2 * np.pi * fc / sr
    A = 10 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2 * Q)

    b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
    a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])


def parametric_eq(
    audio: np.ndarray,
    sr: int,
    bass_boost: float = 0,
    treble_boost: float = 0,
    mid_freq: float = 1000,
    mid_gain: float = 0,
) -> np.ndarray:
    """
    三段参量均衡器 (v2): 正确的搁架滤波器实现

    改进:
      - 低频搁架用 biquad 而非 iirpeak
      - 高频搁架用 biquad 而非 iirpeak
      - 中频峰值保留 iirpeak
    """
    output = audio.copy().astype(np.float64)

    # 低频搁架 (250 Hz 以下)
    if bass_boost != 0:
        b, a = _low_shelf_coeffs(EQ_CROSSOVER_LOW, bass_boost, sr)
        output = lfilter(b, a, output)

    # 中频峰值
    if mid_gain != 0:
        w0 = mid_freq / (sr / 2)
        w0 = np.clip(w0, 0.01, 0.99)
        b, a = iirpeak(w0, Q=2.0)
        gain_linear = 10 ** (mid_gain / 20.0)
        output = lfilter(b * gain_linear, a, output)

    # 高频搁架 (4000 Hz 以上)
    if treble_boost != 0:
        b, a = _high_shelf_coeffs(EQ_CROSSOVER_HIGH, treble_boost, sr)
        output = lfilter(b, a, output)

    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output.astype(np.float32)


def apply_reverb(audio: np.ndarray, sr: int, wet: float = 0.3) -> np.ndarray:
    """
    人工混响后处理 (v2): Freeverb 风格，正确的递归反馈

    改进:
      - 梳状滤波器使用递归反馈而非简单延迟
      - 全通滤波器使用递归而非np.roll
      - 更自然的干湿混合
    """
    if wet == 0:
        return audio.copy()

    wet = np.clip(wet, 0.0, 1.0)
    audio_f = audio.astype(np.float64)

    # 并行梳状滤波器 (带递归反馈)
    comb_out = np.zeros(len(audio_f), dtype=np.float64)
    for delay, gain in zip(REVERB_DELAYS, REVERB_GAINS):
        actual_delay = int(delay * sr / 44100)
        g = gain * wet * 0.84

        y = np.zeros(len(audio_f), dtype=np.float64)
        for n in range(actual_delay, len(audio_f)):
            y[n] = audio_f[n] + g * y[n - actual_delay]
        comb_out += y

    # 串联全通滤波器
    result = comb_out.copy()
    for delay in ALLPASS_DELAYS:
        actual_delay = int(delay * sr / 44100)
        g = ALLPASS_GAIN * wet
        y = np.zeros(len(result), dtype=np.float64)
        for n in range(actual_delay, len(result)):
            y[n] = -g * result[n] + result[n - actual_delay] + g * y[n - actual_delay]
        result = y

    # 干湿混合
    dry = 1.0 - wet * 0.5
    output = dry * audio_f + wet * result

    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output.astype(np.float32)


def normalize_loudness(audio: np.ndarray, sr: int,
                       target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """
    LUFS 响度归一化 — 让输出音量一致，不会忽大忽小

    使用简化的 ITU-R BS.1770 算法:
    1. K-weighting 滤波
    2. 块级能量测量
    3. 计算 integrated loudness
    4. 增益调整到目标 LUFS
    """
    if len(audio) < sr * 0.1:  # 太短的音频跳过
        return audio

    # K-weighting 近似 (高shelf + 高通)
    # Stage 1: 高频搁架 (~4kHz +4dB)
    b_shelf, a_shelf = _high_shelf_coeffs(1681.974450955533, 3.999843853973347, sr, Q=0.7075504)
    audio_kw = lfilter(b_shelf, a_shelf, audio.astype(np.float64))
    # Stage 2: 高通 (~38Hz)
    sos_hp = butter(2, 38.13547087602444 / (sr/2), btype='high', output='sos')
    audio_kw = sosfilt(sos_hp, audio_kw)

    # 块级能量 (400ms 块, 75% 重叠)
    block_len = int(sr * 0.4)
    hop = block_len // 4
    n_blocks = max(1, (len(audio_kw) - block_len) // hop)

    block_loudness = []
    for i in range(n_blocks):
        start = i * hop
        block = audio_kw[start:start + block_len]
        mean_sq = np.mean(block ** 2)
        if mean_sq > 0:
            block_loudness.append(-0.691 + 10 * np.log10(mean_sq + 1e-20))

    if not block_loudness:
        return audio

    # Integrated loudness (门限 -70 LUFS)
    block_loudness = np.array(block_loudness)
    above_gate = block_loudness > -70
    if np.sum(above_gate) == 0:
        return audio

    integrated = -0.691 + 10 * np.log10(
        np.mean(10 ** ((block_loudness[above_gate] + 0.691) / 10)) + 1e-20
    )

    # 计算增益
    gain_db = target_lufs - integrated
    gain = 10 ** (gain_db / 20.0)

    output = audio * gain

    # 防止削波
    max_val = np.max(np.abs(output))
    if max_val > 0.99:
        output = output * 0.99 / max_val

    return output.astype(np.float32)
