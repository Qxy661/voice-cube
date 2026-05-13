"""
声纹魔方 - 后置 DSP 处理模块
在 AI 输出后进行均衡和混响润色，让生成的声音更自然

包含:
  1. 参量均衡器 (Parametric EQ)
  2. 人工混响 (Reverb)
"""

import numpy as np
from scipy.signal import iirpeak, lfilter, iirnotch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, REVERB_DELAYS, REVERB_GAINS, ALLPASS_DELAYS, ALLPASS_GAIN


def parametric_eq(
    audio: np.ndarray,
    sr: int,
    bass_boost: float = 0,
    treble_boost: float = 0,
    mid_freq: float = 1000,
    mid_gain: float = 0,
) -> np.ndarray:
    """
    三段参量均衡器

    Parameters
    ----------
    audio : np.ndarray
        输入音频信号
    sr : int
        采样率
    bass_boost : float
        低频增益 (dB)，正数增强，负数衰减
    treble_boost : float
        高频增益 (dB)
    mid_freq : float
        中频中心频率 (Hz)
    mid_gain : float
        中频增益 (dB)

    Returns
    -------
    np.ndarray
        均衡后的音频
    """
    output = audio.copy()

    # 低频搁架 (200 Hz 以下)
    if bass_boost != 0:
        freq = 200.0
        w0 = freq / (sr / 2)
        w0 = min(w0, 0.99)
        b, a = iirpeak(w0, Q=1.0)
        gain_linear = 10 ** (bass_boost / 20.0)
        if bass_boost > 0:
            output = lfilter(b * gain_linear, a, output)
        else:
            output = lfilter(b / abs(gain_linear), a, output)

    # 中频峰值
    if mid_gain != 0:
        w0 = mid_freq / (sr / 2)
        w0 = min(w0, 0.99)
        b, a = iirpeak(w0, Q=2.0)
        gain_linear = 10 ** (mid_gain / 20.0)
        output = lfilter(b * gain_linear, a, output)

    # 高频搁架 (4000 Hz 以上)
    if treble_boost != 0:
        freq = 4000.0
        w0 = freq / (sr / 2)
        w0 = min(w0, 0.99)
        b, a = iirpeak(w0, Q=1.0)
        gain_linear = 10 ** (treble_boost / 20.0)
        if treble_boost > 0:
            output = lfilter(b * gain_linear, a, output)
        else:
            output = lfilter(b / abs(gain_linear), a, output)

    # 归一化
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output.astype(np.float32)


def apply_reverb(audio: np.ndarray, sr: int, wet: float = 0.3) -> np.ndarray:
    """
    人工混响后处理（Freeverb 简化版）

    Parameters
    ----------
    audio : np.ndarray
        输入音频信号
    sr : int
        采样率
    wet : float
        湿信号比例 0.0 ~ 1.0

    Returns
    -------
    np.ndarray
        添加混响后的音频
    """
    if wet == 0:
        return audio.copy()

    wet = np.clip(wet, 0.0, 1.0)

    # 并行梳状滤波器
    comb_out = np.zeros(len(audio), dtype=np.float64)
    for delay, gain in zip(REVERB_DELAYS, REVERB_GAINS):
        actual_delay = int(delay * sr / 44100)
        delayed = np.zeros(len(audio), dtype=np.float64)
        if actual_delay < len(audio):
            delayed[actual_delay:] = audio[:-actual_delay]
        # 带反馈的梳状滤波器
        feedback = delayed
        for _ in range(3):  # 简化反馈
            feedback = np.zeros(len(audio), dtype=np.float64)
            if actual_delay < len(feedback):
                feedback[actual_delay:] = (delayed + feedback * gain * 0.5)[:-actual_delay] if actual_delay > 0 else delayed
        comb_out += delayed * gain

    # 串联全通滤波器
    result = comb_out.copy()
    for delay in ALLPASS_DELAYS:
        actual_delay = int(delay * sr / 44100)
        g = ALLPASS_GAIN
        delayed = np.zeros(len(result))
        if actual_delay < len(result):
            delayed[actual_delay:] = result[:-actual_delay]
        result = -g * result + delayed + g * np.roll(result, actual_delay)

    # 混合干湿信号
    dry = 1.0 - wet * 0.5
    output = dry * audio.astype(np.float64) + wet * result

    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output.astype(np.float32)
