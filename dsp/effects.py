"""
声纹魔方 - 音频特效处理模块 (v2)
改进: 正反馈混响, 带通气声, 动态压缩, 谐波增强
"""

import numpy as np
from scipy.signal import firwin, lfilter, butter, sosfilt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SAMPLE_RATE, FILTER_ORDER,
    TELEPHONE_LOW, TELEPHONE_HIGH,
    REVERB_DELAYS, REVERB_GAINS, ALLPASS_DELAYS, ALLPASS_GAIN,
    BREATHINESS_BAND_LOW, BREATHINESS_BAND_HIGH,
    COMPRESSOR_THRESHOLD, COMPRESSOR_RATIO, COMPRESSOR_ATTACK,
    COMPRESSOR_RELEASE, COMPRESSOR_KNEE,
)


def ring_modulate(audio: np.ndarray, sr: int, depth: float = 0.5,
                  mod_freq: float = 50.0) -> np.ndarray:
    """环形调制（幅度调制）- 实现机械音/金属音效果"""
    if depth == 0:
        return audio.copy()

    depth = np.clip(depth, 0.0, 1.0)
    n = np.arange(len(audio))
    modulator = np.sin(2.0 * np.pi * mod_freq * n / sr)
    output = audio * (1.0 - depth + depth * modulator)

    return output.astype(np.float32)


def telephone_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    """老式电话带通滤波效果 (v2: 更高阶数，更陡峭滚降)"""
    b = firwin(FILTER_ORDER, [TELEPHONE_LOW, TELEPHONE_HIGH], pass_zero=False, fs=sr)
    filtered = lfilter(b, [1.0], audio)
    return filtered.astype(np.float32)


def comb_reverb(audio: np.ndarray, sr: int, decay: float = 0.5) -> np.ndarray:
    """
    Schroeder 混响 (v2): 正确的递归反馈梳状滤波器 + 全通扩散

    改进: 使用真正的递归反馈 y[n] = x[n] + g * y[n-D]
    而不是简单的延迟线
    """
    if decay == 0:
        return audio.copy()

    decay = np.clip(decay, 0.0, 1.0)

    # ===== 并行梳状滤波器 (带递归反馈) =====
    comb_out = np.zeros(len(audio), dtype=np.float64)
    for delay, gain in zip(REVERB_DELAYS, REVERB_GAINS):
        actual_delay = int(delay * sr / 44100)
        g = gain * decay * 0.84  # 控制反馈量，防止发散

        # 递归梳状滤波器: y[n] = x[n] + g * y[n-D]
        y = np.zeros(len(audio), dtype=np.float64)
        for n in range(actual_delay, len(audio)):
            y[n] = audio[n] + g * y[n - actual_delay]
        comb_out += y

    # ===== 串联全通滤波器 =====
    result = comb_out.copy()
    for delay in ALLPASS_DELAYS:
        actual_delay = int(delay * sr / 44100)
        g = ALLPASS_GAIN * decay
        # 全通: y[n] = -g*x[n] + x[n-D] + g*y[n-D]
        y = np.zeros(len(result), dtype=np.float64)
        for n in range(actual_delay, len(result)):
            y[n] = -g * result[n] + result[n - actual_delay] + g * y[n - actual_delay]
        result = y

    # 混合干湿信号
    wet = decay * 0.6
    dry = 1.0 - wet * 0.4
    output = dry * audio.astype(np.float64) + wet * result

    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output.astype(np.float32)


def add_breathiness(audio: np.ndarray, sr: int, amount: float = 0.5) -> np.ndarray:
    """
    气声效果 (v2): 带通噪声 + 精确包络跟随

    改进:
      - 带通白噪声 (1.5-8kHz) 而非全频白噪声
      - 带 attack/release 的包络跟随器
      - 噪声不依赖原始信号振幅，避免"糊"
    """
    if amount == 0:
        return audio.copy()

    amount = np.clip(amount, 0.0, 1.0)

    # 带通白噪声
    noise = np.random.randn(len(audio))
    # 带通滤波 (1.5kHz - 8kHz)
    nyq = sr / 2
    lo = min(BREATHINESS_BAND_LOW / nyq, 0.95)
    hi = min(BREATHINESS_BAND_HIGH / nyq, 0.99)
    if lo < hi:
        sos = butter(4, [lo, hi], btype="band", output="sos")
        noise = sosfilt(sos, noise)

    # 包络跟随器 (RMS-based, 带 attack/release)
    frame_len = int(sr * 0.02)  # 20ms 帧
    hop = frame_len // 2
    n_frames = len(audio) // hop
    envelope = np.zeros(len(audio))

    attack_coeff = np.exp(-1.0 / (sr * 0.005))   # 5ms attack
    release_coeff = np.exp(-1.0 / (sr * 0.050))   # 50ms release

    env_val = 0.0
    for i in range(n_frames):
        start = i * hop
        end = min(start + frame_len, len(audio))
        rms = np.sqrt(np.mean(audio[start:end] ** 2))

        # attack/release
        if rms > env_val:
            env_val = attack_coeff * env_val + (1 - attack_coeff) * rms
        else:
            env_val = release_coeff * env_val + (1 - release_coeff) * rms

        envelope[start:end] = env_val

    # 平滑噪声并调制
    modulated_noise = noise * envelope * 0.3  # 控制噪声电平

    # 混合
    output = (1.0 - amount) * audio + amount * modulated_noise

    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output.astype(np.float32)


def comb_filter_metallic(audio: np.ndarray, sr: int, freq: float = 1000.0) -> np.ndarray:
    """梳状滤波器 - 实现金属质感"""
    delay_samples = int(sr / freq)
    g = 0.7

    delayed = np.zeros_like(audio)
    if delay_samples < len(audio):
        delayed[delay_samples:] = audio[:-delay_samples]

    output = audio + g * delayed

    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output.astype(np.float32)


def soft_knee_compressor(audio: np.ndarray, sr: int,
                         threshold_db: float = COMPRESSOR_THRESHOLD,
                         ratio: float = COMPRESSOR_RATIO,
                         attack: float = COMPRESSOR_ATTACK,
                         release: float = COMPRESSOR_RELEASE,
                         knee_db: float = COMPRESSOR_KNEE) -> np.ndarray:
    """
    软拐点动态压缩器

    让安静的声音更响，响的声音更平滑，增加语音的"存在感"和清晰度
    """
    threshold = 10 ** (threshold_db / 20.0)
    knee = 10 ** (knee_db / 20.0)

    attack_coeff = np.exp(-1.0 / (sr * attack))
    release_coeff = np.exp(-1.0 / (sr * release))

    output = np.zeros_like(audio, dtype=np.float64)
    env = 0.0

    for i in range(len(audio)):
        level = abs(audio[i])

        # 包络检测 (attack/release)
        if level > env:
            env = attack_coeff * env + (1 - attack_coeff) * level
        else:
            env = release_coeff * env + (1 - release_coeff) * level

        # 计算增益衰减 (软拐点)
        if env < threshold / knee:
            gain_db = 0.0
        elif env > threshold * knee:
            gain_db = threshold_db + (20 * np.log10(env / threshold + 1e-10) - threshold_db) / ratio - 20 * np.log10(env / threshold + 1e-10)
        else:
            # 软拐点区域: 平滑过渡
            x = 20 * np.log10(env / threshold + 1e-10)
            gain_db = (1.0 / ratio - 1.0) * (x - threshold_db + knee_db/2) ** 2 / (2 * knee_db)

        gain = 10 ** (gain_db / 20.0)
        output[i] = audio[i] * gain

    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output.astype(np.float32)


def harmonics_exciter(audio: np.ndarray, sr: int, amount: float = 0.15,
                      freq_min: float = 2000, freq_max: float = 8000) -> np.ndarray:
    """
    谐波激励器 — 增加高频谐波，提升语音清晰度和"存在感"

    防止处理后声音发闷的关键效果
    """
    if amount == 0:
        return audio.copy()

    amount = np.clip(amount, 0.0, 0.5)

    # 提取高频成分
    nyq = sr / 2
    lo = min(freq_min / nyq, 0.95)
    hi = min(freq_max / nyq, 0.99)
    sos = butter(4, [lo, hi], btype="band", output="sos")
    high_band = sosfilt(sos, audio)

    # 软饱和产生谐波 (tanh soft clipping)
    harmonics = np.tanh(high_band * 3.0) * 0.3

    # 混合回原始信号
    output = audio + amount * harmonics

    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output.astype(np.float32)
