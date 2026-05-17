"""
声纹魔方 - 音频特效处理模块 (v4)
改进: 向量化混响/压缩器(无Python循环), 移除中间归一化
"""

import numpy as np
from scipy.signal import firwin, lfilter, butter, sosfilt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SAMPLE_RATE, FILTER_ORDER,
    TELEPHONE_LOW, TELEPHONE_HIGH,
    REVERB_DELAYS_SEC, REVERB_GAINS, ALLPASS_DELAYS_SEC, ALLPASS_GAIN,
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
    """老式电话带通滤波效果"""
    b = firwin(FILTER_ORDER, [TELEPHONE_LOW, TELEPHONE_HIGH], pass_zero=False, fs=sr)
    filtered = lfilter(b, [1.0], audio)
    return filtered.astype(np.float32)


def comb_reverb(audio: np.ndarray, sr: int, decay: float = 0.5) -> np.ndarray:
    """
    Schroeder 混响 (v4): 全向量化实现，无 Python 循环

    梳状滤波器: y[n] = x[n] + g * y[n-D]  → 用 lfilter(b, a, x) 实现
    全通滤波器: y[n] = -g*x[n] + x[n-D] + g*y[n-D]
    """
    if decay == 0:
        return audio.copy()

    decay = np.clip(decay, 0.0, 1.0)
    audio_f = audio.astype(np.float64)

    # ===== 并行梳状滤波器 (向量化) =====
    comb_out = np.zeros(len(audio_f), dtype=np.float64)
    for delay_sec, gain in zip(REVERB_DELAYS_SEC, REVERB_GAINS):
        delay_samples = max(1, int(delay_sec * sr))
        g = gain * decay * 0.84

        # IIR 滤波器: y[n] = x[n] + g * y[n-D]
        # 等价于 b=[1], a=[1, 0, 0, ..., 0, -g] (长度 D+1)
        b = np.array([1.0])
        a = np.zeros(delay_samples + 1)
        a[0] = 1.0
        a[-1] = -g
        comb_out += lfilter(b, a, audio_f)

    # ===== 串联全通滤波器 (向量化) =====
    result = comb_out
    for delay_sec in ALLPASS_DELAYS_SEC:
        delay_samples = max(1, int(delay_sec * sr))
        g = ALLPASS_GAIN * decay

        # 全通: y[n] = -g*x[n] + x[n-D] + g*y[n-D]
        # 等价于 b=[-g, 0, ..., 0, 1], a=[1, 0, ..., 0, -g] (长度 D+1)
        b_ap = np.zeros(delay_samples + 1)
        b_ap[0] = -g
        b_ap[-1] = 1.0
        a_ap = np.zeros(delay_samples + 1)
        a_ap[0] = 1.0
        a_ap[-1] = -g
        result = lfilter(b_ap, a_ap, result)

    # 混合干湿信号
    wet = decay * 0.6
    dry = 1.0 - wet * 0.4
    output = dry * audio_f + wet * result

    # 防混响过冲
    max_val = np.max(np.abs(output))
    if max_val > 0.99:
        output = output * 0.99 / max_val

    return output.astype(np.float32)


def add_breathiness(audio: np.ndarray, sr: int, amount: float = 0.5) -> np.ndarray:
    """
    气声效果 (v2): 带通噪声 + 包络跟随
    """
    if amount == 0:
        return audio.copy()

    amount = np.clip(amount, 0.0, 1.0)

    # 带通白噪声
    noise = np.random.randn(len(audio))
    nyq = sr / 2
    lo = min(BREATHINESS_BAND_LOW / nyq, 0.95)
    hi = min(BREATHINESS_BAND_HIGH / nyq, 0.99)
    if lo < hi:
        sos = butter(4, [lo, hi], btype="band", output="sos")
        noise = sosfilt(sos, noise)

    # 包络跟随器 (RMS-based)
    frame_len = int(sr * 0.02)
    hop = frame_len // 2
    n_frames = len(audio) // hop
    envelope = np.zeros(len(audio))

    attack_coeff = np.exp(-1.0 / (sr * 0.005))
    release_coeff = np.exp(-1.0 / (sr * 0.050))

    env_val = 0.0
    for i in range(n_frames):
        start = i * hop
        end = min(start + frame_len, len(audio))
        rms = np.sqrt(np.mean(audio[start:end] ** 2))

        if rms > env_val:
            env_val = attack_coeff * env_val + (1 - attack_coeff) * rms
        else:
            env_val = release_coeff * env_val + (1 - release_coeff) * rms

        envelope[start:end] = env_val

    modulated_noise = noise * envelope * 0.3
    output = (1.0 - amount) * audio + amount * modulated_noise

    return output.astype(np.float32)


def comb_filter_metallic(audio: np.ndarray, sr: int, freq: float = 1000.0) -> np.ndarray:
    """梳状滤波器 - 实现金属质感"""
    delay_samples = max(1, int(sr / freq))
    g = 0.7

    # 用 lfilter 替代手动延迟
    b = np.array([1.0])
    a = np.zeros(delay_samples + 1)
    a[0] = 1.0
    a[-1] = -g
    output = lfilter(b, a, audio.astype(np.float64))

    return output.astype(np.float32)


def noise_gate(audio: np.ndarray, sr: int,
               threshold_db: float = -50,
               attack_ms: float = 5,
               release_ms: float = 100,
               knee_db: float = 6) -> np.ndarray:
    """
    软噪声门 (v1): RMS 包络跟随 + 软拐点增益

    当信号 RMS 低于阈值时平滑衰减增益，抑制静音段的噪声底。
    拐点区域用平方律过渡避免硬开关咔哒声。
    """
    threshold = 10 ** (threshold_db / 20.0)
    knee_width = 10 ** ((threshold_db - knee_db) / 20.0)

    # RMS 包络检测 (20ms 帧, 75% 重叠)
    frame_len = int(sr * 0.02)
    hop = frame_len // 4
    n_frames = max(1, (len(audio) - frame_len) // hop)

    # 逐帧 RMS → 插值包络
    rms_env = np.zeros(len(audio))
    for i in range(n_frames):
        start = i * hop
        end = min(start + frame_len, len(audio))
        rms = np.sqrt(np.mean(audio[start:end] ** 2))
        rms_env[start:end] = rms

    # attack/release 平滑
    attack = np.exp(-1.0 / (sr * attack_ms / 1000.0))
    release = np.exp(-1.0 / (sr * release_ms / 1000.0))
    smooth_env = np.zeros_like(rms_env)
    env_val = 0.0
    for i in range(len(rms_env)):
        if rms_env[i] > env_val:
            env_val = attack * env_val + (1 - attack) * rms_env[i]
        else:
            env_val = release * env_val + (1 - release) * rms_env[i]
        smooth_env[i] = env_val

    # 计算增益: 低于阈值时衰减, 拐区平方律过渡
    gain = np.ones(len(audio))
    gate_region = smooth_env < threshold
    if np.any(gate_region):
        full_gate = smooth_env < knee_width
        gain[full_gate] = (smooth_env[full_gate] / threshold) ** 2 * 0.01
        knee_region = (~full_gate) & gate_region
        if np.any(knee_region):
            t = (smooth_env[knee_region] - knee_width) / (threshold - knee_width)
            gain[knee_region] = t ** 2

    return (audio * gain).astype(np.float32)


def soft_knee_compressor(audio: np.ndarray, sr: int,
                         threshold_db: float = COMPRESSOR_THRESHOLD,
                         ratio: float = COMPRESSOR_RATIO,
                         attack: float = COMPRESSOR_ATTACK,
                         release: float = COMPRESSOR_RELEASE,
                         knee_db: float = COMPRESSOR_KNEE) -> np.ndarray:
    """
    软拐点动态压缩器 (v4): 向量化包络检测 + 增益计算
    """
    threshold = 10 ** (threshold_db / 20.0)
    knee = 10 ** (knee_db / 20.0)

    attack_coeff = np.exp(-1.0 / (sr * attack))
    release_coeff = np.exp(-1.0 / (sr * release))

    # 包络检测 (仍需逐样本，但增益计算向量化)
    level = np.abs(audio).astype(np.float64)
    env = np.zeros_like(level)
    env[0] = level[0]
    for i in range(1, len(level)):
        if level[i] > env[i-1]:
            env[i] = attack_coeff * env[i-1] + (1 - attack_coeff) * level[i]
        else:
            env[i] = release_coeff * env[i-1] + (1 - release_coeff) * level[i]

    # 向量化增益计算
    gain_db = np.zeros_like(env)
    mask_quiet = env < threshold / knee
    mask_loud = env > threshold * knee
    mask_knee = ~mask_quiet & ~mask_loud

    # 安静区域: 无压缩
    gain_db[mask_quiet] = 0.0

    # 压缩区域
    env_loud = env[mask_loud]
    if len(env_loud) > 0:
        level_db = 20 * np.log10(env_loud / threshold + 1e-10)
        gain_db[mask_loud] = threshold_db + (level_db - threshold_db) / ratio - level_db

    # 软拐点区域
    env_knee = env[mask_knee]
    if len(env_knee) > 0:
        x = 20 * np.log10(env_knee / threshold + 1e-10)
        gain_db[mask_knee] = (1.0 / ratio - 1.0) * (x - threshold_db + knee_db / 2) ** 2 / (2 * knee_db)

    gain = 10 ** (gain_db / 20.0)
    output = audio * gain

    return output.astype(np.float32)


def harmonics_exciter(audio: np.ndarray, sr: int, amount: float = 0.15,
                      freq_min: float = 2000, freq_max: float = 8000) -> np.ndarray:
    """
    谐波激励器 — 增加高频谐波，提升语音清晰度
    """
    if amount == 0:
        return audio.copy()

    amount = np.clip(amount, 0.0, 0.5)

    nyq = sr / 2
    lo = min(freq_min / nyq, 0.95)
    hi = min(freq_max / nyq, 0.99)
    sos = butter(4, [lo, hi], btype="band", output="sos")
    high_band = sosfilt(sos, audio)

    harmonics = np.tanh(high_band * 3.0) * 0.3
    output = audio + amount * harmonics

    return output.astype(np.float32)
