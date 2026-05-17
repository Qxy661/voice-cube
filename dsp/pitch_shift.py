"""
声纹魔方 - 变调引擎 (v7)
双模式: 小偏移用 polyphase resample (保清音, 无Gibbs振铃),
         大偏移用 librosa phase vocoder + polyphase resample
"""

import numpy as np
import librosa
from math import gcd


def _polyphase_resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """
    Polyphase 重采样: 避免 FFT 重采样在语音暂态上的 Gibbs 振铃

    使用 scipy.signal.resample_poly, 通过整数 up/down 因子实现。
    """
    from scipy.signal import resample_poly

    g = gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g

    # 限制最大 up/down 以避免极端计算量 (当 ratio 非有理数时)
    max_ratio = 50
    if up > max_ratio * down or down > max_ratio * up:
        ratio = target_sr / sr
        if ratio > 1.0:
            up, down = max_ratio, max(1, int(max_ratio / ratio))
        else:
            up, down = max(1, int(max_ratio * ratio)), max_ratio

    return resample_poly(audio.astype(np.float64), up, down).astype(np.float32)


def pitch_shift(audio: np.ndarray, sr: int, semitones: float,
                preserve_formants: bool = True) -> np.ndarray:
    """
    变调 (v7): 双模式策略

    模式选择:
      - preserve_formants=True: 使用 phase vocoder, 变调不变共振峰
        (配合后续 formant_shift 独立控制音色体型)
      - preserve_formants=False: 使用 polyphase resample,
        变调同时改变共振峰 (更快, 无相位伪影, 用于不需要独立共振峰控制的场景)

    preserve_formants 语义:
      True  = 仅变调, 声带振源频率变化但声道滤波器不变 (声道体型感不变)
      False = 变调且共振峰随之变化 (整体重采样, 体型感随变调改变)
    """
    if semitones == 0:
        return audio.copy()

    semitones = np.clip(semitones, -12, 12)

    if preserve_formants or abs(semitones) >= 6:
        # ----- 模式 B: phase vocoder (保共振峰, 或大偏移需要相位稳定性) -----
        # phase vocoder 在 STFT 域调整 F0 而保持包络不变 → 共振峰不变
        shifted = librosa.effects.pitch_shift(
            y=audio.astype(np.float32), sr=sr, n_steps=semitones,
            res_type='scipy',
        )
    else:
        # ----- 模式 A: polyphase resample (变调同时改共振峰, 更干净) -----
        # 重采样改变采样率 → 音高和共振峰同步变化
        rate = 2.0 ** (semitones / 12.0)
        orig_len = len(audio)
        target_sr = int(sr * rate)
        target_sr = max(1000, min(target_sr, sr * 4))
        shifted = _polyphase_resample(audio, sr, target_sr)

        # 裁剪/补零到原长度
        if len(shifted) > orig_len:
            start = (len(shifted) - orig_len) // 2
            shifted = shifted[start:start + orig_len]
        else:
            shifted = np.pad(shifted, (0, orig_len - len(shifted)))

    # 后归一化防止过冲
    max_val = np.max(np.abs(shifted))
    if max_val > 0.99:
        shifted = shifted * 0.99 / max_val
    return shifted.astype(np.float32)
