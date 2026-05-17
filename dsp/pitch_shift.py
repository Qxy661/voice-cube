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

    - |semis| < 6: polyphase resample-only (无相位伪影，无振铃，语音清晰度最佳)
    - |semis| >= 6: librosa phase vocoder + polyphase resample (保真度更好)
    """
    if semitones == 0:
        return audio.copy()

    semitones = np.clip(semitones, -12, 12)
    rate = 2.0 ** (semitones / 12.0)

    # 预加重: 轻微高频提升补偿重采样损失 (语音清晰度)
    if abs(semitones) > 2:
        audio = audio.astype(np.float64)
        pre = 0.0 if abs(semitones) < 4 else -0.15
        if pre != 0:
            audio = np.append(audio[0], audio[1:] + pre * (audio[1:] - audio[:-1]))
        audio = audio.astype(np.float32)

    if abs(semitones) < 6 and abs(semitones) > 0:
        # ----- 模式 A: polyphase resample-only (干净，无相位伪影) -----
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

        # 后归一化防止重采样引入的微小过冲
        max_val = np.max(np.abs(shifted))
        if max_val > 0.99:
            shifted = shifted * 0.99 / max_val
        return shifted.astype(np.float32)
    else:
        # ----- 模式 B: librosa pitch_shift (大偏移, scipy resample) -----
        # 大偏移时 phase vocoder 已平滑暂态, scipy FFT resample 不会振铃
        shifted = librosa.effects.pitch_shift(
            y=audio.astype(np.float32), sr=sr, n_steps=semitones,
            res_type='scipy',
        )
        max_val = np.max(np.abs(shifted))
        if max_val > 0.99:
            shifted = shifted * 0.99 / max_val
        return shifted.astype(np.float32)
