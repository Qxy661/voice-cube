"""
声纹魔方 - 变调引擎 (v6)
双模式: 小偏移用 resample (保清音), 大偏移用 librosa
"""

import numpy as np
import librosa


def pitch_shift(audio: np.ndarray, sr: int, semitones: float,
                preserve_formants: bool = True) -> np.ndarray:
    """
    变调 (v6): 双模式策略

    - |semis| < 6: resample-only (无相位伪影，语音清晰度最佳)
    - |semis| >= 6: librosa.effects.pitch_shift (保真度更好)
    """
    if semitones == 0:
        return audio.copy()

    semitones = np.clip(semitones, -12, 12)
    rate = 2.0 ** (semitones / 12.0)

    if abs(semitones) < 6 and abs(semitones) > 0:
        # ----- 模式 A: resample-only (干净，无相位伪影) -----
        # 重采样改变音高 (时长也随之变化)
        orig_len = len(audio)
        target_sr = int(sr * rate)
        # 限制极端重采样比率以避免 aliasing
        target_sr = max(1000, min(target_sr, sr * 4))
        shifted = librosa.resample(
            audio.astype(np.float32),
            orig_sr=sr,
            target_sr=target_sr,
            res_type='scipy',
        )
        # 裁剪/补零到原长度
        if len(shifted) > orig_len:
            # 居中裁剪以保留中间部分(过渡更自然)
            start = (len(shifted) - orig_len) // 2
            shifted = shifted[start:start + orig_len]
        else:
            shifted = np.pad(shifted, (0, orig_len - len(shifted)))
        return shifted.astype(np.float32)
    else:
        # ----- 模式 B: librosa pitch_shift (大偏移) -----
        shifted = librosa.effects.pitch_shift(
            y=audio.astype(np.float32), sr=sr, n_steps=semitones,
            res_type='scipy',
        )
        return shifted.astype(np.float32)
