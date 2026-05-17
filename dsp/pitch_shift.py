"""
声纹魔方 - 变调引擎 (v5)
改用 librosa.effects.pitch_shift 替代手工 Phase Vocoder
"""

import numpy as np
import librosa

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE


def pitch_shift(audio: np.ndarray, sr: int, semitones: float,
                preserve_formants: bool = True) -> np.ndarray:
    """
    变调 (v5): 使用 librosa.effects.pitch_shift

    参数
    ----------
    audio : np.ndarray
        输入音频
    sr : int
        采样率
    semitones : float
        半音偏移，正数升高，负数降低
    preserve_formants : bool
        是否补偿共振峰偏移（注：管线中的显式 formant_shift 与此配合使用）

    返回
    -------
    np.ndarray
        变调后音频
    """
    if semitones == 0:
        return audio.copy()

    semitones = np.clip(semitones, -12, 12)

    # 使用 librosa 的 pitch_shift (内部使用 time_stretch + resample)
    shifted = librosa.effects.pitch_shift(
        y=audio.astype(np.float32), sr=sr, n_steps=semitones,
    )

    return shifted.astype(np.float32)
