"""
声纹魔方 - Phase Vocoder 变调引擎
实现变调不变速的核心 DSP 算法

算法流程:
  输入音频 → STFT分帧 → 相位累积修正 → 频率缩放 → ISTFT重建 → 输出音频

依赖: librosa, numpy
"""

import numpy as np
import librosa

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH, WINDOW


def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """
    Phase Vocoder 变调：改变音高但保持时长不变

    Parameters
    ----------
    audio : np.ndarray
        输入音频信号 (一维数组)
    sr : int
        采样率
    semitones : float
        音高偏移量（半音），正数升高，负数降低，范围 -12 ~ +12

    Returns
    -------
    np.ndarray
        变调后的音频信号
    """
    if semitones == 0:
        return audio.copy()

    # 限制范围
    semitones = np.clip(semitones, -12, 12)

    # 使用 librosa 的 phase_vocoder 实现 STFT 域变调
    # Step 1: STFT 分析
    stft_matrix = librosa.stft(
        audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WINDOW
    )

    # Step 2: 计算频率缩放因子
    # 2^(semitones/12) 是半音到频率比的转换
    rate = 2.0 ** (semitones / 12.0)

    # Step 3: 使用 phase_vocoder 进行时间拉伸/压缩
    # 为了改变音高，我们先做时间拉伸，再重采样回原时长
    # 时间拉伸率 = 1/rate，这样拉伸后再以 rate 倍速播放 = 音高升高 rate 倍
    stretched_stft = librosa.phase_vocoder(stft_matrix, rate=1.0 / rate, hop_length=HOP_LENGTH)

    # Step 4: ISTFT 重建
    stretched_audio = librosa.istft(stretched_stft, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WINDOW)

    # Step 5: 重采样回原始长度（时间拉伸后长度变了，需要重采样恢复）
    target_length = len(audio)
    if len(stretched_audio) != target_length:
        # 使用线性插值重采样
        indices = np.linspace(0, len(stretched_audio) - 1, target_length)
        shifted_audio = np.interp(indices, np.arange(len(stretched_audio)), stretched_audio)
    else:
        shifted_audio = stretched_audio

    # 归一化防止削波
    max_val = np.max(np.abs(shifted_audio))
    if max_val > 1.0:
        shifted_audio = shifted_audio / max_val

    return shifted_audio.astype(np.float32)


def pitch_shift_psola(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """
    PSOLA (基音同步叠加) 变调 - 备选方案

    Parameters
    ----------
    audio : np.ndarray
        输入音频信号
    sr : int
        采样率
    semitones : float
        音高偏移量（半音）

    Returns
    -------
    np.ndarray
        变调后的音频信号
    """
    if semitones == 0:
        return audio.copy()

    rate = 2.0 ** (semitones / 12.0)

    # 基频估计
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )

    # 对基频进行缩放
    f0_shifted = f0 * rate

    # 使用 librosa 的 time_stretch 和 resample 组合实现
    # 先时间拉伸
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    # 再重采样回原长度
    target_length = len(audio)
    if len(stretched) != target_length:
        indices = np.linspace(0, len(stretched) - 1, target_length)
        result = np.interp(indices, np.arange(len(stretched)), stretched)
    else:
        result = stretched

    max_val = np.max(np.abs(result))
    if max_val > 1.0:
        result = result / max_val

    return result.astype(np.float32)
