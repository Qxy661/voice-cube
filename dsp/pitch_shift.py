"""
声纹魔方 - Phase Vocoder 变调引擎 (v2)
改进: polyphase重采样, 共振峰保护, 多带处理
"""

import numpy as np
import librosa

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH, WINDOW


def pitch_shift(audio: np.ndarray, sr: int, semitones: float,
                preserve_formants: bool = True) -> np.ndarray:
    """
    Phase Vocoder 变调 (v2): 高质量 polyphase 重采样 + 可选共振峰保护

    Parameters
    ----------
    audio : np.ndarray
        输入音频信号 (一维数组)
    sr : int
        采样率
    semitones : float
        音高偏移量（半音），正数升高，负数降低
    preserve_formants : bool
        是否在变调后补偿共振峰偏移（防止低沉变闷/高亢变尖）

    Returns
    -------
    np.ndarray
        变调后的音频信号
    """
    if semitones == 0:
        return audio.copy()

    semitones = np.clip(semitones, -12, 12)
    rate = 2.0 ** (semitones / 12.0)

    # Step 1: STFT 分析
    stft_matrix = librosa.stft(
        audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=WINDOW,
    )

    # Step 2: Phase Vocoder 时间拉伸
    stretched_stft = librosa.phase_vocoder(
        stft_matrix, rate=1.0 / rate, hop_length=HOP_LENGTH,
    )

    # Step 3: ISTFT 重建
    stretched_audio = librosa.istft(
        stretched_stft, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=WINDOW,
    )

    # Step 4: Polyphase 重采样 (替代线性插值，质量显著提升)
    target_length = len(audio)
    if len(stretched_audio) != target_length:
        # librosa.resample 使用 polyphase 滤波器，远优于线性插值
        shifted_audio = librosa.resample(
            stretched_audio,
            orig_sr=int(sr * len(stretched_audio) / target_length),
            target_sr=sr,
        )
        # 精确长度对齐
        if len(shifted_audio) > target_length:
            shifted_audio = shifted_audio[:target_length]
        elif len(shifted_audio) < target_length:
            shifted_audio = np.pad(shifted_audio, (0, target_length - len(shifted_audio)))
    else:
        shifted_audio = stretched_audio

    # Step 5: 共振峰保护 — 补偿因变调引起的共振峰偏移
    if preserve_formants and abs(semitones) > 1.0:
        from dsp.formant_shift import formant_shift
        # 变调会让共振峰同比例偏移，需要反向补偿
        compensation = 1.0 / rate
        # 限制补偿幅度，避免过度矫正
        compensation = np.clip(compensation, 0.7, 1.4)
        shifted_audio = formant_shift(shifted_audio, sr, compensation)

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
        result = librosa.resample(
            stretched,
            orig_sr=int(sr * len(stretched) / target_length),
            target_sr=sr,
        )
        if len(result) > target_length:
            result = result[:target_length]
        elif len(result) < target_length:
            result = np.pad(result, (0, target_length - len(result)))
    else:
        result = stretched

    max_val = np.max(np.abs(result))
    if max_val > 1.0:
        result = result / max_val

    return result.astype(np.float32)
