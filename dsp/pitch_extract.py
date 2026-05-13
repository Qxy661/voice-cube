"""
声纹魔方 - 基频提取模块
使用传统 DSP 方法精确提取语音基频 (F0) 轮廓线

支持两种算法:
  1. librosa.pyin (概率 YIN 算法)
  2. 自相关法 (Autocorrelation)
"""

import numpy as np
import librosa

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, HOP_LENGTH


def extract_f0(audio: np.ndarray, sr: int, method: str = "pyin") -> np.ndarray:
    """
    提取音频的基频 (F0) 轮廓线

    Parameters
    ----------
    audio : np.ndarray
        输入音频信号
    sr : int
        采样率
    method : str
        提取方法: "pyin" (概率YIN) 或 "autocorrelation" (自相关法)

    Returns
    -------
    np.ndarray
        F0 轮廓线 (Hz)，未检测到基频的帧为 0 或 NaN
    """
    if method == "pyin":
        return _extract_f0_pyin(audio, sr)
    elif method == "autocorrelation":
        return _extract_f0_autocorrelation(audio, sr)
    else:
        raise ValueError(f"未知方法: {method}，支持 'pyin' 或 'autocorrelation'")


def _extract_f0_pyin(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    概率 YIN 算法提取 F0
    librosa.pyin 基于 YIN 算法的改进版本，引入概率模型处理清浊音判断
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),    # 最低 65 Hz
        fmax=librosa.note_to_hz("C7"),    # 最高 2093 Hz
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    # 将 NaN 替换为 0
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0


def _extract_f0_autocorrelation(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    自相关法提取 F0（纯手工 DSP 实现）

    算法:
    1. 分帧加窗
    2. 计算自相关函数
    3. 在合理延迟范围内找峰值
    4. 峰值位置 = 基频周期
    """
    frame_len = 2048
    hop = HOP_LENGTH
    fmin, fmax = 65, 1000  # 基频范围

    # 延迟范围 (采样数)
    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)

    n_frames = 1 + (len(audio) - frame_len) // hop
    f0 = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + frame_len] * np.hanning(frame_len)

        # 自相关
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]  # 只取正半部分

        # 在合理延迟范围内找峰值
        search_range = corr[lag_min:lag_max + 1]
        if len(search_range) == 0:
            continue

        peak_idx = np.argmax(search_range) + lag_min

        # 验证峰值是否显著（浊音判断）
        if corr[peak_idx] > 0.3 * corr[0]:
            f0[i] = sr / peak_idx
        else:
            f0[i] = 0.0

    return f0


def f0_to_midi(f0: np.ndarray) -> np.ndarray:
    """将 F0 (Hz) 转换为 MIDI 音高编号"""
    midi = np.zeros_like(f0)
    nonzero = f0 > 0
    midi[nonzero] = 69 + 12 * np.log2(f0[nonzero] / 440.0)
    return midi
