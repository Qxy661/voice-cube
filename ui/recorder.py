"""
声纹魔方 - 麦克风录音组件 (v3)
支持 Gradio 音频输入（麦克风录音或文件上传）
v3: DC偏移去除、临时文件清理、录音质量提升
"""

import numpy as np
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE

# 临时文件追踪（防止泄漏）
_temp_files = []


def process_audio_input(audio_input) -> tuple:
    """
    处理 Gradio 音频输入（统一格式）

    Parameters
    ----------
    audio_input :
        Gradio 音频组件的输出，格式为 (sample_rate, numpy_array) 或文件路径

    Returns
    -------
    tuple
        (audio_data, sample_rate)
    """
    if audio_input is None:
        return None, None

    if isinstance(audio_input, tuple):
        sr, audio = audio_input
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # 转单声道
        audio = remove_dc_offset(audio)
        return audio.astype(np.float32), sr

    if isinstance(audio_input, str):
        import soundfile as sf
        audio, sr = sf.read(audio_input)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = remove_dc_offset(audio)
        return audio.astype(np.float32), sr

    return None, None


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """
    去除 DC 偏移（均值减法）

    DC偏移会导致F0提取不准、频谱分析偏差，必须在处理前去除
    """
    return (audio - np.mean(audio)).astype(np.float32)


def save_audio(audio: np.ndarray, sr: int, suffix: str = ".wav") -> str:
    """
    保存音频到临时文件（v3: 自动清理旧文件）

    Parameters
    ----------
    audio : np.ndarray
        音频数据
    sr : int
        采样率
    suffix : str
        文件后缀

    Returns
    -------
    str
        临时文件路径
    """
    import soundfile as sf

    # 清理之前的临时文件
    cleanup_temp_files()

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    sf.write(tmp.name, audio, sr)
    tmp.close()

    _temp_files.append(tmp.name)
    return tmp.name


def cleanup_temp_files():
    """清理所有已追踪的临时文件"""
    global _temp_files
    for path in _temp_files:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass
    _temp_files.clear()


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    音频响度标准化（RMS-based）

    注意: 此函数不应与 normalize_loudness() (LUFS-based) 同时使用
    推荐在管线末尾使用 normalize_loudness() 进行最终响度归一化

    Parameters
    ----------
    audio : np.ndarray
        输入音频
    target_db : float
        目标响度 (dB)

    Returns
    -------
    np.ndarray
        标准化后的音频
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return audio

    target_rms = 10 ** (target_db / 20.0)
    gain = target_rms / rms

    output = audio * gain

    # 防止削波
    max_val = np.max(np.abs(output))
    if max_val > 0.99:
        output = output * 0.99 / max_val

    return output.astype(np.float32)
