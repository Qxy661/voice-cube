"""
声纹魔方 - 麦克风录音组件
支持 Gradio 音频输入（麦克风录音或文件上传）
"""

import numpy as np
import tempfile
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE


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
        # Gradio 格式: (sample_rate, numpy_array)
        sr, audio = audio_input
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # 转单声道
        return audio.astype(np.float32), sr

    if isinstance(audio_input, str):
        # 文件路径
        import soundfile as sf
        audio, sr = sf.read(audio_input)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), sr

    return None, None


def save_audio(audio: np.ndarray, sr: int, suffix: str = ".wav") -> str:
    """
    保存音频到临时文件

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

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    sf.write(tmp.name, audio, sr)
    tmp.close()
    return tmp.name


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    音频响度标准化

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
