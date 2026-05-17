"""
声纹魔方 - LPC 共振峰移位引擎 (v5)
改进: 清浊音检测, 仅处理浊音帧, LPC-46 @44100Hz
"""

import numpy as np
from scipy.signal import lfilter

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, LPC_ORDER, LPC_PRE_EMPHASIS, HOP_LENGTH


def _lpc(signal: np.ndarray, order: int, pre_emphasis: float = 0.0) -> np.ndarray:
    """
    手动实现 LPC 系数计算

    Parameters
    ----------
    signal : np.ndarray
        输入信号帧
    order : int
        LPC 阶数
    pre_emphasis : float
        预加重系数

    Returns
    -------
    np.ndarray
        LPC 系数 (首元素为 1.0)
    """
    n = len(signal)
    if n <= order:
        return np.concatenate(([1.0], np.zeros(order)))

    # 预加重
    if pre_emphasis > 0:
        signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # 自相关
    r = np.correlate(signal, signal, mode="full")
    r = r[n - 1:]

    if r[0] < 1e-10:
        return np.concatenate(([1.0], np.zeros(order)))

    # Levinson-Durbin
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]

    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]

        if abs(e) < 1e-10:
            break

        k = -acc / e
        k = np.clip(k, -0.999, 0.999)

        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        a = a_new

        e *= (1 - k * k)
        if e <= 0:
            e = 1e-10

    return a


def _detect_voiced_frames(audio: np.ndarray, sr: int, frame_len: int, hop: int) -> np.ndarray:
    """
    清浊音检测：计算每帧 RMS 和过零率，标记浊音帧

    返回
    -------
    np.ndarray (bool)
        True = 浊音帧（需要共振峰处理）
    """
    n_frames = 1 + (len(audio) - frame_len) // hop
    voiced = np.zeros(n_frames, dtype=bool)

    # 全局 RMS 用于自适应阈值
    global_rms = np.sqrt(np.mean(audio ** 2))
    threshold = max(global_rms * 0.3, 0.002)  # 自适应能量阈值

    for i in range(n_frames):
        start = i * hop
        end = min(start + frame_len, len(audio))
        frame = audio[start:end]

        if len(frame) < frame_len // 2:
            continue

        # 能量检测
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < threshold:
            continue

        # 过零率检测（浊音过零率低，清音/摩擦音过零率高）
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        zcr_threshold = 0.3  # 浊音过零率通常 < 0.2-0.3

        # 浊音条件：有足够能量 + 低过零率
        voiced[i] = (zcr < zcr_threshold)

    return voiced


def formant_shift(audio: np.ndarray, sr: int, ratio: float,
                  order: int = LPC_ORDER) -> np.ndarray:
    """
    LPC 共振峰移位 (v5): 仅处理浊音帧

    改进:
      - 清浊音检测，仅对浊音帧做共振峰移位
      - LPC-46 @44100Hz
      - 清音/无声帧直接通过，保留自然感

    Parameters
    ----------
    audio : np.ndarray
        输入音频
    sr : int
        采样率
    ratio : float
        共振峰缩放比例
    order : int
        LPC 阶数
    """
    if ratio == 1.0:
        return audio.copy()

    ratio = np.clip(ratio, 0.4, 2.5)

    frame_len = 2048
    hop = HOP_LENGTH
    window = np.hanning(frame_len)
    n_frames = 1 + (len(audio) - frame_len) // hop

    # 清浊音检测
    voiced_flags = _detect_voiced_frames(audio, sr, frame_len, hop)

    output = np.zeros(len(audio) + frame_len)
    window_sum = np.zeros(len(audio) + frame_len)

    for i in range(n_frames):
        start = i * hop
        end = start + frame_len
        if end > len(audio):
            break

        frame = audio[start:end]

        if not voiced_flags[i]:
            # 清音/无声帧：直通
            output[start:end] += frame * window
            window_sum[start:end] += window ** 2
            continue

        windowed = frame * window

        # Step 1: LPC 分析
        try:
            a = _lpc(windowed, order=order, pre_emphasis=LPC_PRE_EMPHASIS)
        except Exception:
            output[start:end] += windowed
            window_sum[start:end] += window ** 2
            continue

        # Step 2: 求根并筛选
        roots = np.roots(a)
        new_roots = []
        used = set()

        for j, r in enumerate(roots):
            if j in used:
                continue
            if np.abs(r) >= 1.0:
                new_roots.append(0.99 * r / np.abs(r))
                continue

            conj_r = np.conj(r)
            found_pair = False
            for k, r2 in enumerate(roots):
                if k != j and k not in used and np.abs(r2 - conj_r) < 1e-5:
                    used.add(j)
                    used.add(k)

                    angle = np.angle(r)
                    mag = np.abs(r)

                    new_angle = angle * ratio
                    new_r = mag * np.exp(1j * new_angle)
                    new_roots.append(new_r)
                    new_roots.append(np.conj(new_r))
                    found_pair = True
                    break

            if not found_pair:
                new_roots.append(r)
                used.add(j)

        # Step 3: 重建 LPC 系数
        new_roots = np.array(new_roots)
        new_a = np.poly(new_roots)
        new_a = np.real(new_a)

        # 稳定性检查
        poles = np.roots(new_a)
        if np.any(np.abs(poles) >= 1.0):
            scale = 0.98 / np.max(np.abs(poles))
            poles = poles * scale
            new_a = np.poly(poles)
            new_a = np.real(new_a)

        # Step 4: LPC 合成
        residual = lfilter(a, [1.0], windowed)
        synthesized = lfilter([1.0], new_a, residual)

        output[start:end] += synthesized * window
        window_sum[start:end] += window ** 2

    # 归一化窗重叠
    nonzero = window_sum > 1e-8
    output[:len(audio)][nonzero[:len(audio)]] /= window_sum[:len(audio)][nonzero[:len(audio)]]
    output = output[:len(audio)]

    return output.astype(np.float32)


def extract_formants(audio: np.ndarray, sr: int, order: int = LPC_ORDER) -> list:
    """提取共振峰频率"""
    frame_len = 2048
    center = len(audio) // 2
    start = max(0, center - frame_len // 2)
    actual_len = min(frame_len, len(audio) - start)
    frame = audio[start:start + actual_len] * np.hanning(actual_len)

    try:
        a = _lpc(frame, order=order, pre_emphasis=LPC_PRE_EMPHASIS)
    except Exception:
        return []

    roots = np.roots(a)

    formants = []
    for r in roots:
        if np.abs(r) < 1.0 and np.imag(r) > 0:
            freq = np.angle(r) * sr / (2 * np.pi)
            if 50 < freq < sr / 2:
                formants.append(float(freq))

    formants.sort()
    return formants
