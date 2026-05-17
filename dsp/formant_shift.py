"""
声纹魔方 - LPC 共振峰移位引擎 (v2)
改进: 预加重, 高阶LPC, 稳定性检查, 重叠相加优化
"""

import numpy as np
from scipy.signal import lfilter, lfiltic

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, LPC_ORDER, LPC_PRE_EMPHASIS, HOP_LENGTH


def _lpc(signal: np.ndarray, order: int, pre_emphasis: float = 0.0) -> np.ndarray:
    """
    手动实现 LPC 系数计算 (v2)
    支持预加重，减少数值问题

    Parameters
    ----------
    signal : np.ndarray
        输入信号帧
    order : int
        LPC 阶数
    pre_emphasis : float
        预加重系数 (0.0 = 不预加重, 0.97 = 标准)

    Returns
    -------
    np.ndarray
        LPC 系数 (首元素为 1.0)
    """
    n = len(signal)
    if n <= order:
        return np.concatenate(([1.0], np.zeros(order)))

    # 预加重: 增强高频，改善LPC稳定性
    if pre_emphasis > 0:
        signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # 计算自相关函数
    r = np.correlate(signal, signal, mode="full")
    r = r[n - 1:]  # 取正延迟部分

    if r[0] < 1e-10:
        return np.concatenate(([1.0], np.zeros(order)))

    # Levinson-Durbin 递推
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

        # 反射系数限幅，防止不稳定
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


def formant_shift(audio: np.ndarray, sr: int, ratio: float,
                  order: int = LPC_ORDER) -> np.ndarray:
    """
    LPC 共振峰移位 (v2): 改变声音的"体型"感

    改进:
      - 预加重减少数值误差
      - 高阶LPC(24)提供更精确的共振峰分辨率
      - 稳定性检查防止滤波器发散
      - 重叠相加窗函数优化

    Parameters
    ----------
    audio : np.ndarray
        输入音频信号
    sr : int
        采样率
    ratio : float
        共振峰缩放比例，1.0=不变，<1.0=更小，>1.0=更大
    order : int
        LPC 阶数，默认 24

    Returns
    -------
    np.ndarray
        共振峰移位后的音频信号
    """
    if ratio == 1.0:
        return audio.copy()

    ratio = np.clip(ratio, 0.4, 2.5)

    frame_len = 2048
    hop = HOP_LENGTH
    window = np.hanning(frame_len)
    n_frames = 1 + (len(audio) - frame_len) // hop

    output = np.zeros(len(audio) + frame_len)  # 多分配空间
    window_sum = np.zeros(len(audio) + frame_len)

    for i in range(n_frames):
        start = i * hop
        end = start + frame_len
        if end > len(audio):
            break

        frame = audio[start:end]
        windowed = frame * window

        # Step 1: LPC 分析 (带预加重)
        try:
            a = _lpc(windowed, order=order, pre_emphasis=LPC_PRE_EMPHASIS)
        except Exception:
            output[start:end] += windowed
            window_sum[start:end] += window ** 2
            continue

        # Step 2: 求 LPC 多项式的根
        roots = np.roots(a)

        # Step 3: 筛选单位圆内的根，移动共振峰
        new_roots = []
        used = set()
        for j, r in enumerate(roots):
            if j in used:
                continue
            if np.abs(r) >= 1.0:
                # 不稳定的根映射回单位圆内
                new_roots.append(0.99 * r / np.abs(r))
                continue

            # 找共轭对
            conj_r = np.conj(r)
            for k, r2 in enumerate(roots):
                if k != j and k not in used and np.abs(r2 - conj_r) < 1e-5:
                    used.add(j)
                    used.add(k)

                    angle = np.angle(r)
                    magnitude = np.abs(r)

                    # 缩放角度 = 缩放共振峰频率
                    new_angle = angle * ratio
                    new_r = magnitude * np.exp(1j * new_angle)
                    new_roots.append(new_r)
                    new_roots.append(np.conj(new_r))
                    break
            else:
                new_roots.append(r)
                used.add(j)

        # Step 4: 从新根重建 LPC 系数
        new_roots = np.array(new_roots)
        new_a = np.poly(new_roots)
        new_a = np.real(new_a)

        # 稳定性检查: 所有极点必须在单位圆内
        poles = np.roots(new_a)
        if np.any(np.abs(poles) >= 1.0):
            # 不稳定，缩小极点半径
            scale = 0.98 / np.max(np.abs(poles))
            poles = poles * scale
            new_a = np.poly(poles)
            new_a = np.real(new_a)

        # Step 5: LPC 合成
        residual = lfilter(a, [1.0], windowed)
        synthesized = lfilter([1.0], new_a, residual)

        # 重叠相加
        output[start:end] += synthesized * window
        window_sum[start:end] += window ** 2

    # 归一化窗重叠
    nonzero = window_sum > 1e-8
    output[:len(audio)][nonzero[:len(audio)]] /= window_sum[:len(audio)][nonzero[:len(audio)]]
    output = output[:len(audio)]

    return output.astype(np.float32)


def extract_formants(audio: np.ndarray, sr: int, order: int = LPC_ORDER) -> list:
    """
    提取音频的共振峰频率（用于可视化分析）

    Parameters
    ----------
    audio : np.ndarray
        输入音频信号
    sr : int
        采样率
    order : int
        LPC 阶数

    Returns
    -------
    list
        共振峰频率列表 (Hz)
    """
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
