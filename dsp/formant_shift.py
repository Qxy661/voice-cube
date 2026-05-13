"""
声纹魔方 - LPC 共振峰移位引擎
通过线性预测编码提取声道共振峰，移动极点改变音色"体型"感

算法流程:
  输入音频 → 分帧 → LPC分析 → 求根找共振峰 → 移动极点 → LPC合成 → 输出

依赖: numpy, scipy
"""

import numpy as np
from scipy.signal import lfilter


def _lpc(signal: np.ndarray, order: int) -> np.ndarray:
    """
    手动实现 LPC (线性预测编码) 系数计算
    使用自相关法 + Levinson-Durbin 递推

    Parameters
    ----------
    signal : np.ndarray
        输入信号帧
    order : int
        LPC 阶数

    Returns
    -------
    np.ndarray
        LPC 系数 (首元素为 1.0)
    """
    n = len(signal)
    if n <= order:
        return np.concatenate(([1.0], np.zeros(order)))

    # 计算自相关函数
    r = np.correlate(signal, signal, mode="full")
    r = r[n - 1:]  # 取正延迟部分

    if r[0] == 0:
        return np.concatenate(([1.0], np.zeros(order)))

    # Levinson-Durbin 递推
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]

    for i in range(1, order + 1):
        # 计算反射系数
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]

        if abs(e) < 1e-10:
            break

        k = -acc / e

        # 更新预测系数
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        a = a_new

        # 更新预测误差
        e *= (1 - k * k)
        if e <= 0:
            break

    return a


import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, LPC_ORDER, HOP_LENGTH


def formant_shift(audio: np.ndarray, sr: int, ratio: float, order: int = LPC_ORDER) -> np.ndarray:
    """
    LPC 共振峰移位：改变声音的"体型"感

    Parameters
    ----------
    audio : np.ndarray
        输入音频信号 (一维数组)
    sr : int
        采样率
    ratio : float
        共振峰缩放比例，1.0=不变，<1.0=更小（小黄人），>1.0=更大（巨人）
        推荐范围: 0.4 ~ 2.5
    order : int
        LPC 阶数，默认 16

    Returns
    -------
    np.ndarray
        共振峰移位后的音频信号
    """
    if ratio == 1.0:
        return audio.copy()

    ratio = np.clip(ratio, 0.4, 2.5)

    # 分帧处理
    frame_len = 2048
    hop = HOP_LENGTH
    n_frames = 1 + (len(audio) - frame_len) // hop

    output = np.zeros_like(audio)

    for i in range(n_frames):
        start = i * hop
        end = start + frame_len
        if end > len(audio):
            break

        frame = audio[start:end]

        # 应用汉宁窗
        windowed = frame * np.hanning(frame_len)

        # Step 1: LPC 分析
        try:
            a = _lpc(windowed, order=order)
        except Exception:
            # LPC 失败时保留原帧
            output[start:end] += frame * np.hanning(frame_len)
            continue

        # Step 2: 求 LPC 多项式的根
        roots = np.roots(a)

        # Step 3: 筛选单位圆内的根（极点）
        # 只处理共轭极点对（共振峰）
        new_roots = []
        used = set()
        for j, r in enumerate(roots):
            if j in used:
                continue
            if np.abs(r) >= 1.0:
                new_roots.append(r)
                continue

            # 找共轭对
            conj_r = np.conj(r)
            for k, r2 in enumerate(roots):
                if k != j and k not in used and np.abs(r2 - conj_r) < 1e-6:
                    used.add(j)
                    used.add(k)

                    # 移动极点：按 ratio 缩放角度（改变共振峰频率）
                    angle = np.angle(r)
                    magnitude = np.abs(r)

                    # 缩放角度 = 缩放共振峰频率
                    new_angle = angle * ratio
                    new_r = magnitude * np.exp(1j * new_angle)
                    new_roots.append(new_r)
                    new_roots.append(np.conj(new_r))
                    break
            else:
                # 非共轭根，保持不变
                new_roots.append(r)
                used.add(j)

        # Step 4: 从新根重建 LPC 系数
        new_roots = np.array(new_roots)
        new_a = np.poly(new_roots)

        # 确保系数为实数
        new_a = np.real(new_a)

        # Step 5: LPC 合成（滤波）
        # 用原始残差激励 + 新的声道模型
        residual = lfilter(a, [1.0], windowed)
        synthesized = lfilter([1.0], new_a, residual)

        # 重叠相加
        output[start:end] += synthesized * np.hanning(frame_len)

    # 归一化
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

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
    # 取中间帧分析
    frame_len = 2048
    center = len(audio) // 2
    start = max(0, center - frame_len // 2)
    frame = audio[start:start + frame_len] * np.hanning(min(frame_len, len(audio) - start))

    try:
        a = _lpc(frame, order=order)
    except Exception:
        return []

    roots = np.roots(a)

    formants = []
    for r in roots:
        if np.abs(r) < 1.0 and np.imag(r) > 0:
            freq = np.angle(r) * sr / (2 * np.pi)
            if 50 < freq < sr / 2:
                formants.append(freq)

    formants.sort()
    return formants
