"""
声纹魔方 - LPC 共振峰移位引擎 (v6)
双模式: 小变化用EQ近似 (干净), 大变化用LPC (精确)
"""

import numpy as np
from scipy.signal import lfilter, butter, sosfilt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, LPC_ORDER, LPC_PRE_EMPHASIS, HOP_LENGTH


def _lpc(signal: np.ndarray, order: int, pre_emphasis: float = 0.0) -> np.ndarray:
    """手动实现 LPC 系数计算"""
    n = len(signal)
    if n <= order:
        return np.concatenate(([1.0], np.zeros(order)))

    if pre_emphasis > 0:
        signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    r = np.correlate(signal, signal, mode="full")
    r = r[n - 1:]

    if r[0] < 1e-10:
        return np.concatenate(([1.0], np.zeros(order)))

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


def _formant_eq_approximation(audio: np.ndarray, ratio: float) -> np.ndarray:
    """
    EQ 近似共振峰变化 (v6 新增)
    对微小比例变化使用双二阶倾斜滤波替代 LPC 重合成

    ratio > 1.0 = 提升高频 (更亮/更小体型感)
    ratio < 1.0 = 衰减高频 (更暗/更大体型感)
    """
    sr = SAMPLE_RATE

    # 将 ratio 映射为 dB 倾斜量
    # ratio 1.05 → 约 +0.75 dB tilt (轻微更亮)
    # ratio 1.20 → 约 +3.0 dB tilt (明显更亮)
    # ratio 0.85 → 约 -2.5 dB tilt (明显更暗)
    tilt_db = (ratio - 1.0) * 15.0

    if abs(tilt_db) < 0.5:
        return audio.copy()

    # 设计一阶倾斜滤波器: y[n] = x[n] + k * (x[n] - x[n-1])
    # k > 0 提升高频, k < 0 衰减高频
    k = tilt_db / 60.0  # 映射系数
    k = np.clip(k, -0.8, 0.8)

    output = np.zeros_like(audio, dtype=np.float64)
    output[0] = audio[0]
    for i in range(1, len(audio)):
        output[i] = audio[i] + k * (audio[i] - audio[i - 1])

    return output.astype(np.float32)


def _detect_voiced_frames(audio: np.ndarray, sr: int, frame_len: int, hop: int) -> np.ndarray:
    """清浊音检测：RMS + 过零率"""
    n_frames = 1 + (len(audio) - frame_len) // hop
    voiced = np.zeros(n_frames, dtype=bool)

    global_rms = np.sqrt(np.mean(audio ** 2))
    threshold = max(global_rms * 0.3, 0.002)

    for i in range(n_frames):
        start = i * hop
        end = min(start + frame_len, len(audio))
        frame = audio[start:end]

        if len(frame) < frame_len // 2:
            continue

        rms = np.sqrt(np.mean(frame ** 2))
        if rms < threshold:
            continue

        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        voiced[i] = (zcr < 0.3)

    return voiced


def _lpc_formant_shift(audio: np.ndarray, ratio: float, order: int) -> np.ndarray:
    """
    完整 LPC 共振峰移位 (仅用于极端比例)
    ratio < 0.85 或 ratio > 1.15
    """
    sr = SAMPLE_RATE
    frame_len = 2048
    hop = HOP_LENGTH
    window = np.hanning(frame_len)
    n_frames = 1 + (len(audio) - frame_len) // hop

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
            output[start:end] += frame * window
            window_sum[start:end] += window ** 2
            continue

        windowed = frame * window

        try:
            a = _lpc(windowed, order=order, pre_emphasis=LPC_PRE_EMPHASIS)
        except Exception:
            output[start:end] += windowed
            window_sum[start:end] += window ** 2
            continue

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

        new_roots = np.array(new_roots)
        new_a = np.poly(new_roots)
        new_a = np.real(new_a)

        poles = np.roots(new_a)
        if np.any(np.abs(poles) >= 1.0):
            scale = 0.98 / np.max(np.abs(poles))
            poles = poles * scale
            new_a = np.poly(poles)
            new_a = np.real(new_a)

        residual = lfilter(a, [1.0], windowed)
        synthesized = lfilter([1.0], new_a, residual)

        output[start:end] += synthesized * window
        window_sum[start:end] += window ** 2

    nonzero = window_sum > 1e-8
    output[:len(audio)][nonzero[:len(audio)]] /= window_sum[:len(audio)][nonzero[:len(audio)]]
    output = output[:len(audio)]

    return output.astype(np.float32)


def formant_shift(audio: np.ndarray, sr: int, ratio: float,
                  order: int = LPC_ORDER) -> np.ndarray:
    """
    共振峰移位 (v6): 双模式策略

    - |ratio-1.0| < 0.15: EQ 倾斜滤波近似 (干净无伪影)
    - |ratio-1.0| >= 0.15: 完整 LPC 共振峰移位 + 清浊音检测
    """
    if ratio == 1.0:
        return audio.copy()

    ratio = np.clip(ratio, 0.4, 2.5)

    # 判断使用哪种模式
    if abs(ratio - 1.0) < 0.15:
        # 模式 A: EQ 近似 (clean, 无 LPC 伪影)
        return _formant_eq_approximation(audio, ratio)
    else:
        # 模式 B: 完整 LPC (极端比例)
        return _lpc_formant_shift(audio, ratio, order)


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
