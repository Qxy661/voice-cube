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


def _low_shelf_coeffs(fc, gain_db, sr, Q=0.707):
    """低频搁架滤波器系数 (RBJ Cookbook)"""
    w0 = 2 * np.pi * fc / sr
    A = 10 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2 * Q)
    b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
    a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])


def _high_shelf_coeffs(fc, gain_db, sr, Q=0.707):
    """高频搁架滤波器系数 (RBJ Cookbook)"""
    w0 = 2 * np.pi * fc / sr
    A = 10 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2 * Q)
    b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
    a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])


def _formant_eq_approximation(audio: np.ndarray, ratio: float) -> np.ndarray:
    """
    EQ 近似共振峰变化 (v7): 双二阶搁架滤波器

    使用 RBJ 搁架滤波器替代旧的差分高通，避免高频增益不稳定。

    ratio > 1.0 = 提升高频 (更亮/更小体型感)
    ratio < 1.0 = 衰减高频 (更暗/更大体型感)
    """
    sr = SAMPLE_RATE

    # ratio → dB 搁架增益
    tilt_db = (ratio - 1.0) * 12.0

    if abs(tilt_db) < 0.5:
        return audio.copy()

    output = audio.astype(np.float64)

    if tilt_db > 0:
        # 提升: 高频搁架 boost
        b, a = _high_shelf_coeffs(3000, tilt_db, sr)
    else:
        # 衰减: 低频搁架 cut (等效于相对提升高频)
        b, a = _low_shelf_coeffs(500, tilt_db, sr)

    output = lfilter(b, a, output)

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
