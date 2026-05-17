"""
声纹魔方 - 共振峰移位引擎 (v8)
三模式策略: EQ近似(微调) → STFT倒谱包络(主流) → LPC(备用)

v8: 增加 STFT 倒谱包络扭曲作为主力模式 (开源变声器通行做法)
     LPC 降为备用 (仅当 STFT 模式异常时)
"""

import numpy as np
from scipy.signal import lfilter
import librosa

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, LPC_ORDER, LPC_PRE_EMPHASIS, HOP_LENGTH


# ════════════════════════════════════════════════════════════════════
#  LPC 子函数 (备用模式)
# ════════════════════════════════════════════════════════════════════

def _lpc(signal: np.ndarray, order: int, pre_emphasis: float = 0.0) -> np.ndarray:
    """Levinson-Durbin LPC 系数计算"""
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


# ════════════════════════════════════════════════════════════════════
#  EQ 近似 (微调用, |ratio-1| < 0.10)
# ════════════════════════════════════════════════════════════════════

def _low_shelf_coeffs(fc, gain_db, sr, Q=0.707):
    """低频搁架滤波器 (RBJ)"""
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
    """高频搁架滤波器 (RBJ)"""
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


def _formant_eq_approx(audio: np.ndarray, ratio: float) -> np.ndarray:
    """EQ 近似共振峰变化 (微小偏移时使用, RBJ搁架滤波)"""
    sr = SAMPLE_RATE
    tilt_db = (ratio - 1.0) * 12.0
    if abs(tilt_db) < 0.5:
        return audio.copy()
    output = audio.astype(np.float64)
    if tilt_db > 0:
        b, a = _high_shelf_coeffs(3000, tilt_db, sr)
    else:
        b, a = _low_shelf_coeffs(500, tilt_db, sr)
    output = lfilter(b, a, output)
    return output.astype(np.float32)


# ════════════════════════════════════════════════════════════════════
#  STFT 倒谱包络扭曲 (主力模式, |ratio-1| >= 0.10)
#  ════════════════════════════════════════════════════════════════════
#  原理:
#    1. STFT → 幅度谱 + 相位
#    2. log幅度 → FFT → 保留低阶倒谱系数 → IFFT → 平滑包络
#    3. 包络频率轴拉伸/压缩 (ratio) → 新包络
#    4. 新包络 + 原始精细结构 → 新幅度谱 → ISTFT
#
#  开源变声器通行做法 (WORLD CheapTrick 的简化版), 无 LPC 数值不稳定问题.


def _stft_formant_shift(audio: np.ndarray, ratio: float) -> np.ndarray:
    """
    STFT 倒谱包络扭曲法共振峰移位

    对幅度谱进行倒谱域平滑 → 频率轴扭曲 → 重合成。
    全程无需多项式求根, 无数值发散风险。
    """
    n_fft = 2048
    hop = HOP_LENGTH
    win_length = n_fft

    # 1. STFT
    D = librosa.stft(
        audio.astype(np.float32),
        n_fft=n_fft, hop_length=hop, win_length=win_length,
        window='hann',
    )
    mag = np.abs(D) + 1e-10
    phase = np.angle(D)
    n_freq = mag.shape[0]
    n_frames = mag.shape[1]

    # 2. Log magnitude
    log_mag = np.log(mag)

    # 3. Cepstral envelope with smooth lifter (cosine roll-off, reduce Gibbs ringing)
    cep = np.fft.fft(log_mag, axis=0)  # n_freq × n_frames

    #   Smooth lifter: 全保留 → 余弦滚降 → 零
    #   相比硬截断(矩形窗), 余弦滚降减少包络在频域的振铃伪影
    n_keep = 20       # 前 20 个系数完全保留
    n_taper = 8       # 之后 8 个系数余弦滚降到零 (等效截止 ~24,保持原平滑度)
    half = n_freq // 2
    lifter = np.ones(n_freq)
    taper_start = min(n_keep + 1, half)
    taper_end = min(n_keep + 1 + n_taper, half + 1)
    n_actual_taper = taper_end - taper_start
    if n_actual_taper > 0:
        t = np.arange(n_actual_taper) / n_actual_taper
        lifter[taper_start:taper_end] = 0.5 * (1 + np.cos(np.pi * t))
    lifter[taper_end:half + 1] = 0
    lifter[half + 1:] = lifter[half:0:-1]  # mirror for conjugate symmetry

    cep_low = cep * lifter[:, np.newaxis]
    envelope = np.fft.ifft(cep_low, axis=0).real

    # 4. Fine structure = log_mag - envelope
    fine = log_mag - envelope

    # 5. Frequency warping of envelope
    orig_bins = np.arange(n_freq)
    warped_env = np.zeros_like(envelope)

    if ratio > 1.0:
        # 拉伸: 共振峰上移
        new_bins = orig_bins / ratio
        for f in range(n_frames):
            warped_env[:, f] = np.interp(
                new_bins, orig_bins, envelope[:, f],
                left=envelope[0, f], right=envelope[-1, f],
            )
        # 倾斜补偿: 拉伸后高频能量扩散, 轻微提升
        tilt_comp = np.log(ratio) * 0.5
        warped_env[:n_freq//2] += tilt_comp
    else:
        # 压缩: 共振峰下移
        new_bins = orig_bins * ratio
        for f in range(n_frames):
            warped_env[:, f] = np.interp(
                new_bins, orig_bins, envelope[:, f],
                left=envelope[0, f], right=envelope[-1, f],
            )

    # 6. Combine fine + warped envelope
    new_log_mag = fine + warped_env
    new_mag = np.exp(new_log_mag)

    # 7. ISTFT with original phase
    D_new = new_mag.astype(np.complex64) * np.exp(1j * phase)
    y = librosa.istft(
        D_new, hop_length=hop, win_length=win_length,
        window='hann', length=len(audio),
    )

    # 防极端比例过冲 (频谱扭曲可能引入瞬态增益)
    max_val = np.max(np.abs(y))
    if max_val > 0.99:
        y = y * (0.99 / max_val)

    return y.astype(np.float32)


# ════════════════════════════════════════════════════════════════════
#  LPC 模式 (备用, 仅当 STFT 产生 NaN 时降级使用)
# ════════════════════════════════════════════════════════════════════

def _lpc_formant_shift(audio: np.ndarray, ratio: float, order: int) -> np.ndarray:
    """LPC 共振峰移位 (v7 备用, 含帧级稳定性保护)"""
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

        try:
            roots = np.roots(a)
        except Exception:
            output[start:end] += windowed
            window_sum[start:end] += window ** 2
            continue

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
                    new_angle = np.clip(new_angle, 0.001, np.pi * 0.97)
                    new_r = mag * np.exp(1j * new_angle)
                    new_roots.append(new_r)
                    new_roots.append(np.conj(new_r))
                    found_pair = True
                    break
            if not found_pair:
                new_roots.append(r)
                used.add(j)

        new_roots = np.array(new_roots)
        try:
            new_a = np.poly(new_roots)
            new_a = np.real(new_a)
        except Exception:
            output[start:end] += windowed
            window_sum[start:end] += window ** 2
            continue

        try:
            poles = np.roots(new_a)
            if np.any(np.abs(poles) >= 1.0):
                scale = 0.95 / np.max(np.abs(poles))
                poles = poles * scale
                new_a = np.poly(poles)
                new_a = np.real(new_a)
        except Exception:
            output[start:end] += windowed
            window_sum[start:end] += window ** 2
            continue

        residual = lfilter(a, [1.0], windowed)
        synthesized = lfilter([1.0], new_a, residual)

        # 帧级稳定性检查
        frame_in_rms = np.sqrt(np.mean(frame ** 2))
        frame_out_rms = np.sqrt(np.mean(synthesized ** 2))
        frame_in_peak = np.max(np.abs(frame))
        frame_out_peak = np.max(np.abs(synthesized))
        use_fallback = False
        if not np.all(np.isfinite(synthesized)):
            use_fallback = True
        elif frame_in_rms > 1e-8 and frame_out_rms > frame_in_rms * 8:
            use_fallback = True
        elif frame_in_peak > 1e-8 and frame_out_peak > frame_in_peak * 5:
            use_fallback = True
        if use_fallback:
            synthesized = windowed

        output[start:end] += synthesized * window
        window_sum[start:end] += window ** 2

    nonzero = window_sum > 1e-8
    output[:len(audio)][nonzero[:len(audio)]] /= window_sum[:len(audio)][nonzero[:len(audio)]]
    output = output[:len(audio)]
    input_max = np.max(np.abs(audio))
    output_max = np.max(np.abs(output))
    if output_max > input_max * 5 and input_max > 1e-8:
        output = output * (input_max * 5 / output_max)
    return output.astype(np.float32)


# ════════════════════════════════════════════════════════════════════
#  对外接口
# ════════════════════════════════════════════════════════════════════

def formant_shift(audio: np.ndarray, sr: int, ratio: float,
                  order: int = LPC_ORDER) -> np.ndarray:
    """
    共振峰移位 (v8): 三模式策略

    模式选择:
      - |ratio-1.0| < 0.10: EQ 搁架滤波 (最干净, 无伪影)
      - |ratio-1.0| >= 0.10: STFT 倒谱包络扭曲 (主力, 所有比例稳定)
      - 降级: 如果 STFT 产生 NaN/Inf → LPC (含帧级稳定性保护)

    参考: 开源变声器通行做法 (WORLD vocoder 风格分析-合成)
    """
    if ratio == 1.0:
        return audio.copy()

    ratio = np.clip(ratio, 0.4, 2.5)

    if abs(ratio - 1.0) < 0.10:
        # 模式 A: EQ (cleanest)
        return _formant_eq_approx(audio, ratio)

    # 模式 B: STFT 倒谱包络扭曲 (主力)
    try:
        shifted = _stft_formant_shift(audio, ratio)
        if np.all(np.isfinite(shifted)):
            return shifted
    except Exception:
        pass

    # 降级: LPC (备用)
    return _lpc_formant_shift(audio, ratio, order)


def extract_formants(audio: np.ndarray, sr: int, order: int = LPC_ORDER) -> list:
    """提取共振峰频率 (LPC 法, 用于可视化)"""
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
