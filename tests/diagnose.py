"""
声纹魔方 - 深度诊断 (v7)
逐模块测试，精确定位失真来源
自动保存每个阶段的 .wav 文件用于人工听测
"""

import numpy as np
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE

OK = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
OUT_DIR = os.path.join(os.path.dirname(__file__), "diagnose_output")


def _save_wav(audio, sr, name):
    """保存 wav 用于人工听测"""
    import soundfile as sf
    os.makedirs(OUT_DIR, exist_ok=True)
    safe = name.replace(" ", "_").replace("=", "_").replace("+", "p").replace("-", "m").replace(",", "")
    path = os.path.join(OUT_DIR, f"{safe}.wav")
    sf.write(path, audio, sr)
    return path


def generate_speech_like_continuous(duration=1.5, sr=SAMPLE_RATE):
    """
    连续类语音信号: 无长静默段, 适中的 crest factor (6-12).
    模拟连续语音的包络起伏而非开关式静音.
    """
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    signal = np.zeros(n, dtype=np.float64)

    # 包络: 连续起伏, 没有完全静音
    envelope = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    # 元音 1: F0=200Hz, 连续性, 0-0.5s
    v1 = t < 0.5
    f0_1 = 200.0
    for h in range(1, 20):
        freq = h * f0_1
        env = 1.0
        for fmt in [800, 1400, 2800, 3800]:
            bw = fmt * 0.12
            env *= (bw**2) / ((freq - fmt)**2 + bw**2)
        signal[v1] += 0.2 / h * env * np.sin(2 * np.pi * freq * t[v1])

    # 过渡 0.4-0.6s (元音过渡, 非静音)
    trans = (t >= 0.4) & (t < 0.6)
    signal[trans] *= (1.0 - 0.5 * (t[trans] - 0.4) / 0.2)

    # 元音 2: F0=280Hz, 0.5-0.9s
    v2 = (t >= 0.5) & (t < 0.9)
    f0_2 = 280.0
    for h in range(1, 20):
        freq = h * f0_2
        env = 1.0
        for fmt in [900, 1600, 2800, 3800]:
            bw = fmt * 0.12
            env *= (bw**2) / ((freq - fmt)**2 + bw**2)
        signal[v2] += 0.2 / h * env * np.sin(2 * np.pi * freq * t[v2])

    # 摩擦音 (叠加在元音上, 不产生静音)
    fric = (t >= 0.7) & (t < 0.85)
    from scipy.signal import butter, sosfilt
    noise = np.random.randn(np.sum(fric)) * 0.06
    sos = butter(4, [3000/(sr/2), 8000/(sr/2)], btype="band", output="sos")
    signal[fric] += sosfilt(sos, noise)

    # 元音 3: F0=220Hz, 0.9-1.4s
    v3 = t >= 0.9
    f0_3 = 220.0
    for h in range(1, 20):
        freq = h * f0_3
        env = 1.0
        for fmt in [700, 1200, 2600, 3600]:
            bw = fmt * 0.12
            env *= (bw**2) / ((freq - fmt)**2 + bw**2)
        signal[v3] += 0.2 / h * env * np.sin(2 * np.pi * freq * t[v3])

    signal = signal * envelope
    max_val = np.max(np.abs(signal))
    if max_val > 0: signal = signal * 0.5 / max_val
    return signal.astype(np.float32)


def measure(audio, name=""):
    """测量信号特征"""
    if len(audio) == 0:
        return {"name": name, "valid": False, "rms": 0, "peak": 0, "crest": 0}
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    crest = peak / (rms + 1e-10)
    fl = int(0.03 * SAMPLE_RATE)
    hop = fl // 2
    nf = max(1, (len(audio) - fl) // hop)
    frms = np.array([np.sqrt(np.mean(audio[i*hop:i*hop+fl]**2)) for i in range(nf)])
    active = frms > (np.mean(frms) * 0.3) if np.mean(frms) > 0 else np.ones_like(frms, dtype=bool)
    active_ratio = float(np.mean(active))
    has_nan = bool(np.any(~np.isfinite(audio)))
    return {
        "name": name, "valid": True,
        "peak": peak, "rms": rms, "crest": crest,
        "active_ratio": active_ratio,
        "has_nan": has_nan,
    }


def check(m):
    """检查指标是否正常"""
    if not m["valid"]:
        return False, "空信号"
    issues = []
    if m["has_nan"]:
        issues.append("含 NaN/Inf!")
    if m["peak"] > 1.01:
        issues.append(f"削波 peak={m['peak']:.3f}")
    if m["rms"] < 1e-6:
        issues.append("完全静音")
    if m["rms"] < 0.005:
        issues.append(f"电平过低 rms={m['rms']:.5f}")
    if m["crest"] > 30:
        issues.append(f"峰值因子过高 crest={m['crest']:.1f}")
    if m["crest"] < 1.5:
        issues.append(f"动态范围被压扁 crest={m['crest']:.1f}")
    if m["active_ratio"] > 0.99:
        issues.append(f"所有帧都有能量 (平坦)")
    if m["active_ratio"] < 0.1:
        issues.append("几乎所有帧都静音")
    return (len(issues) == 0, "; ".join(issues)) if issues else (True, "")


def print_measure(m, prefix=""):
    ok, reason = check(m)
    status = OK if ok else FAIL
    extra = f" ({reason})" if reason else ""
    print(f"  {status} {prefix}{m['name']}: peak={m['peak']:.4f} rms={m['rms']:.5f} crest={m['crest']:.1f}{extra}")


# ════════════════════════════════════════════════════════════════════
# 逐模块测试
# ════════════════════════════════════════════════════════════════════

def test_pitch_shift():
    print("\n--- 1. pitch_shift ---")
    from dsp.pitch_shift import pitch_shift
    audio = generate_speech_like_continuous(1.0)
    _save_wav(audio, SAMPLE_RATE, "00_input")
    all_pass = True

    for semis in [0, -2, 2, -5, 5, -12, 12]:
        out = pitch_shift(audio, SAMPLE_RATE, semis)
        m = measure(out, f"pitch={semis:+.0f}")
        ok, reason = check(m)
        if not ok: all_pass = False
        print_measure(m)
        if not ok:
            path = _save_wav(out, SAMPLE_RATE, f"pitch_{semis}")
            print(f"    -> 已保存: {os.path.basename(path)}")

    return all_pass


def test_formant_shift():
    print("\n--- 2. formant_shift ---")
    from dsp.formant_shift import formant_shift
    audio = generate_speech_like_continuous(1.0)
    all_pass = True

    # EQ模式 (|ratio-1| < 0.20)
    eq_ratios = [1.0, 1.05, 0.95, 1.10, 0.90, 1.15, 0.85]
    print("  [EQ模式]")
    for r in eq_ratios:
        out = formant_shift(audio, SAMPLE_RATE, r)
        m = measure(out, f"ratio={r:.2f}")
        ok, reason = check(m)
        if not ok: all_pass = False
        print_measure(m)

    # LPC模式 (|ratio-1| >= 0.20)
    lpc_ratios = [0.8, 0.7, 0.5, 1.2, 1.3, 1.5, 2.0]
    print("  [LPC模式]")
    for r in lpc_ratios:
        out = formant_shift(audio, SAMPLE_RATE, r)
        m = measure(out, f"ratio={r:.2f}")
        ok, reason = check(m)
        if not ok:
            all_pass = False
            if m["has_nan"]:
                print(f"    -> 原因: 含 NaN/Inf!")
            elif m["peak"] > 2.0:
                print(f"    -> 原因: 严重过冲 peak={m['peak']:.4f}")
            path = _save_wav(out, SAMPLE_RATE, f"formant_lpc_{r}")
            print(f"    -> 已保存: {os.path.basename(path)}")
        print_measure(m)

    # 帧间连续性检查
    for r in [1.2, 1.5, 0.7]:
        out = formant_shift(audio, SAMPLE_RATE, r)
        diff = np.diff(out)
        md = np.max(np.abs(diff))
        if md > 0.5:
            print(f"  {WARN} LPC ratio={r:.1f} max_diff={md:.4f}")
            _save_wav(out, SAMPLE_RATE, f"formant_click_check_r{r}")
        else:
            print(f"  {OK} LPC ratio={r:.1f} 连续 max_diff={md:.4f}")

    return all_pass


def test_reverb():
    print("\n--- 3. reverb ---")
    from dsp.postprocessor import apply_reverb
    audio = generate_speech_like_continuous(1.0)
    all_pass = True

    for decay in [0.0, 0.05, 0.2, 0.5, 0.8, 1.0]:
        out = apply_reverb(audio, SAMPLE_RATE, decay)
        m = measure(out, f"decay={decay:.2f}")
        ok, reason = check(m)
        if not ok: all_pass = False
        print_measure(m)

    return all_pass


def test_eq():
    print("\n--- 4. parametric_eq ---")
    from dsp.postprocessor import parametric_eq
    audio = generate_speech_like_continuous(1.0)
    all_pass = True

    cases = [(0, 0), (6, 0), (0, 6), (-6, -6), (12, 0), (0, 12), (-12, -12)]
    for bass, treble in cases:
        out = parametric_eq(audio, SAMPLE_RATE, bass, treble)
        m = measure(out, f"bass={bass:+.0f},treble={treble:+.0f}")
        ok, reason = check(m)
        if not ok: all_pass = False
        print_measure(m)

    return all_pass


def test_full_pipeline():
    """完整 DSP + RVC 管线"""
    print("\n--- 5. 完整管线 ---")
    from app import apply_dsp_preset
    from ai.rvc_engine import RVCEngine
    audio = generate_speech_like_continuous(1.5)
    _save_wav(audio, SAMPLE_RATE, "input_original")
    all_pass = True

    # DSP presets
    print("  [DSP预设]")
    presets = ["deep_male", "child", "robot", "minion", "giant", "whisper", "metallic", "old_telephone"]
    for preset in presets:
        processed, _ = apply_dsp_preset(audio, SAMPLE_RATE, preset)
        m = measure(processed, preset)
        ok, reason = check(m)
        if not ok: all_pass = False
        print_measure(m)
        if not ok:
            _save_wav(processed, SAMPLE_RATE, f"dsp_{preset}")

    # RVC simulation
    print("  [RVC模拟]")
    engine = RVCEngine()
    for target in ["kobe", "spongebob"]:
        engine.load_model(target)
        out = engine.infer(audio.copy(), SAMPLE_RATE)
        m = measure(out, target)
        ok, reason = check(m)
        if not ok:
            all_pass = False
            print(f"  {FAIL} {target}: {reason}")
            # 深度诊断
            if m["has_nan"]:
                print(f"    -> NaN/Inf!")
            if np.allclose(out, audio):
                print(f"    -> 输出与输入完全相同!")
            if m["peak"] < 0.001:
                print(f"    -> 几乎静音!")
            if m["crest"] < 2.0:
                print(f"    -> 动态被压扁, 输出平坦")
            _save_wav(out, SAMPLE_RATE, f"rvc_{target}")
        else:
            print_measure(m)

    return all_pass


def test_stage_by_stage():
    """逐级追踪 deep_male 管线, 保存每步 wav"""
    print("\n--- 6. 逐级追踪 ---")
    from dsp.pitch_shift import pitch_shift
    from dsp.formant_shift import formant_shift
    from dsp.postprocessor import apply_reverb, parametric_eq, normalize_loudness
    from app import _soft_limiter
    from dsp.presets import get_preset

    audio = generate_speech_like_continuous(1.0)
    params = get_preset("deep_male")["params"]
    print(f"  参数: pitch={params['pitch_shift']}, formant={params['formant_ratio']}, "
          f"reverb={params['reverb']}, bass={params['eq_bass_boost']}, treble={params['eq_treble_boost']}")

    stages = [
        ("01_input", lambda x: x.copy()),
        ("02_pitch", lambda x: pitch_shift(x, SAMPLE_RATE, params["pitch_shift"])),
        ("03_formant", lambda x: formant_shift(x, SAMPLE_RATE, params["formant_ratio"])),
        ("04_reverb", lambda x: apply_reverb(x, SAMPLE_RATE, params["reverb"])),
        ("05_eq", lambda x: parametric_eq(x, SAMPLE_RATE, params["eq_bass_boost"], params["eq_treble_boost"])),
        ("06_soft_limit", lambda x: _soft_limiter(x)),
        ("07_lufs", lambda x: normalize_loudness(x, SAMPLE_RATE)),
    ]

    x = audio
    for name, fn in stages:
        x = fn(x)
        m = measure(x, name[3:])
        ok, reason = check(m)
        status = OK if ok else FAIL
        extra = f" ({reason})" if reason else ""
        print(f"  {status} {name[3:]}: peak={m['peak']:.4f} rms={m['rms']:.5f} crest={m['crest']:.1f}{extra}")
        _save_wav(x, SAMPLE_RATE, f"trace_{name}")

    return np.all(np.isfinite(x)) and np.max(np.abs(x)) <= 1.0


def main():
    print("=" * 60)
    print("  声纹魔方 v7 深度诊断")
    print(f"  采样率: {SAMPLE_RATE} Hz")
    print(f"  输出目录: {OUT_DIR}")
    print("=" * 60)

    # Clean output dir
    import shutil
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    results = []
    tests = [
        ("pitch_shift", test_pitch_shift),
        ("formant_shift", test_formant_shift),
        ("reverb", test_reverb),
        ("EQ", test_eq),
        ("完整管线", test_full_pipeline),
        ("逐级追踪", test_stage_by_stage),
    ]

    for name, fn in tests:
        try:
            ok = fn()
            results.append((name, ok))
            print(f"  {OK if ok else FAIL} [{name}]")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {FAIL} [{name}]: {e}")
            results.append((name, False))
        print()

    print("=" * 60)
    print("  诊断结论")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        print(f"  {OK if ok else FAIL} {name}")
        if not ok: all_pass = False

    if all_pass:
        print(f"\n  {OK} 所有模块指标正常")
    else:
        print(f"\n  {FAIL} 存在模块指标异常, 检查 diagnose_output/*.wav 听测")

    return int(not all_pass)


if __name__ == "__main__":
    sys.exit(main())
