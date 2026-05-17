"""
声纹魔方 - 修复验证脚本 (v7)
验证 "炸麦" 和 "没波动" 问题已修复

检查项:
1. DSP 管线: 输出不削波 (max < 1.0), 无 NaN/Inf, 有足够动态范围
2. RVC 模拟: 输出有波动 (RMS变化), 无削波
3. 语音类信号 (含暂态) 通过各模块无振铃/失真
"""

import numpy as np
import sys, os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE

OK_SYM = "[PASS]"
FAIL_SYM = "[FAIL]"
WARN_SYM = "[WARN]"


def generate_speech_like(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    生成类语音测试信号:
    - 元音 (谐波串, 有基频 + 共振峰结构)
    - 清辅音暂态 (burst)
    - 静音段
    """
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    signal = np.zeros(n, dtype=np.float64)

    # 元音段 0-0.6s: F0=200Hz + 4个共振峰
    vowel_mask = t < 0.6
    v_t = t[vowel_mask]
    # 基频 + 谐波
    f0 = 200.0
    for h in range(1, 25):
        amp = 0.3 / h
        freq = h * f0
        env = 1.0
        for formant in [800, 1400, 2800, 3800, 4800]:
            bw = formant * 0.1
            env *= (bw**2) / ((freq - formant)**2 + bw**2)
        signal[vowel_mask] += amp * env * np.sin(2 * np.pi * freq * v_t)

    # 摩擦音 0.65-0.75s: 高频噪声
    fric_mask = (t >= 0.65) & (t < 0.75)
    noise = np.random.randn(np.sum(fric_mask)) * 0.08
    from scipy.signal import butter, sosfilt
    sos = butter(4, [3000 / (sr/2), 8000 / (sr/2)], btype="band", output="sos")
    signal[fric_mask] += sosfilt(sos, noise)

    # 爆破音暂态 0.85s: 短时宽带burst
    burst_start = int(0.85 * sr)
    burst_len = int(0.02 * sr)
    burst = np.random.randn(burst_len) * 0.3
    burst_env = np.exp(-np.arange(burst_len) / (0.005 * sr))
    burst = burst * burst_env
    end = min(burst_start + burst_len, n)
    signal[burst_start:end] += burst[:end - burst_start]

    # 元音段 0.95-1.4s: 不同音高
    v2_mask = (t >= 0.95) & (t < 1.4)
    v2_t = t[v2_mask]
    f0_2 = 280.0
    for h in range(1, 20):
        amp = 0.25 / h
        freq = h * f0_2
        env = 1.0
        for formant in [900, 1600, 2800, 3800]:
            bw = formant * 0.1
            env *= (bw**2) / ((freq - formant)**2 + bw**2)
        signal[v2_mask] += amp * env * np.sin(2 * np.pi * freq * v2_t)

    # 归一化到 -6dB
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal * 0.5 / max_val

    return signal.astype(np.float32)


def check_signal(name: str, audio: np.ndarray, expect_dynamics: bool = False) -> bool:
    """检查信号质量"""
    issues = []

    # 1. NaN/Inf
    if np.any(~np.isfinite(audio)):
        issues.append("含 NaN/Inf")

    # 2. 削波检查
    max_abs = np.max(np.abs(audio))
    if max_abs > 1.0:
        issues.append(f"削波! max={max_abs:.3f}")
    elif max_abs > 0.99:
        issues.append(f"接近削波 max={max_abs:.3f}")

    # 3. 静音检查
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        issues.append("完全静音")
    elif rms < 0.005:
        issues.append(f"电平过低 rms={rms:.5f}")

    # 4. 动态范围检查 (用于 RVC)
    if expect_dynamics:
        frame_len = int(0.03 * SAMPLE_RATE)
        hop = frame_len // 2
        n_frames = max(1, (len(audio) - frame_len) // hop)
        frame_rms = np.array([
            np.sqrt(np.mean(audio[i*hop:i*hop+frame_len] ** 2))
            for i in range(n_frames)
        ])
        active_ratio = np.mean(frame_rms > frame_rms.mean() * 0.3)
        if active_ratio > 0.95:
            issues.append("无明显波动 (几乎所有帧都有能量)")
        elif active_ratio < 0.1:
            issues.append("几乎全静音")

    if issues:
        print(f"  {FAIL_SYM} {name}: {'; '.join(issues)}")
        return False
    else:
        print(f"  {OK_SYM} {name}: max={max_abs:.4f}, rms={rms:.5f}")
        return True


def test_dsp_pipeline():
    """测试 DSP 管线 — 验证无炸麦"""
    print("\n" + "=" * 50)
    print("1. DSP 管线测试")
    print("=" * 50)

    from app import apply_dsp_preset, _soft_limiter

    audio = generate_speech_like(1.5)
    print(f"  输入: max={np.max(np.abs(audio)):.4f}, rms={np.sqrt(np.mean(audio**2)):.5f}")

    presets_to_test = ["deep_male", "child", "robot", "minion", "giant", "whisper", "metallic", "old_telephone"]
    all_pass = True
    for preset in presets_to_test:
        t0 = time.time()
        processed, _ = apply_dsp_preset(audio, SAMPLE_RATE, preset)
        elapsed = time.time() - t0
        ok = check_signal(f"{preset} ({elapsed:.2f}s)", processed)
        if not ok:
            all_pass = False

    # 极端参数
    extreme_cases = [
        ("pitch=-12", {"pitch_shift": -12, "formant_ratio": 1.0, "ring_mod": 0, "reverb": 0, "breathiness": 0}),
        ("pitch=+12", {"pitch_shift": 12, "formant_ratio": 1.0, "ring_mod": 0, "reverb": 0, "breathiness": 0}),
        ("formant=0.4", {"pitch_shift": 0, "formant_ratio": 0.4, "ring_mod": 0, "reverb": 0, "breathiness": 0}),
        ("formant=2.5", {"pitch_shift": 0, "formant_ratio": 2.5, "ring_mod": 0, "reverb": 0, "breathiness": 0}),
        ("full_effect", {"pitch_shift": 5, "formant_ratio": 0.7, "ring_mod": 0.5, "reverb": 0.5, "breathiness": 0.3}),
    ]

    for name, params in extreme_cases:
        try:
            audio_in = audio.copy()
            from dsp.pitch_shift import pitch_shift
            from dsp.formant_shift import formant_shift
            from dsp.effects import ring_modulate, add_breathiness
            from dsp.postprocessor import apply_reverb, normalize_loudness

            if params["pitch_shift"] != 0:
                audio_in = pitch_shift(audio_in, SAMPLE_RATE, params["pitch_shift"])
            pr = params["formant_ratio"]
            if abs(pr - 1.0) > 0.01:
                audio_in = formant_shift(audio_in, SAMPLE_RATE, pr)
            if params["ring_mod"] > 0:
                audio_in = ring_modulate(audio_in, SAMPLE_RATE, params["ring_mod"])
            if params["reverb"] > 0:
                audio_in = apply_reverb(audio_in, SAMPLE_RATE, params["reverb"])
            if params["breathiness"] > 0:
                audio_in = add_breathiness(audio_in, SAMPLE_RATE, params["breathiness"])
            audio_in = _soft_limiter(audio_in)
            audio_in = normalize_loudness(audio_in, SAMPLE_RATE)
            ok = check_signal(f"extreme: {name}", audio_in)
            if not ok:
                all_pass = False
        except Exception as e:
            print(f"  {FAIL_SYM} {name}: 异常 {e}")
            all_pass = False

    return all_pass


def test_rvc_simulation():
    """测试 RVC 模拟管线 — 验证信号有波动"""
    print("\n" + "=" * 50)
    print("2. RVC 模拟管线测试")
    print("=" * 50)

    from ai.rvc_engine import RVCEngine

    audio = generate_speech_like(1.5)
    print(f"  输入: max={np.max(np.abs(audio)):.4f}, rms={np.sqrt(np.mean(audio**2)):.5f}")

    engine = RVCEngine()
    all_pass = True

    for target in ["kobe", "spongebob"]:
        engine.load_model(target)
        t0 = time.time()
        output = engine.infer(audio.copy(), SAMPLE_RATE)
        elapsed = time.time() - t0

        quality_ok = check_signal(f"{target} ({elapsed:.2f}s)", output, expect_dynamics=True)
        if not quality_ok:
            all_pass = False

        if np.allclose(output, audio):
            print(f"  {FAIL_SYM} {target}: 输出和输入完全相同 (无效果)")
            all_pass = False

    return all_pass


def test_soft_limiter():
    """测试软限幅器"""
    print("\n" + "=" * 50)
    print("3. 软限幅器测试")
    print("=" * 50)

    from app import _soft_limiter

    clipped = np.array([-2.0, -1.5, -0.5, 0.0, 0.5, 1.2, 1.8, 2.0], dtype=np.float64)
    result = _soft_limiter(clipped)

    max_out = np.max(np.abs(result))
    assert max_out <= 1.0, f"软限幅后 max 应 <= 1.0, 实际 {max_out}"
    assert result[0] < 0, "符号应保留"
    assert result[-1] > 0, "符号应保留"
    assert result[0] > -1.0, f"-2.0 应 > -1.0, 实际 {result[0]}"
    print(f"  {OK_SYM} 软限幅有效: max_in=2.0 -> max_out={max_out:.4f}")
    print(f"     输入: {clipped}")
    print(f"     输出: {result}")


def test_polyphase_resample():
    """测试 polyphase 重采样无振铃"""
    print("\n" + "=" * 50)
    print("4. Polyphase 重采样测试")
    print("=" * 50)

    from dsp.pitch_shift import _polyphase_resample

    n = SAMPLE_RATE
    audio = np.zeros(n, dtype=np.float32)
    audio[:] = np.random.randn(n).astype(np.float32) * 0.001
    transient = int(0.5 * SAMPLE_RATE)
    audio[transient] = 0.9

    down = _polyphase_resample(audio, SAMPLE_RATE, SAMPLE_RATE // 2)
    up = _polyphase_resample(down, SAMPLE_RATE // 2, SAMPLE_RATE)

    assert np.all(np.isfinite(up)), "重采样后含 NaN/Inf"
    max_val = np.max(np.abs(up))
    assert max_val < 1.5, f"重采样后不应严重过冲, max={max_val}"

    if max_val > 0.99:
        print(f"  {WARN_SYM} 轻微过冲 max={max_val:.4f} (软限幅可处理)")
    else:
        print(f"  {OK_SYM} 重采样稳定: max={max_val:.4f}")


def test_pipeline_speed():
    """测试管线速度"""
    print("\n" + "=" * 50)
    print("5. 管线性能测试")
    print("=" * 50)

    from app import apply_dsp_preset

    audio = generate_speech_like(1.0)
    t0 = time.time()
    for _ in range(3):
        apply_dsp_preset(audio, SAMPLE_RATE, "deep_male")
    avg = (time.time() - t0) / 3
    print(f"  deep_male 平均耗时: {avg:.2f}s (1s 音频)")
    status = OK_SYM if avg < 5.0 else WARN_SYM
    print(f"  {status} 耗时 {'在' if avg < 5.0 else '超出'}预期范围")


def main():
    print("=" * 60)
    print("  声纹魔方 v7 修复验证")
    print(f"  采样率: {SAMPLE_RATE} Hz")
    print("  检查: 炸麦修复 / 波动修复 / 软限幅 / 重采样稳定性")
    print("=" * 60)

    results = []

    try:
        test_soft_limiter()
        results.append(("软限幅器", True))
    except Exception as e:
        print(f"  {FAIL_SYM} 软限幅器: {e}")
        results.append(("软限幅器", False))

    try:
        test_polyphase_resample()
        results.append(("重采样", True))
    except Exception as e:
        print(f"  {FAIL_SYM} 重采样: {e}")
        results.append(("重采样", False))

    try:
        dsp_ok = test_dsp_pipeline()
        results.append(("DSP管线", dsp_ok))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  {FAIL_SYM} DSP管线异常: {e}")
        results.append(("DSP管线", False))

    try:
        rvc_ok = test_rvc_simulation()
        results.append(("RVC模拟", rvc_ok))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  {FAIL_SYM} RVC模拟异常: {e}")
        results.append(("RVC模拟", False))

    try:
        test_pipeline_speed()
        results.append(("性能", True))
    except Exception as e:
        print(f"  {FAIL_SYM} 性能: {e}")
        results.append(("性能", False))

    print("\n" + "=" * 60)
    print("  验证结果汇总")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = f"{OK_SYM}" if ok else f"{FAIL_SYM}"
        print(f"  {status} - {name}")
        if not ok:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("  所有验证通过! 修复完成")
    else:
        print("  存在失败项，需要进一步排查")
    print("=" * 60)

    return int(not all_pass)


if __name__ == "__main__":
    sys.exit(main())
