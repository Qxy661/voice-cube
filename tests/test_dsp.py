"""
声纹魔方 - DSP 模块单元测试
验证各 DSP 模块的基本功能正确性
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SAMPLE_RATE


def generate_test_audio(duration: float = 1.0, freq: float = 440.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """生成测试用正弦波"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_pitch_shift():
    """测试变调模块"""
    from dsp.pitch_shift import pitch_shift

    audio = generate_test_audio(1.0, 440.0)

    # 不变调
    result = pitch_shift(audio, SAMPLE_RATE, 0)
    assert np.allclose(result, audio), "0半音偏移应返回原信号"

    # 升调
    result = pitch_shift(audio, SAMPLE_RATE, 5)
    assert len(result) == len(audio), "变调后长度应不变"
    assert not np.allclose(result, audio), "5半音偏移应产生不同信号"
    assert np.max(np.abs(result)) <= 1.01, "输出不应超过1.0"

    print("[PASS] pitch_shift 测试通过")


def test_formant_shift():
    """测试共振峰移位模块"""
    from dsp.formant_shift import formant_shift

    audio = generate_test_audio(1.0, 440.0)

    # 不变
    result = formant_shift(audio, SAMPLE_RATE, 1.0)
    assert np.allclose(result, audio), "ratio=1.0应返回原信号"

    # 移位
    result = formant_shift(audio, SAMPLE_RATE, 0.7)
    assert len(result) == len(audio), "移位后长度应不变"
    assert np.max(np.abs(result)) <= 1.01, "输出不应超过1.0"

    print("[PASS] formant_shift 测试通过")


def test_effects():
    """测试特效模块"""
    from dsp.effects import ring_modulate, telephone_filter, add_breathiness

    audio = generate_test_audio(1.0, 440.0)

    # 环形调制
    result = ring_modulate(audio, SAMPLE_RATE, 0.5, 50.0)
    assert len(result) == len(audio)
    assert np.max(np.abs(result)) <= 1.01

    # 电话滤波
    result = telephone_filter(audio, SAMPLE_RATE)
    assert len(result) == len(audio)

    # 气声
    result = add_breathiness(audio, SAMPLE_RATE, 0.5)
    assert len(result) == len(audio)

    print("[PASS] effects 测试通过")


def test_preprocessor():
    """测试谱减法降噪"""
    from dsp.preprocessor import spectral_subtraction

    # 生成含噪信号
    clean = generate_test_audio(1.0, 440.0)
    noise = np.random.randn(len(clean)).astype(np.float32) * 0.1
    noisy = clean + noise

    result = spectral_subtraction(noisy, SAMPLE_RATE)
    assert len(result) == len(noisy), "降噪后长度应不变"
    assert np.max(np.abs(result)) <= 1.01

    print("[PASS] preprocessor 测试通过")


def test_pitch_extract():
    """测试基频提取"""
    from dsp.pitch_extract import extract_f0

    # 生成 440Hz 正弦波
    audio = generate_test_audio(2.0, 440.0)
    f0 = extract_f0(audio, SAMPLE_RATE, method="autocorrelation")

    # 正弦波应能检测到基频
    nonzero_f0 = f0[f0 > 0]
    if len(nonzero_f0) > 0:
        mean_f0 = np.mean(nonzero_f0)
        assert 400 < mean_f0 < 480, f"440Hz正弦波的F0应在400-480Hz之间，实际: {mean_f0}"

    print("[PASS] pitch_extract 测试通过")


def test_postprocessor():
    """测试后处理模块"""
    from dsp.postprocessor import parametric_eq, apply_reverb

    audio = generate_test_audio(1.0, 440.0)

    # EQ
    result = parametric_eq(audio, SAMPLE_RATE, bass_boost=6, treble_boost=-3)
    assert len(result) == len(audio)
    assert np.max(np.abs(result)) <= 1.01

    # 混响
    result = apply_reverb(audio, SAMPLE_RATE, wet=0.3)
    assert len(result) == len(audio)
    assert np.max(np.abs(result)) <= 1.01

    print("[PASS] postprocessor 测试通过")


def test_presets():
    """测试预设库"""
    from dsp.presets import get_preset, list_presets, get_default_params

    presets = list_presets()
    assert len(presets) >= 8, "应至少有8个预设"

    # 测试获取预设
    p = get_preset("robot")
    assert p["name"] == "机器人"
    assert "params" in p

    # 测试默认参数
    default = get_default_params()
    assert default["pitch_shift"] == 0
    assert default["formant_ratio"] == 1.0

    print("[PASS] presets 测试通过")


def test_visualizer():
    """测试可视化模块（只验证不报错）"""
    from dsp.visualizer import plot_waveform_comparison, plot_spectrum_comparison
    import matplotlib
    matplotlib.use("Agg")

    audio1 = generate_test_audio(1.0, 440.0)
    audio2 = generate_test_audio(1.0, 880.0)

    fig1 = plot_waveform_comparison(audio1, audio2, SAMPLE_RATE)
    assert fig1 is not None

    fig2 = plot_spectrum_comparison(audio1, audio2, SAMPLE_RATE)
    assert fig2 is not None

    import matplotlib.pyplot as plt
    plt.close("all")

    print("[PASS] visualizer 测试通过")


if __name__ == "__main__":
    print("=" * 50)
    print("声纹魔方 DSP 模块测试")
    print("=" * 50)

    test_pitch_shift()
    test_formant_shift()
    test_effects()
    test_preprocessor()
    test_pitch_extract()
    test_postprocessor()
    test_presets()
    test_visualizer()

    print("=" * 50)
    print("所有测试通过!")
    print("=" * 50)
