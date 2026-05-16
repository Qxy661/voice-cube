"""
声纹魔方 - DSP 模块单元测试 (v3)
验证各 DSP 模块的基本功能正确性

v3: 新增9个测试（压缩器/激励器/LUFS/噪声门/梳状滤波/共振峰提取/F0-MIDI/RVC引擎/录音组件）
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SAMPLE_RATE


def generate_test_audio(duration: float = 1.0, freq: float = 440.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """生成测试用正弦波"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def generate_multitone(duration: float = 1.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """生成多频测试信号（含低中高频）"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.4 * np.sin(2 * np.pi * 1000 * t) +
        0.3 * np.sin(2 * np.pi * 4000 * t)
    )
    return signal.astype(np.float32)


# ════════════════════════════════════════════════════════════════════
#  原有测试
# ════════════════════════════════════════════════════════════════════

def test_pitch_shift():
    """测试变调模块"""
    from dsp.pitch_shift import pitch_shift

    audio = generate_test_audio(1.0, 440.0)

    result = pitch_shift(audio, SAMPLE_RATE, 0)
    assert np.allclose(result, audio), "0半音偏移应返回原信号"

    result = pitch_shift(audio, SAMPLE_RATE, 5)
    assert len(result) == len(audio), "变调后长度应不变"
    assert not np.allclose(result, audio), "5半音偏移应产生不同信号"
    assert np.max(np.abs(result)) <= 1.01, "输出不应超过1.0"

    print("[PASS] pitch_shift")


def test_formant_shift():
    """测试共振峰移位模块"""
    from dsp.formant_shift import formant_shift

    audio = generate_test_audio(1.0, 440.0)

    result = formant_shift(audio, SAMPLE_RATE, 1.0)
    assert np.allclose(result, audio), "ratio=1.0应返回原信号"

    result = formant_shift(audio, SAMPLE_RATE, 0.7)
    assert len(result) == len(audio), "移位后长度应不变"
    assert np.max(np.abs(result)) <= 1.01, "输出不应超过1.0"

    print("[PASS] formant_shift")


def test_effects():
    """测试特效模块"""
    from dsp.effects import ring_modulate, telephone_filter, add_breathiness

    audio = generate_test_audio(1.0, 440.0)

    result = ring_modulate(audio, SAMPLE_RATE, 0.5, 50.0)
    assert len(result) == len(audio)
    assert np.max(np.abs(result)) <= 1.01

    result = telephone_filter(audio, SAMPLE_RATE)
    assert len(result) == len(audio)

    result = add_breathiness(audio, SAMPLE_RATE, 0.5)
    assert len(result) == len(audio)

    print("[PASS] effects")


def test_preprocessor():
    """测试谱减法降噪"""
    from dsp.preprocessor import spectral_subtraction

    clean = generate_test_audio(1.0, 440.0)
    noise = np.random.randn(len(clean)).astype(np.float32) * 0.1
    noisy = clean + noise

    result = spectral_subtraction(noisy, SAMPLE_RATE)
    assert len(result) == len(noisy), "降噪后长度应不变"
    assert np.max(np.abs(result)) <= 1.01

    print("[PASS] preprocessor")


def test_pitch_extract():
    """测试基频提取"""
    from dsp.pitch_extract import extract_f0

    audio = generate_test_audio(2.0, 440.0)
    f0 = extract_f0(audio, SAMPLE_RATE, method="autocorrelation")

    nonzero_f0 = f0[f0 > 0]
    if len(nonzero_f0) > 0:
        mean_f0 = np.mean(nonzero_f0)
        assert 400 < mean_f0 < 480, f"440Hz正弦波的F0应在400-480Hz之间，实际: {mean_f0}"

    print("[PASS] pitch_extract")


def test_postprocessor():
    """测试后处理模块"""
    from dsp.postprocessor import parametric_eq, apply_reverb

    audio = generate_test_audio(1.0, 440.0)

    result = parametric_eq(audio, SAMPLE_RATE, bass_boost=6, treble_boost=-3)
    assert len(result) == len(audio)
    assert np.max(np.abs(result)) <= 1.01

    result = apply_reverb(audio, SAMPLE_RATE, wet=0.3)
    assert len(result) == len(audio)
    assert np.max(np.abs(result)) <= 1.01

    print("[PASS] postprocessor")


def test_presets():
    """测试预设库"""
    from dsp.presets import get_preset, list_presets, get_default_params

    presets = list_presets()
    assert len(presets) >= 8, "应至少有8个预设"

    p = get_preset("robot")
    assert p["name"] == "机器人"
    assert "params" in p

    default = get_default_params()
    assert default["pitch_shift"] == 0
    assert default["formant_ratio"] == 1.0

    print("[PASS] presets")


def test_visualizer():
    """测试可视化模块"""
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

    print("[PASS] visualizer")


# ════════════════════════════════════════════════════════════════════
#  v3 新增测试
# ════════════════════════════════════════════════════════════════════

def test_soft_knee_compressor():
    """测试软拐点压缩器：验证动态范围被压缩"""
    from dsp.effects import soft_knee_compressor

    # 生成幅度差异大的信号（安静+响亮）
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    quiet = 0.05 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    loud = 0.8 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    signal = np.concatenate([quiet, loud])

    compressed = soft_knee_compressor(signal, sr=SAMPLE_RATE)

    assert len(compressed) == len(signal), "压缩后长度应不变"
    assert np.max(np.abs(compressed)) <= 1.01, "输出不应削波"

    # 压缩器应让安静部分相对变响（动态范围减小）
    quiet_rms_orig = np.sqrt(np.mean(quiet ** 2))
    loud_rms_orig = np.sqrt(np.mean(loud ** 2))
    quiet_rms_comp = np.sqrt(np.mean(compressed[:len(quiet)] ** 2))
    loud_rms_comp = np.sqrt(np.mean(compressed[len(quiet):] ** 2))

    ratio_orig = loud_rms_orig / (quiet_rms_orig + 1e-10)
    ratio_comp = loud_rms_comp / (quiet_rms_comp + 1e-10)
    assert ratio_comp < ratio_orig, "压缩后动态范围应减小"

    print("[PASS] soft_knee_compressor")


def test_harmonics_exciter():
    """测试谐波激励器：验证高频能量增加"""
    from dsp.effects import harmonics_exciter
    from scipy.signal import welch

    audio = generate_test_audio(1.0, 440.0)
    excited = harmonics_exciter(audio, SAMPLE_RATE, amount=0.3)

    assert len(excited) == len(audio), "处理后长度应不变"
    assert np.max(np.abs(excited)) <= 1.01, "输出不应削波"
    assert not np.allclose(excited, audio), "激励后信号应有变化"

    # 检查高频段能量是否增加
    f_orig, psd_orig = welch(audio, fs=SAMPLE_RATE, nperseg=2048)
    f_exc, psd_exc = welch(excited, fs=SAMPLE_RATE, nperseg=2048)
    mask = (f_orig >= 2000) & (f_orig <= 8000)
    energy_orig = np.mean(psd_orig[mask])
    energy_exc = np.mean(psd_exc[mask])
    assert energy_exc > energy_orig, "2-8kHz频段能量应增加"

    print("[PASS] harmonics_exciter")


def test_normalize_loudness():
    """测试 LUFS 响度归一化"""
    from dsp.postprocessor import normalize_loudness

    audio = generate_test_audio(2.0, 440.0)
    # 故意降低音量
    quiet_audio = (audio * 0.1).astype(np.float32)

    normalized = normalize_loudness(quiet_audio, SAMPLE_RATE, target_lufs=-16.0)

    assert len(normalized) == len(quiet_audio), "归一化后长度应不变"
    assert np.max(np.abs(normalized)) <= 1.01, "输出不应削波"

    # 归一化后 RMS 应比原始安静信号大
    rms_orig = np.sqrt(np.mean(quiet_audio ** 2))
    rms_norm = np.sqrt(np.mean(normalized ** 2))
    assert rms_norm > rms_orig, "归一化后响度应提升"

    print("[PASS] normalize_loudness")


def test_noise_gate():
    """测试噪声门：验证静音段被衰减"""
    from dsp.preprocessor import noise_gate

    # 生成信号：前半段静音，后半段有声音
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    signal = generate_test_audio(1.0, 440.0) * 0.5
    combined = np.concatenate([silence, signal])

    gated = noise_gate(combined, threshold=0.01, sr=SAMPLE_RATE)

    assert len(gated) == len(combined), "处理后长度应不变"

    # 静音段应被衰减
    silence_rms = np.sqrt(np.mean(gated[:SAMPLE_RATE] ** 2))
    signal_rms = np.sqrt(np.mean(gated[SAMPLE_RATE:] ** 2))
    assert silence_rms < 0.01, f"静音段应被门控衰减，实际RMS: {silence_rms}"
    assert signal_rms > 0.01, f"有声段应保留，实际RMS: {signal_rms}"

    print("[PASS] noise_gate")


def test_comb_filter_metallic():
    """测试金属质感梳状滤波器"""
    from dsp.effects import comb_filter_metallic

    audio = generate_test_audio(1.0, 440.0)
    result = comb_filter_metallic(audio, SAMPLE_RATE, freq=1000.0)

    assert len(result) == len(audio), "处理后长度应不变"
    assert np.max(np.abs(result)) <= 1.01, "输出不应削波"
    assert not np.allclose(result, audio), "梳状滤波应产生不同信号"

    print("[PASS] comb_filter_metallic")


def test_extract_formants():
    """测试共振峰提取"""
    from dsp.formant_shift import extract_formants

    # 用多频信号模拟有共振峰的语音
    audio = generate_multitone(1.0)
    formants = extract_formants(audio, SAMPLE_RATE)

    assert isinstance(formants, list), "应返回列表"
    # 即使没有明显的共振峰，也不应报错
    for f in formants:
        assert f > 0, f"共振峰频率应为正数，实际: {f}"
        assert f < SAMPLE_RATE / 2, f"共振峰频率应低于奈奎斯特频率，实际: {f}"

    print("[PASS] extract_formants")


def test_f0_to_midi():
    """测试 F0 到 MIDI 转换"""
    from dsp.pitch_extract import f0_to_midi

    # 440Hz = MIDI 69 (A4)
    f0 = np.array([0, 440, 880, 261.63])
    midi = f0_to_midi(f0)

    assert midi[0] == 0, "F0=0 应映射为 MIDI 0"
    assert abs(midi[1] - 69) < 0.5, f"440Hz应映射为MIDI 69，实际: {midi[1]}"
    assert abs(midi[2] - 81) < 0.5, f"880Hz应映射为MIDI 81，实际: {midi[2]}"
    assert abs(midi[3] - 60) < 0.5, f"261.63Hz应映射为MIDI 60，实际: {midi[3]}"

    print("[PASS] f0_to_midi")


def test_rvc_engine():
    """测试 RVC 引擎（模拟模式）"""
    from ai.rvc_engine import RVCEngine

    engine = RVCEngine()

    # 初始化
    info = engine.get_model_info()
    assert info["mode"] in ["RVC", "DSP模拟"], "模式应为 RVC 或 DSP模拟"

    # 加载模型（模拟模式）
    result = engine.load_model("kobe")
    assert result, "模型加载应成功"

    # 推理
    audio = generate_test_audio(1.0, 440.0)
    output = engine.infer(audio, SAMPLE_RATE)
    assert len(output) == len(audio), "推理后长度应不变"
    assert np.max(np.abs(output)) <= 1.01, "输出不应削波"

    print("[PASS] rvc_engine")


def test_recorder():
    """测试录音组件"""
    from ui.recorder import process_audio_input, normalize_audio, remove_dc_offset

    # 测试 process_audio_input（元组格式）
    audio_data = generate_test_audio(1.0, 440.0)
    result_audio, result_sr = process_audio_input((SAMPLE_RATE, audio_data))
    assert result_audio is not None, "应返回有效音频"
    assert result_sr == SAMPLE_RATE, "采样率应匹配"
    assert len(result_audio) == len(audio_data), "长度应匹配"

    # 测试 remove_dc_offset
    dc_signal = audio_data + 0.5  # 添加 DC 偏移
    assert abs(np.mean(dc_signal)) > 0.1, "DC信号应有偏移"
    cleaned = remove_dc_offset(dc_signal)
    assert abs(np.mean(cleaned)) < 0.01, f"去除后DC应接近0，实际: {np.mean(cleaned)}"

    # 测试 normalize_audio
    quiet = (audio_data * 0.01).astype(np.float32)
    normalized = normalize_audio(quiet, target_db=-3.0)
    rms = np.sqrt(np.mean(normalized ** 2))
    assert rms > np.sqrt(np.mean(quiet ** 2)), "归一化后响度应提升"

    # 测试 None 输入
    result = process_audio_input(None)
    assert result == (None, None), "None输入应返回(None, None)"

    print("[PASS] recorder")


# ════════════════════════════════════════════════════════════════════
#  运行所有测试
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("声纹魔方 DSP 模块测试 v3")
    print("=" * 50)

    # 原有测试
    test_pitch_shift()
    test_formant_shift()
    test_effects()
    test_preprocessor()
    test_pitch_extract()
    test_postprocessor()
    test_presets()
    test_visualizer()

    # v3 新增测试
    test_soft_knee_compressor()
    test_harmonics_exciter()
    test_normalize_loudness()
    test_noise_gate()
    test_comb_filter_metallic()
    test_extract_formants()
    test_f0_to_midi()
    test_rvc_engine()
    test_recorder()

    print("=" * 50)
    print(f"所有 {17} 个测试通过!")
    print("=" * 50)
