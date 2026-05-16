"""
声纹魔方 Voice Cube v3 — 基于 DSP 与混合 AI 的语音模仿秀系统
主界面入口：Gradio Blocks 构建双 Tab 交互工作台

v3 改进: Bug修复、CJK字体、RVC模拟增强、测试覆盖、工程质量提升
"""

import os
import sys
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import librosa

from config import SAMPLE_RATE, APP_TITLE, APP_DESCRIPTION
from dsp.presets import DSP_PRESETS, AI_PRESETS, get_preset, get_default_params
from dsp.pitch_shift import pitch_shift
from dsp.formant_shift import formant_shift
from dsp.effects import (
    ring_modulate, telephone_filter, comb_reverb, add_breathiness,
    comb_filter_metallic, soft_knee_compressor, harmonics_exciter,
)
from dsp.preprocessor import spectral_subtraction
from dsp.pitch_extract import extract_f0
from dsp.postprocessor import parametric_eq, apply_reverb, normalize_loudness
from dsp.visualizer import (
    plot_waveform_comparison, plot_spectrum_comparison,
    plot_spectrogram_comparison, plot_f0_contour, plot_pipeline_flow,
    plot_formant_trajectory, plot_delta_spectrogram,
    plot_quality_metrics,
)
from ui.recorder import process_audio_input, save_audio, normalize_audio
from ai.rvc_engine import RVCEngine


# ==================== 全局实例 ====================
rvc_engine = RVCEngine()


# ==================== DSP 处理管线 ====================
def apply_dsp_preset(audio: np.ndarray, sr: int, preset_name: str,
                     pitch_shift_val=None, formant_ratio_val=None,
                     ring_mod_val=None, reverb_val=None,
                     breathiness_val=None) -> tuple:
    """应用 DSP 预设处理音频 (v3: 含压缩器/激励器/LUFS归一化)"""
    if audio is None:
        return None, None

    audio = np.clip(audio, -1.0, 1.0)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    preset = get_preset(preset_name)
    params = preset["params"].copy()

    if pitch_shift_val is not None:
        params["pitch_shift"] = pitch_shift_val
    if formant_ratio_val is not None:
        params["formant_ratio"] = formant_ratio_val
    if ring_mod_val is not None:
        params["ring_mod"] = ring_mod_val
    if reverb_val is not None:
        params["reverb"] = reverb_val
    if breathiness_val is not None:
        params["breathiness"] = breathiness_val

    # Step 1: 变调 (带共振峰保护)
    if params.get("pitch_shift", 0) != 0:
        audio = pitch_shift(audio, sr, params["pitch_shift"], preserve_formants=True)

    # Step 2: 共振峰移位
    if params.get("formant_ratio", 1.0) != 1.0:
        audio = formant_shift(audio, sr, params["formant_ratio"])

    # Step 3: 环形调制
    if params.get("ring_mod", 0) > 0:
        ring_freq = params.get("ring_freq", 50)
        audio = ring_modulate(audio, sr, params["ring_mod"], ring_freq)

    # Step 4: 电话滤波
    if params.get("telephone", False):
        audio = telephone_filter(audio, sr)

    # Step 5: 梳状滤波
    if params.get("comb_filter", False):
        audio = comb_filter_metallic(audio, sr)

    # Step 6: 气声
    if params.get("breathiness", 0) > 0:
        audio = add_breathiness(audio, sr, params["breathiness"])

    # Step 7: 谐波激励器 (增加清晰度，防止发闷)
    exciter_amount = params.get("exciter", 0)
    if exciter_amount > 0:
        audio = harmonics_exciter(audio, sr, amount=exciter_amount)

    # Step 8: 动态压缩 (让安静部分更响，整体更均匀)
    if params.get("compressor", False):
        audio = soft_knee_compressor(audio, sr)

    # Step 9: 混响
    if params.get("reverb", 0) > 0:
        audio = apply_reverb(audio, sr, params["reverb"])

    # Step 10: EQ
    bass_boost = params.get("eq_bass_boost", 0)
    treble_boost = params.get("eq_treble_boost", 0)
    if bass_boost != 0 or treble_boost != 0:
        audio = parametric_eq(audio, sr, bass_boost=bass_boost, treble_boost=treble_boost)

    # Step 11: LUFS 响度归一化
    audio = normalize_loudness(audio, sr)

    return audio, sr


def process_basic_imitation(audio_input, preset_name,
                            pitch_shift_slider, formant_slider,
                            ring_mod_slider, reverb_slider,
                            breathiness_slider):
    """基础模仿区处理函数 (v2: 含新可视化)"""
    import matplotlib.pyplot as plt

    audio, sr = process_audio_input(audio_input)
    if audio is None or len(audio) == 0:
        gr.Warning("请先录音或上传音频文件！")
        return None, None, None, None, None, None, None, None

    try:
        original = audio.copy()
        if sr != SAMPLE_RATE:
            original = librosa.resample(original, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        t_start = time.time()
        processed, _ = apply_dsp_preset(
            audio, sr, preset_name,
            pitch_shift_slider, formant_slider,
            ring_mod_slider, reverb_slider,
            breathiness_slider,
        )
        elapsed = time.time() - t_start

        if processed is None or len(processed) == 0:
            gr.Warning("音频处理失败，请检查输入！")
            return None, None, None, None, None, None, None, None

        min_len = min(len(original), len(processed))
        original = original[:min_len]
        processed = processed[:min_len]

        processed_path = save_audio(processed, sr)

        # 生成所有可视化
        fig_wave = plot_waveform_comparison(original, processed, sr)
        fig_spec = plot_spectrum_comparison(original, processed, sr)
        fig_mel = plot_spectrogram_comparison(original, processed, sr)
        fig_delta = plot_delta_spectrogram(original, processed, sr)
        fig_metrics = plot_quality_metrics(original, processed, sr)

        return (
            (sr, processed),
            processed_path,
            fig_wave,
            fig_spec,
            fig_mel,
            fig_delta,
            fig_metrics,
            f"处理耗时: {elapsed:.2f}s",
        )
    except Exception as e:
        gr.Error(f"处理出错: {str(e)}")
        return None, None, None, None, None, None, None, None
    finally:
        plt.close("all")


# ==================== AI 克隆处理管线 ====================
def process_ai_clone(audio_input, clone_target, custom_model_file):
    """AI 克隆区处理函数 (v2: 含新可视化)"""
    import matplotlib.pyplot as plt

    audio, sr = process_audio_input(audio_input)
    if audio is None or len(audio) == 0:
        gr.Warning("请先录音或上传音频文件！")
        return None, None, None, None, None, None, None, None, None

    try:
        audio = np.clip(audio, -1.0, 1.0)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        original = audio.copy()

        steps = [
            {"name": "原声输入", "icon": "🎤", "status": "done"},
            {"name": "DSP降噪", "icon": "🔇", "status": "pending"},
            {"name": "F0提取", "icon": "📊", "status": "pending"},
            {"name": "AI克隆", "icon": "🤖", "status": "pending"},
            {"name": "混响润色", "icon": "🎵", "status": "pending"},
            {"name": "成品", "icon": "✨", "status": "pending"},
        ]

        # Step 2: DSP 降噪
        t0 = time.time()
        steps[1]["status"] = "active"
        try:
            denoised = spectral_subtraction(audio, sr)
        except Exception as e:
            logger.warning(f"谱减法降噪失败，使用原音频: {e}")
            denoised = audio
        steps[1]["status"] = "done"
        steps[1]["time"] = f"{time.time()-t0:.2f}s"

        # Step 3: F0 提取
        t0 = time.time()
        steps[2]["status"] = "active"
        try:
            f0 = extract_f0(denoised, sr)
        except Exception as e:
            logger.error(f"F0提取失败: {e}")
            f0 = np.zeros(max(1, len(denoised) // 512))
        steps[2]["status"] = "done"
        steps[2]["time"] = f"{time.time()-t0:.2f}s"

        # Step 4: AI 克隆
        t0 = time.time()
        steps[3]["status"] = "active"
        try:
            if clone_target == "custom" and custom_model_file is not None:
                model_path = custom_model_file.name if hasattr(custom_model_file, "name") else custom_model_file
                rvc_engine.load_model(model_path)
            else:
                preset = AI_PRESETS.get(clone_target, {})
                model_path = preset.get("model_path", "")
                rvc_engine.load_model(model_path)
            processed = rvc_engine.infer(denoised, sr, f0)
        except Exception as e:
            logger.error(f"AI克隆处理失败: {e}")
            gr.Warning("AI克隆处理失败，已返回降噪后音频")
            processed = denoised
        steps[3]["status"] = "done"
        steps[3]["time"] = f"{time.time()-t0:.2f}s"

        # Step 5: 混响润色 + LUFS
        t0 = time.time()
        steps[4]["status"] = "active"
        processed = apply_reverb(processed, sr, wet=0.2)
        processed = normalize_loudness(processed, sr)
        steps[4]["status"] = "done"
        steps[4]["time"] = f"{time.time()-t0:.2f}s"
        steps[5]["status"] = "done"

        min_len = min(len(original), len(processed))
        original = original[:min_len]
        processed = processed[:min_len]

        processed_path = save_audio(processed, sr)

        # 可视化
        fig_wave = plot_waveform_comparison(original, processed, sr, labels=("原声", "AI克隆后"))
        fig_spec = plot_spectrum_comparison(original, processed, sr, labels=("原声频谱", "AI克隆后频谱"))
        fig_mel = plot_spectrogram_comparison(original, processed, sr, labels=("原声语谱图", "AI克隆后语谱图"))
        fig_f0 = plot_f0_contour(f0, sr)
        fig_pipeline = plot_pipeline_flow(steps)
        fig_formant = plot_formant_trajectory(original, processed, sr)
        fig_delta = plot_delta_spectrogram(original, processed, sr)

        return (
            (sr, processed),
            processed_path,
            fig_wave,
            fig_spec,
            fig_mel,
            fig_f0,
            fig_pipeline,
            fig_formant,
            fig_delta,
        )
    except Exception as e:
        gr.Error(f"AI 克隆处理出错: {str(e)}")
        return None, None, None, None, None, None, None, None, None
    finally:
        plt.close("all")


# ==================== 预设选择回调 ====================
def on_preset_select(preset_name):
    """当用户选择预设时，更新滑块值"""
    preset = get_preset(preset_name)
    p = preset["params"]
    return (
        p.get("pitch_shift", 0),
        p.get("formant_ratio", 1.0),
        p.get("ring_mod", 0.0),
        p.get("reverb", 0.0),
        p.get("breathiness", 0.0),
    )


# ==================== Gradio 界面构建 ====================
def build_ui():
    """构建完整的 Gradio 界面 (v2)"""

    custom_css = """
    .preset-card {
        border: 2px solid #333355;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        background: #1A1A2E;
    }
    .preset-card:hover {
        border-color: #6C63FF;
        background: #252545;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(108,99,255,0.3);
    }
    .preset-card.selected {
        border-color: #6C63FF;
        background: #2A2A5A;
        box-shadow: 0 0 20px rgba(108,99,255,0.4);
    }
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #6C63FF, #FF6584);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.2em;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1em;
    }
    """

    with gr.Blocks(
        title=APP_TITLE,
        css=custom_css,
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="pink",
            neutral_hue="slate",
        ),
    ) as app:

        gr.HTML('<div class="main-title">声纹魔方 Voice Cube</div>')
        gr.HTML('<div class="subtitle">基于 DSP 与混合 AI 的语音模仿秀系统 · v2 科研级可视化</div>')
        gr.Markdown("---")

        with gr.Tabs():
            # ==================== Tab 1: 基础模仿区 ====================
            with gr.Tab("🎭 基础模仿区（DSP）"):
                gr.Markdown("### 选择你要模仿的音色，系统通过纯 DSP 算法实现声音变换")

                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input_basic = gr.Audio(
                            sources=["microphone", "upload"],
                            type="numpy",
                            label="🎤 输入音频（录音或上传）",
                        )

                        gr.Markdown("#### 🎭 选择模仿目标")
                        preset_selector = gr.Radio(
                            choices=[
                                (f"{v['icon']} {v['name']}", k)
                                for k, v in DSP_PRESETS.items()
                            ],
                            value="deep_male",
                            label="模仿预设",
                        )

                        preset_desc = gr.Markdown(
                            value=f"**{DSP_PRESETS['deep_male']['description']}**",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### ⚙️ 高级微调（可选）")

                        pitch_slider = gr.Slider(
                            minimum=-12, maximum=12, step=1, value=-3,
                            label="音高偏移（半音）",
                            info="-12 = 低八度, 0 = 不变, +12 = 高八度",
                        )
                        formant_slider = gr.Slider(
                            minimum=0.4, maximum=2.5, step=0.05, value=1.10,
                            label="共振峰缩放",
                            info="0.4 = 小黄人, 1.0 = 不变, 2.5 = 巨人",
                        )
                        ring_mod_slider = gr.Slider(
                            minimum=0, maximum=1, step=0.05, value=0,
                            label="环形调制深度（机械音）",
                        )
                        reverb_slider = gr.Slider(
                            minimum=0, maximum=1, step=0.05, value=0.08,
                            label="混响强度",
                        )
                        breathiness_slider = gr.Slider(
                            minimum=0, maximum=1, step=0.05, value=0,
                            label="气声混合（耳语效果）",
                        )

                        process_btn = gr.Button("🚀 开始模仿", variant="primary", size="lg")
                        time_display = gr.Markdown("")

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column():
                        audio_output_basic = gr.Audio(label="▶ 模仿结果", type="numpy")
                        download_basic = gr.File(label="⬇ 下载音频")

                with gr.Row():
                    with gr.Column():
                        plot_waveform = gr.Plot(label="📊 波形对比")
                    with gr.Column():
                        plot_spectrum = gr.Plot(label="📊 频谱对比")

                with gr.Row():
                    with gr.Column():
                        plot_mel = gr.Plot(label="📊 语谱图对比")

                with gr.Row():
                    with gr.Column():
                        plot_delta = gr.Plot(label="📊 频谱差异")
                    with gr.Column():
                        plot_metrics = gr.Plot(label="📊 音质指标")

                # 事件绑定
                preset_selector.change(
                    fn=on_preset_select,
                    inputs=[preset_selector],
                    outputs=[pitch_slider, formant_slider, ring_mod_slider, reverb_slider, breathiness_slider],
                )

                def update_preset_desc(name):
                    p = get_preset(name)
                    return f"**{p['description']}**"

                preset_selector.change(
                    fn=update_preset_desc,
                    inputs=[preset_selector],
                    outputs=[preset_desc],
                )

                process_btn.click(
                    fn=process_basic_imitation,
                    inputs=[
                        audio_input_basic, preset_selector,
                        pitch_slider, formant_slider,
                        ring_mod_slider, reverb_slider,
                        breathiness_slider,
                    ],
                    outputs=[
                        audio_output_basic, download_basic,
                        plot_waveform, plot_spectrum, plot_mel,
                        plot_delta, plot_metrics, time_display,
                    ],
                )

            # ==================== Tab 2: AI 克隆区 ====================
            with gr.Tab("🚀 AI 克隆区（DSP+AI）"):
                gr.Markdown("### 选择目标人物，系统通过 DSP 预处理 + AI 模型实现音色克隆")

                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input_clone = gr.Audio(
                            sources=["microphone", "upload"],
                            type="numpy",
                            label="🎤 输入音频（录音或上传）",
                        )

                        gr.Markdown("#### 🌟 选择模仿人物")
                        clone_target = gr.Radio(
                            choices=[
                                (f"{v['icon']} {v['name']}", k)
                                for k, v in AI_PRESETS.items()
                            ],
                            value="kobe",
                            label="克隆目标",
                        )

                        custom_model = gr.File(
                            label="📁 上传自定义 RVC 模型（可选）",
                            file_types=[".pth", ".onnx"],
                            visible=False,
                        )

                        clone_desc = gr.Markdown(
                            value=f"**{AI_PRESETS['kobe']['description']}**",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### 📋 处理流水线")
                        gr.Markdown(
                            "```\n"
                            "原声 → [DSP降噪] → [F0提取] → [AI克隆] → [混响润色] → 成品\n"
                            "```"
                        )

                        clone_btn = gr.Button("🚀 开始克隆", variant="primary", size="lg")

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column():
                        audio_output_clone = gr.Audio(label="▶ 克隆结果", type="numpy")
                        download_clone = gr.File(label="⬇ 下载音频")

                with gr.Row():
                    plot_pipeline = gr.Plot(label="📋 处理流水线状态")

                with gr.Row():
                    with gr.Column():
                        plot_waveform_clone = gr.Plot(label="📊 波形对比")
                    with gr.Column():
                        plot_spectrum_clone = gr.Plot(label="📊 频谱对比")

                with gr.Row():
                    with gr.Column():
                        plot_mel_clone = gr.Plot(label="📊 语谱图对比")
                    with gr.Column():
                        plot_f0 = gr.Plot(label="📊 基频轮廓线")

                with gr.Row():
                    with gr.Column():
                        plot_formant_clone = gr.Plot(label="📊 共振峰轨迹")
                    with gr.Column():
                        plot_delta_clone = gr.Plot(label="📊 频谱差异")

                # 事件绑定
                def toggle_custom_model(target):
                    return gr.File(visible=(target == "custom"))

                clone_target.change(
                    fn=toggle_custom_model,
                    inputs=[clone_target],
                    outputs=[custom_model],
                )

                def update_clone_desc(target):
                    p = AI_PRESETS.get(target, {})
                    return f"**{p.get('description', '')}**"

                clone_target.change(
                    fn=update_clone_desc,
                    inputs=[clone_target],
                    outputs=[clone_desc],
                )

                clone_btn.click(
                    fn=process_ai_clone,
                    inputs=[audio_input_clone, clone_target, custom_model],
                    outputs=[
                        audio_output_clone, download_clone,
                        plot_waveform_clone, plot_spectrum_clone,
                        plot_mel_clone, plot_f0, plot_pipeline,
                        plot_formant_clone, plot_delta_clone,
                    ],
                )

        gr.Markdown("---")
        gr.Markdown(
            "<div style='text-align:center; color:#666;'>"
            "声纹魔方 Voice Cube v2 | 基于 DSP 与混合 AI 的语音模仿秀系统 | "
            "科研级可视化 + 高质量音频处理"
            "</div>"
        )

    return app


# ==================== 启动入口 ====================
def main():
    """启动声纹魔方应用"""
    print("=" * 50)
    print("  声纹魔方 Voice Cube v2")
    print("  基于 DSP 与混合 AI 的语音模仿秀系统")
    print("=" * 50)
    print(f"  采样率: {SAMPLE_RATE} Hz")
    print(f"  DSP 预设: {len(DSP_PRESETS)} 种")
    print(f"  AI 模型: {len(AI_PRESETS)} 种")
    print(f"  改进: polyphase重采样/LPC24阶/递归反馈混响/压缩器/激励器/LUFS")
    print("=" * 50)

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
