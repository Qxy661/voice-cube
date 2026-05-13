"""
声纹魔方 Voice Cube — 基于 DSP 与混合 AI 的语音模仿秀系统
主界面入口：Gradio Blocks 构建双 Tab 交互工作台

Tab 1: 基础模仿区 — DSP 模仿经典音色（低音炮/小黄人/机器人等）
Tab 2: AI 克隆区 — RVC 模仿特定人物（科比/海绵宝宝等）
"""

import os
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import librosa

from config import SAMPLE_RATE, APP_TITLE, APP_DESCRIPTION
from dsp.presets import DSP_PRESETS, AI_PRESETS, get_preset, get_default_params
from dsp.pitch_shift import pitch_shift
from dsp.formant_shift import formant_shift
from dsp.effects import ring_modulate, telephone_filter, comb_reverb, add_breathiness, comb_filter_metallic
from dsp.preprocessor import spectral_subtraction
from dsp.pitch_extract import extract_f0
from dsp.postprocessor import parametric_eq, apply_reverb
from dsp.visualizer import (
    plot_waveform_comparison, plot_spectrum_comparison,
    plot_spectrogram_comparison, plot_f0_contour, plot_pipeline_flow,
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
    """
    应用 DSP 预设处理音频

    Returns
    -------
    tuple
        (处理后音频, 原音频, 采样率)
    """
    if audio is None:
        return None, None, None

    # 标准化
    audio = normalize_audio(audio)
    # 重采样到标准采样率
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # 获取预设参数
    preset = get_preset(preset_name)
    params = preset["params"].copy()

    # 允许滑块覆盖预设值
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

    # Step 1: 变调
    if params.get("pitch_shift", 0) != 0:
        audio = pitch_shift(audio, sr, params["pitch_shift"])

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

    # Step 5: 梳状滤波（金属质感）
    if params.get("comb_filter", False):
        audio = comb_filter_metallic(audio, sr)

    # Step 6: 气声
    if params.get("breathiness", 0) > 0:
        audio = add_breathiness(audio, sr, params["breathiness"])

    # Step 7: 混响
    if params.get("reverb", 0) > 0:
        audio = apply_reverb(audio, sr, params["reverb"])

    # Step 8: EQ
    bass_boost = params.get("eq_bass_boost", 0)
    treble_boost = params.get("eq_treble_boost", 0)
    if bass_boost != 0 or treble_boost != 0:
        audio = parametric_eq(audio, sr, bass_boost=bass_boost, treble_boost=treble_boost)

    return audio, sr


def process_basic_imitation(audio_input, preset_name,
                            pitch_shift_slider, formant_slider,
                            ring_mod_slider, reverb_slider,
                            breathiness_slider):
    """基础模仿区处理函数"""
    import matplotlib.pyplot as plt

    audio, sr = process_audio_input(audio_input)
    if audio is None or len(audio) == 0:
        gr.Warning("请先录音或上传音频文件！")
        return None, None, None, None, None

    try:
        # 保存原声
        original = audio.copy()

        # 重采样
        if sr != SAMPLE_RATE:
            original = librosa.resample(original, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # 应用 DSP 处理
        processed, _ = apply_dsp_preset(
            audio, sr, preset_name,
            pitch_shift_slider, formant_slider,
            ring_mod_slider, reverb_slider,
            breathiness_slider,
        )

        if processed is None or len(processed) == 0:
            gr.Warning("音频处理失败，请检查输入！")
            return None, None, None, None, None

        # 确保长度一致
        min_len = min(len(original), len(processed))
        original = original[:min_len]
        processed = processed[:min_len]

        # 保存音频文件
        processed_path = save_audio(processed, sr)

        # 生成可视化
        fig_wave = plot_waveform_comparison(original, processed, sr)
        fig_spec = plot_spectrum_comparison(original, processed, sr)
        fig_mel = plot_spectrogram_comparison(original, processed, sr)

        return (
            (sr, processed),        # Gradio 音频输出
            processed_path,         # 下载文件
            fig_wave,               # 波形图
            fig_spec,               # 频谱图
            fig_mel,                # 语谱图
        )
    except Exception as e:
        gr.Error(f"处理出错: {str(e)}")
        return None, None, None, None, None
    finally:
        plt.close("all")


# ==================== AI 克隆处理管线 ====================
def process_ai_clone(audio_input, clone_target, custom_model_file):
    """AI 克隆区处理函数：原声 → DSP降噪 → F0提取 → AI克隆 → DSP混响 → 成品"""
    import matplotlib.pyplot as plt

    audio, sr = process_audio_input(audio_input)
    if audio is None or len(audio) == 0:
        gr.Warning("请先录音或上传音频文件！")
        return None, None, None, None, None, None, None

    try:
        # 标准化和重采样
        audio = normalize_audio(audio)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        original = audio.copy()

        # 流水线步骤状态
        steps = [
            {"name": "原声输入", "icon": "🎤", "status": "done"},
            {"name": "DSP降噪", "icon": "🔇", "status": "pending"},
            {"name": "F0提取", "icon": "📊", "status": "pending"},
            {"name": "AI克隆", "icon": "🤖", "status": "pending"},
            {"name": "混响润色", "icon": "🎵", "status": "pending"},
            {"name": "成品", "icon": "✨", "status": "pending"},
        ]

        # Step 2: DSP 降噪
        steps[1]["status"] = "active"
        try:
            denoised = spectral_subtraction(audio, sr)
        except Exception as e:
            logger.warning(f"谱减法降噪失败，使用原音频: {e}")
            denoised = audio
        steps[1]["status"] = "done"

        # Step 3: F0 提取
        steps[2]["status"] = "active"
        try:
            f0 = extract_f0(denoised, sr)
        except Exception as e:
            logger.error(f"F0提取失败: {e}")
            gr.Warning("基频提取失败，F0 轮廓图可能不完整")
            f0 = np.zeros(max(1, len(denoised) // 512))
        steps[2]["status"] = "done"

        # Step 4: AI 克隆
        steps[3]["status"] = "active"
        try:
            # 选择模型
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

        # Step 5: 混响润色
        steps[4]["status"] = "active"
        processed = apply_reverb(processed, sr, wet=0.2)
        steps[4]["status"] = "done"
        steps[5]["status"] = "done"

        # 确保长度一致
        min_len = min(len(original), len(processed))
        original = original[:min_len]
        processed = processed[:min_len]

        # 保存
        processed_path = save_audio(processed, sr)

        # 可视化
        fig_wave = plot_waveform_comparison(original, processed, sr, labels=("原声", "AI克隆后"))
        fig_spec = plot_spectrum_comparison(original, processed, sr, labels=("原声频谱", "AI克隆后频谱"))
        fig_mel = plot_spectrogram_comparison(original, processed, sr, labels=("原声语谱图", "AI克隆后语谱图"))
        fig_f0 = plot_f0_contour(f0, sr)
        fig_pipeline = plot_pipeline_flow(steps)

        return (
            (sr, processed),        # Gradio 音频输出
            processed_path,         # 下载文件
            fig_wave,               # 波形图
            fig_spec,               # 频谱图
            fig_mel,                # 语谱图
            fig_f0,                 # F0 轮廓线
            fig_pipeline,           # 流水线状态
        )
    except Exception as e:
        gr.Error(f"AI 克隆处理出错: {str(e)}")
        return None, None, None, None, None, None, None
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
    """构建完整的 Gradio 界面"""

    # 自定义 CSS 样式
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

        # ==================== 标题 ====================
        gr.HTML('<div class="main-title">声纹魔方 Voice Cube</div>')
        gr.HTML('<div class="subtitle">基于 DSP 与混合 AI 的语音模仿秀系统</div>')
        gr.Markdown("---")

        with gr.Tabs():
            # ==================== Tab 1: 基础模仿区 ====================
            with gr.Tab("🎭 基础模仿区（DSP）"):
                gr.Markdown("### 选择你要模仿的音色，系统通过纯 DSP 算法实现声音变换")

                with gr.Row():
                    # 左列：输入 + 预设选择
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

                    # 右列：高级微调 + 输出
                    with gr.Column(scale=1):
                        gr.Markdown("#### ⚙️ 高级微调（可选）")

                        pitch_slider = gr.Slider(
                            minimum=-12, maximum=12, step=1, value=-4,
                            label="音高偏移（半音）",
                            info="-12 = 低八度, 0 = 不变, +12 = 高八度",
                        )
                        formant_slider = gr.Slider(
                            minimum=0.4, maximum=2.5, step=0.05, value=1.15,
                            label="共振峰缩放",
                            info="0.4 = 小黄人, 1.0 = 不变, 2.5 = 巨人",
                        )
                        ring_mod_slider = gr.Slider(
                            minimum=0, maximum=1, step=0.05, value=0,
                            label="环形调制深度（机械音）",
                        )
                        reverb_slider = gr.Slider(
                            minimum=0, maximum=1, step=0.05, value=0.1,
                            label="混响强度",
                        )
                        breathiness_slider = gr.Slider(
                            minimum=0, maximum=1, step=0.05, value=0,
                            label="气声混合（耳语效果）",
                        )

                        process_btn = gr.Button("🚀 开始模仿", variant="primary", size="lg")

                gr.Markdown("---")

                # 输出区域
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
                    plot_mel = gr.Plot(label="📊 语谱图对比")

                # ==================== 事件绑定 ====================
                # 预设选择更新滑块
                preset_selector.change(
                    fn=on_preset_select,
                    inputs=[preset_selector],
                    outputs=[pitch_slider, formant_slider, ring_mod_slider, reverb_slider, breathiness_slider],
                )

                # 预设选择更新描述
                def update_preset_desc(name):
                    p = get_preset(name)
                    return f"**{p['description']}**"

                preset_selector.change(
                    fn=update_preset_desc,
                    inputs=[preset_selector],
                    outputs=[preset_desc],
                )

                # 处理按钮
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
                    ],
                )

            # ==================== Tab 2: AI 克隆区 ====================
            with gr.Tab("🚀 AI 克隆区（DSP+AI）"):
                gr.Markdown("### 选择目标人物，系统通过 DSP 预处理 + AI 模型实现音色克隆")

                with gr.Row():
                    # 左列：输入 + 人物选择
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

                    # 右列：处理流水线 + 输出
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📋 处理流水线")
                        gr.Markdown(
                            "```\n"
                            "原声 → [DSP降噪] → [F0提取] → [AI克隆] → [混响润色] → 成品\n"
                            "```"
                        )

                        clone_btn = gr.Button("🚀 开始克隆", variant="primary", size="lg")

                gr.Markdown("---")

                # 输出区域
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

                # ==================== 事件绑定 ====================
                # 显示/隐藏自定义模型上传
                def toggle_custom_model(target):
                    return gr.update(visible=(target == "custom"))

                clone_target.change(
                    fn=toggle_custom_model,
                    inputs=[clone_target],
                    outputs=[custom_model],
                )

                # 克隆目标描述更新
                def update_clone_desc(target):
                    p = AI_PRESETS.get(target, {})
                    return f"**{p.get('description', '')}**"

                clone_target.change(
                    fn=update_clone_desc,
                    inputs=[clone_target],
                    outputs=[clone_desc],
                )

                # 克隆按钮
                clone_btn.click(
                    fn=process_ai_clone,
                    inputs=[audio_input_clone, clone_target, custom_model],
                    outputs=[
                        audio_output_clone, download_clone,
                        plot_waveform_clone, plot_spectrum_clone,
                        plot_mel_clone, plot_f0, plot_pipeline,
                    ],
                )

        # ==================== 页脚 ====================
        gr.Markdown("---")
        gr.Markdown(
            "<div style='text-align:center; color:#666;'>"
            "声纹魔方 Voice Cube | 基于 DSP 与混合 AI 的语音模仿秀系统 | "
            "纯 DSP 引擎 + RVC 音色克隆"
            "</div>"
        )

    return app


# ==================== 启动入口 ====================
def main():
    """启动声纹魔方应用"""
    print("=" * 50)
    print("  声纹魔方 Voice Cube")
    print("  基于 DSP 与混合 AI 的语音模仿秀系统")
    print("=" * 50)
    print(f"  采样率: {SAMPLE_RATE} Hz")
    print(f"  DSP 预设: {len(DSP_PRESETS)} 种")
    print(f"  AI 模型: {len(AI_PRESETS)} 种")
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
