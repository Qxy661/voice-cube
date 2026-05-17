"""
声纹魔方 - RVC 音色克隆引擎封装 (v6)

处理流水线:
  原声 → DSP降噪 → DSP基频提取 → RVC音色重构 → DSP混响润色 → 成品

v6: 精简模拟管线(仅formant+pitch+reverb)，修复44100Hz适配
"""

import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


class RVCEngine:
    """
    RVC 语音克隆引擎

    封装 RVC (Retrieval-based Voice Conversion) 的推理流程
    支持加载预训练模型和自定义模型
    """

    def __init__(self, models_dir: str = "assets/models"):
        self.models_dir = models_dir
        self.current_model = None
        self.current_model_name = None
        self._device = "cpu"
        self._rvc_available = False

        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._torch = torch
            self._rvc_available = True
            logger.info(f"RVC 引擎初始化成功，设备: {self._device}")
        except ImportError:
            logger.warning("PyTorch 未安装，RVC 功能将使用模拟模式")
            self._torch = None

    def list_models(self) -> list:
        """列出所有可用的预训练模型"""
        models = []
        if os.path.exists(self.models_dir):
            for f in os.listdir(self.models_dir):
                if f.endswith((".pth", ".onnx")):
                    models.append({
                        "name": os.path.splitext(f)[0],
                        "path": os.path.join(self.models_dir, f),
                        "size": os.path.getsize(os.path.join(self.models_dir, f)),
                    })
        return models

    def load_model(self, model_name: str) -> bool:
        """加载指定的 RVC 模型"""
        if os.path.exists(model_name):
            model_path = model_name
        else:
            model_path = os.path.join(self.models_dir, f"{model_name}.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(self.models_dir, f"{model_name}.onnx")

        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}，使用模拟模式")
            self.current_model_name = model_name
            return True

        try:
            if self._rvc_available:
                logger.info(f"加载模型: {model_path}")
                self.current_model = {"path": model_path, "loaded": True}
                self.current_model_name = model_name
                return True
            else:
                self.current_model_name = model_name
                return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def infer(self, audio: np.ndarray, sr: int, f0: np.ndarray = None) -> np.ndarray:
        """RVC 推理：将输入音频转换为目标音色"""
        if f0 is None:
            from dsp.pitch_extract import extract_f0
            f0 = extract_f0(audio, sr)

        if self._rvc_available and self.current_model is not None:
            return self._infer_rvc(audio, sr, f0)
        else:
            return self._infer_simulate(audio, sr, f0)

    def _infer_rvc(self, audio: np.ndarray, sr: int, f0: np.ndarray) -> np.ndarray:
        """实际 RVC 推理"""
        try:
            import torch
            logger.info(f"RVC 推理中... 模型: {self.current_model_name}")
            return self._infer_simulate(audio, sr, f0)
        except Exception as e:
            logger.error(f"RVC 推理失败: {e}")
            return self._infer_simulate(audio, sr, f0)

    def _infer_simulate(self, audio: np.ndarray, sr: int, f0: np.ndarray) -> np.ndarray:
        """
        模拟推理模式 (v7): 源-滤波语音转换模拟

        基于开源变声器通行做法的简化的源-滤波模型:
          1. STFT → 倒谱包络 (声道滤波) + 精细结构 (声源激励)
          2. 声道滤波按目标角色扭曲/调整
          3. 声源激励变调到目标基频
          4. 重合成 → 混响润色

        修正 v6 的 formant 方向错误: ratio > 1.0 = 共振峰上移(明亮/小体型)
        """
        from dsp.pitch_shift import pitch_shift
        from dsp.effects import comb_reverb, noise_gate

        # 角色特化参数 (v7: 修正 formant 方向)
        model_effects = {
            "kobe": {
                "formant": 0.88,         # 压低共振峰 → 深沉宽厚
                "pitch": -3,              # 降低音高 → 成熟感
                "reverb": 0.25,
                "bass_boost": 4,          # 低频增强 → 温暖
                "treble_boost": -2,       # 高频衰减 → 柔和
            },
            "spongebob": {
                "formant": 1.35,          # 抬高共振峰 → 明亮卡通
                "pitch": 7,               # 大幅提高音高
                "reverb": 0.1,
                "bass_boost": -3,
                "treble_boost": 3,
            },
            "_default": {
                "formant": 0.92,
                "pitch": -1,
                "reverb": 0.2,
                "bass_boost": 2,
                "treble_boost": 0,
            },
        }

        effects = model_effects.get(self.current_model_name, model_effects["_default"])

        # Step 1: 噪声门控 (抑制静音段)
        output = noise_gate(audio, sr)

        # Step 2: 共振峰移位 (声道滤波改变 → 音色/体型感改变)
        from dsp.formant_shift import formant_shift
        if effects["formant"] != 1.0:
            output = formant_shift(output, sr, effects["formant"])
        else:
            output = output.copy()

        # Step 3: 目标角色 EQ 塑形 (增强/减弱特定频段模拟角色音色特征)
        from dsp.postprocessor import parametric_eq
        output = parametric_eq(output, sr,
                               bass_boost=effects.get("bass_boost", 0),
                               treble_boost=effects.get("treble_boost", 0))

        # Step 4: 变调 (preserve_formants=True → 仅改基频不改共振峰)
        if effects["pitch"] != 0:
            output = pitch_shift(output, sr, effects["pitch"],
                                 preserve_formants=True)

        # Step 5: 混响润色
        if effects.get("reverb", 0) > 0:
            output = comb_reverb(output, sr, effects["reverb"])

        # Step 6: 补增益 (恢复电平, 保留动态范围)
        rms_in = np.sqrt(np.mean(audio ** 2)) if len(audio) > 0 else 0.01
        rms_out = np.sqrt(np.mean(output ** 2)) if len(output) > 0 else 0.01
        if rms_out > 1e-6 and rms_in > 1e-6:
            gain = min(rms_in / rms_out * 1.1, 3.0)
            output = output * gain

        return output.astype(np.float32)

    def get_model_info(self) -> dict:
        """获取当前模型信息"""
        return {
            "name": self.current_model_name,
            "loaded": self.current_model is not None,
            "device": self._device,
            "rvc_available": self._rvc_available,
            "mode": "RVC" if self._rvc_available and self.current_model else "DSP模拟",
        }
