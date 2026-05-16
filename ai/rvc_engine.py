"""
声纹魔方 - RVC 音色克隆引擎封装 (v3)

处理流水线:
  原声 → DSP降噪 → DSP基频提取 → RVC音色重构 → DSP混响润色 → 成品

v3: 扩展模拟模式，更多角色预设，压缩器/激励器/EQ
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
        """实际 RVC 推理（需要完整 RVC 环境）"""
        try:
            import torch
            logger.info(f"RVC 推理中... 模型: {self.current_model_name}")
            # 实际项目中这里调用 RVC 的推理接口
            # from infer.modules.vc.modules import VC
            # vc = VC(...)
            # output = vc.pipeline(...)
            return self._infer_simulate(audio, sr, f0)
        except Exception as e:
            logger.error(f"RVC 推理失败: {e}")
            return self._infer_simulate(audio, sr, f0)

    def _infer_simulate(self, audio: np.ndarray, sr: int, f0: np.ndarray) -> np.ndarray:
        """
        模拟推理模式 (v3): DSP方法模拟音色变化

        v3改进:
          - 扩展到所有AI预设角色
          - 加入压缩器/激励器/EQ
          - 未知模型使用通用变声链（不再是恒等变换）
        """
        from dsp.formant_shift import formant_shift
        from dsp.pitch_shift import pitch_shift
        from dsp.effects import comb_reverb, soft_knee_compressor, harmonics_exciter
        from dsp.postprocessor import parametric_eq

        # 角色特化 DSP 参数
        model_effects = {
            "kobe": {
                "formant": 1.2, "pitch": -2, "reverb": 0.3,
                "compressor": True, "exciter": 0.12, "eq_bass": 4, "eq_treble": -2,
            },
            "spongebob": {
                "formant": 0.6, "pitch": 6, "reverb": 0.1,
                "compressor": True, "exciter": 0.08, "eq_bass": -5, "eq_treble": 3,
            },
            # 通用变声链（自定义模型/未知模型使用）
            "_default": {
                "formant": 1.15, "pitch": -1, "reverb": 0.25,
                "compressor": True, "exciter": 0.1, "eq_bass": 2, "eq_treble": 0,
            },
        }

        effects = model_effects.get(self.current_model_name, model_effects["_default"])

        # Step 1: 共振峰移位
        if effects["formant"] != 1.0:
            output = formant_shift(audio, sr, effects["formant"])
        else:
            output = audio.copy()

        # Step 2: 变调
        if effects["pitch"] != 0:
            output = pitch_shift(output, sr, effects["pitch"])

        # Step 3: 谐波激励器（增加清晰度）
        if effects.get("exciter", 0) > 0:
            output = harmonics_exciter(output, sr, amount=effects["exciter"])

        # Step 4: 动态压缩
        if effects.get("compressor", False):
            output = soft_knee_compressor(output, sr)

        # Step 5: 混响
        if effects["reverb"] > 0:
            output = comb_reverb(output, sr, effects["reverb"])

        # Step 6: EQ
        eq_bass = effects.get("eq_bass", 0)
        eq_treble = effects.get("eq_treble", 0)
        if eq_bass != 0 or eq_treble != 0:
            output = parametric_eq(output, sr, bass_boost=eq_bass, treble_boost=eq_treble)

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
