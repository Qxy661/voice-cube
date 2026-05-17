"""
声纹魔方 - RVC 音色克隆引擎封装 (v7)

双模式:
  1. ONNX 推理: 需要 .onnx 模型文件 (含 contentvec + VITS 解码器)
  2. DSP 模拟: 无模型时的源-滤波回退

处理流水线 (ONNX 模式):
  原声(16kHz) → HPF 48Hz → reflect pad → contentvec特征
  + F0提取 → mel-scale coarse pitch + fine pitch
  + 随机噪声
  → ONNX decoder → trim pad → 重采样回 sr → 成品

v7: 补齐 ONNX 推理管线 (原来 _infer_rvc 是空壳)
"""

import numpy as np
import os
import logging
import librosa
from scipy.signal import butter, sosfilt

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SAMPLE_RATE, MODELS_DIR,
    RVC_SAMPLE_RATE, RVC_HOP_LENGTH, RVC_PAD_SEC, RVC_F0_MIN, RVC_F0_MAX,
)


class RVCEngine:
    """
    RVC 语音克隆引擎

    封装 RVC (Retrieval-based Voice Conversion) 的 ONNX 推理流程。
    支持加载 .onnx 模型进行实时变声, 无模型时回退到 DSP 模拟。
    """

    def __init__(self, models_dir: str = None):
        self.models_dir = models_dir or MODELS_DIR
        self.current_model = None  # 用于 DSP 模拟的角色参数标记
        self.current_model_name = None
        self._device = "cpu"
        self._rvc_available = False
        self._ort = None
        self._ort_available = False
        self._session = None  # ONNX Runtime InferenceSession (RVC decoder)
        self._vec_session = None  # ContentVec model session (feature extractor)
        self._net_g = None  # PyTorch SynthesizerTrn model
        self._tgt_sr = None  # 目标采样率 (from .pth config)

        # 检查 PyTorch
        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._torch = torch
            self._rvc_available = True
            logger.info(f"PyTorch 可用, 设备: {self._device}")
        except ImportError:
            logger.warning("PyTorch 未安装")
            self._torch = None

        # 检查 ONNX Runtime
        try:
            import onnxruntime as ort
            self._ort = ort
            self._ort_available = True
            logger.info(f"ONNX Runtime 可用 ({ort.__version__})")
        except ImportError:
            logger.warning("onnxruntime 未安装, RVC ONNX 推理不可用")

    # ─── 模型管理 ────────────────────────────────────────────

    def list_models(self) -> list:
        """列出 assets/models/ 下所有可用模型文件"""
        models = []
        if os.path.exists(self.models_dir):
            for f in sorted(os.listdir(self.models_dir)):
                if f.endswith((".pth", ".onnx")):
                    models.append({
                        "name": os.path.splitext(f)[0],
                        "path": os.path.join(self.models_dir, f),
                        "size": os.path.getsize(os.path.join(self.models_dir, f)),
                        "type": "ONNX" if f.endswith(".onnx") else "PyTorch",
                    })
        return models

    def load_model(self, model_name: str) -> bool:
        """
        加载 RVC 模型。

        model_name 可以是:
          - 干净名称: "woman_1", "kobe", "spongebob"
          - 完整路径: "assets/models/woman_1.onnx"

        优先级:
          1. 直接路径 (model_name 是完整文件路径且文件存在)
          2. assets/models/{name}.onnx
          3. assets/models/{name}.pth  (仅记录, 不支持 torch.load 推理)

        无可用模型 → 标记角色参数用于 DSP 模拟回退。
        """
        # 提取干净名称 (用于 model_effects 查找 + 文件名构建)
        clean_name = os.path.splitext(os.path.basename(model_name))[0]
        self.current_model_name = clean_name

        # 情况 A: 直接路径且文件存在
        if os.path.isfile(model_name):
            model_path = model_name
        else:
            # 情况 B: 按名称在 assets/models/ 中查找
            model_path = os.path.join(self.models_dir, f"{clean_name}.onnx")
            if not os.path.exists(model_path):
                model_path = os.path.join(self.models_dir, f"{clean_name}.pth")

        if not os.path.exists(model_path):
            logger.warning(
                f"模型文件不存在: {model_path}\n"
                f"  回退到 DSP 模拟模式。\n"
                f"  如需真实 RVC 推理, 请将 .onnx 模型放入 {self.models_dir}/"
            )
            self.current_model = {"mode": "simulate", "role": clean_name}
            return True

        # 尝试加载 ONNX
        if model_path.endswith(".onnx") and self._ort_available:
            try:
                providers = ["CPUExecutionProvider"]
                if self._device == "cuda":
                    try:
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    except Exception:
                        pass
                self._session = self._ort.InferenceSession(
                    model_path, providers=providers,
                )
                logger.info(f"ONNX 模型加载成功: {model_path}")
                logger.info(f"  输入: {[inp.name + str(inp.shape) for inp in self._session.get_inputs()]}")
                logger.info(f"  输出: {[out.name + str(out.shape) for out in self._session.get_outputs()]}")

                # 尝试加载 ContentVec 特征提取模型 (同目录下)
                self._load_vec_model()

                self.current_model = {"mode": "onnx", "path": model_path}
                return True
            except Exception as e:
                logger.error(f"ONNX 模型加载失败: {e}, 回退到 DSP 模拟")

        # 尝试加载 PyTorch .pth
        if model_path.endswith(".pth") and self._torch is not None:
            try:
                self._load_pth_model(model_path)
                self.current_model = {"mode": "pytorch", "path": model_path}
                return True
            except Exception as e:
                logger.error(f"PyTorch 模型加载失败: {e}, 回退到 DSP 模拟")

        # 加载失败 → DSP 模拟
        logger.info(f"使用 DSP 模拟回退 (角色: {clean_name})")
        self.current_model = {"mode": "simulate", "role": clean_name, "path": model_path}
        return True

    # ─── 主推理入口 ──────────────────────────────────────────

    def infer(self, audio: np.ndarray, sr: int, f0: np.ndarray = None) -> np.ndarray:
        """RVC 推理入口: 自动选择 PyTorch / ONNX / DSP 模拟"""
        if f0 is None:
            from dsp.pitch_extract import extract_f0
            f0 = extract_f0(audio, sr)

        if not self.current_model:
            return self._infer_simulate(audio, sr, f0)

        mode = self.current_model.get("mode")
        if mode == "pytorch" and self._net_g is not None:
            return self._infer_rvc_pytorch(audio, sr, f0)
        elif mode == "onnx" and self._session is not None:
            return self._infer_rvc(audio, sr, f0)
        else:
            return self._infer_simulate(audio, sr, f0)

    def _load_vec_model(self, version: str = "v1"):
        """
        加载 ContentVec 特征提取模型。

        v1 模型需要 256-dim 特征 (vec-256-layer-9.onnx)
        v2 模型需要 768-dim 特征 (vec-768-layer-12.onnx)
        """
        if version == "v2":
            candidates = ["vec-768-layer-12.onnx", "contentvec_768.onnx"]
        else:
            candidates = ["vec-256-layer-9.onnx", "contentvec.onnx", "hubert_base.onnx"]

        for name in candidates:
            path = os.path.join(self.models_dir, name)
            if os.path.exists(path):
                try:
                    self._vec_session = self._ort.InferenceSession(
                        path, providers=["CPUExecutionProvider"],
                    )
                    logger.info(f"ContentVec 模型加载成功 ({version}): {path}")
                    inp_name = self._vec_session.get_inputs()[0].name
                    out_shape = self._vec_session.get_outputs()[0].shape
                    logger.info(f"  输入: {inp_name}, 输出: {out_shape}")
                    return
                except Exception as e:
                    logger.warning(f"ContentVec 加载失败 ({name}): {e}")

        logger.info(f"ContentVec {version} 模型未找到, 使用 DSP 模拟代替 RVC ONNX 推理")

    # ─── PyTorch .pth 模型加载 ─────────────────────────────

    def _load_pth_model(self, model_path: str):
        """
        加载 PyTorch RVC .pth 模型 (SynthesizerTrnMs768NSFsid / v256)。

        需要 rvc_python 包的模型架构代码。如果不可用，使用 stub 导入。
        """
        torch = self._torch

        # 动态导入 RVC 模型架构 (兼容 rvc_python 或直接 stub)
        try:
            from rvc_python.lib.infer_pack.models import (
                SynthesizerTrnMs256NSFsid,
                SynthesizerTrnMs768NSFsid,
            )
        except ImportError:
            # rvc_python 未安装或依赖缺失 — 尝试 stub 方式
            SynthesizerTrnMs256NSFsid, SynthesizerTrnMs768NSFsid = \
                self._import_rvc_architecture()

        # 加载 checkpoint
        cpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self._tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        if_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")

        # 选择模型类
        model_classes = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
        }
        ModelClass = model_classes.get((version, if_f0), SynthesizerTrnMs768NSFsid)

        self._net_g = ModelClass(*cpt["config"], is_half=False)
        del self._net_g.enc_q
        self._net_g.load_state_dict(cpt["weight"], strict=False)
        self._net_g.eval().to(self._device).float()

        # 加载 ContentVec 模型 (v1→256-dim, v2→768-dim)
        self._load_vec_model(version=version)

        logger.info(f"PyTorch RVC 模型加载成功: {model_path}")
        logger.info(f"  版本: {version}, F0: {if_f0}, 目标SR: {self._tgt_sr}")
        logger.info(f"  参数量: {sum(p.numel() for p in self._net_g.parameters()):,}")

    def _import_rvc_architecture(self):
        """
        动态导入 RVC 模型架构代码。

        如果 rvc_python 已安装但缺少依赖(faiss等)，通过 stub 绕过。
        """
        import sys, types

        # Stub 缺失的依赖
        stubs = ['faiss', 'parselmouth', 'pyworld', 'torchcrepe',
                 'torchcrepe.decode', 'torchcrepe.load',
                 'fairseq', 'fairseq.checkpoint_utils']
        original = {}
        for mod_name in stubs:
            if mod_name not in sys.modules:
                original[mod_name] = None
                sys.modules[mod_name] = types.ModuleType(mod_name)
            else:
                original[mod_name] = sys.modules[mod_name]

        try:
            from rvc_python.lib.infer_pack.models import (
                SynthesizerTrnMs256NSFsid,
                SynthesizerTrnMs768NSFsid,
            )
            return SynthesizerTrnMs256NSFsid, SynthesizerTrnMs768NSFsid
        finally:
            # 清理 stub (不影响已导入的模块)
            for mod_name, orig in original.items():
                if orig is None and mod_name in sys.modules:
                    del sys.modules[mod_name]

    # ─── PyTorch RVC 推理 ──────────────────────────────────

    def _infer_rvc_pytorch(self, audio: np.ndarray, sr: int, f0: np.ndarray) -> np.ndarray:
        """PyTorch RVC 推理管线"""
        torch = self._torch
        try:
            # Step 1: 重采样到 16kHz
            if sr != RVC_SAMPLE_RATE:
                audio_16k = librosa.resample(
                    audio.astype(np.float32), orig_sr=sr, target_sr=RVC_SAMPLE_RATE,
                )
            else:
                audio_16k = audio.astype(np.float32).copy()

            # Step 2: 高通滤波 48Hz
            sos = butter(5, 48.0 / (RVC_SAMPLE_RATE / 2), btype="high", output="sos")
            audio_16k = sosfilt(sos, audio_16k).astype(np.float32)

            # Step 3: 反射填充
            pad_len = int(RVC_SAMPLE_RATE * 0.08)
            audio_pad = np.pad(audio_16k, pad_len, mode="reflect")

            # Step 4: ContentVec 特征提取
            phone = self._extract_content_features(audio_pad)
            if phone is None:
                logger.warning("ContentVec 特征提取失败, 回退 DSP 模拟")
                return self._infer_simulate(audio, sr, f0)

            # Step 5: F0 → pitch (mel-scale coarse + fine)
            pitch_int, pitch_f = self._f0_to_pitch(f0, sr, audio_pad.shape[0], phone)

            # Step 6: 构建输入张量
            device = self._device
            phone_t = torch.from_numpy(phone).to(device)
            pitch_t = torch.from_numpy(pitch_int).to(device)
            pitchf_t = torch.from_numpy(pitch_f).to(device)

            # Speaker ID (单 speaker 模型) + phone_lengths
            ds = torch.LongTensor([0]).to(device)
            phone_lengths = torch.LongTensor([phone.shape[1]]).to(device)

            # Step 7: 推理 (签名: phone, phone_lengths, pitch, nsff0, sid)
            with torch.no_grad():
                audio_out = self._net_g.infer(
                    phone_t, phone_lengths, pitch_t, pitchf_t, ds,
                )[0][0, 0].cpu().numpy()

            # Step 8: 后处理 — 裁剪填充
            out_trimmed = audio_out[pad_len * self._tgt_sr // RVC_SAMPLE_RATE:
                                     -pad_len * self._tgt_sr // RVC_SAMPLE_RATE] \
                          if len(audio_out) > pad_len * 2 * self._tgt_sr // RVC_SAMPLE_RATE \
                          else audio_out

            # Step 9: 重采样回原 sr
            if sr != self._tgt_sr:
                out_final = librosa.resample(
                    out_trimmed, orig_sr=self._tgt_sr, target_sr=sr,
                )
            else:
                out_final = out_trimmed

            # Step 10: 对齐长度
            target_len = len(audio)
            if len(out_final) > target_len:
                out_final = out_final[:target_len]
            elif len(out_final) < target_len:
                out_final = np.pad(out_final, (0, target_len - len(out_final)))

            # Step 11: 输出质量验证
            out_rms = np.sqrt(np.mean(out_final ** 2))
            if out_rms < 1e-6:
                logger.warning(f"PyTorch RVC 输出静音 (RMS={out_rms:.2e}), 回退 DSP")
                return self._infer_simulate(audio, sr, f0)
            if np.any(np.isnan(out_final)) or np.any(np.isinf(out_final)):
                logger.warning("PyTorch RVC 输出含 NaN/Inf, 回退 DSP")
                return self._infer_simulate(audio, sr, f0)

            logger.info(f"PyTorch RVC 推理完成 ({len(out_final)} samples, RMS={out_rms:.4f})")
            return out_final.astype(np.float32)

        except Exception as e:
            logger.error(f"PyTorch RVC 推理失败: {e}, 回退 DSP 模拟")
            return self._infer_simulate(audio, sr, f0)

    # ─── ONNX 推理 (真实 RVC) ─────────────────────────────

    def _infer_rvc(self, audio: np.ndarray, sr: int, f0: np.ndarray) -> np.ndarray:
        """ONNX RVC 推理管线"""
        try:
            # Step 1: 重采样到 16000 Hz (HuBERT 输入采样率)
            if sr != RVC_SAMPLE_RATE:
                audio_16k = librosa.resample(
                    audio.astype(np.float32), orig_sr=sr, target_sr=RVC_SAMPLE_RATE,
                )
            else:
                audio_16k = audio.astype(np.float32).copy()

            # Step 2: 高通滤波 48 Hz (去亚低频)
            sos = butter(5, 48.0 / (RVC_SAMPLE_RATE / 2), btype="high", output="sos")
            audio_16k = sosfilt(sos, audio_16k).astype(np.float32)

            # Step 3: 反射填充 (边缘伪影抑制)
            pad_len = int(RVC_SAMPLE_RATE * RVC_PAD_SEC)
            audio_pad = np.pad(audio_16k, pad_len, mode="reflect")

            # Step 4: 按正确顺序构建 ONNX 输入 (phone → pitch → pitchf → 其他)
            input_meta = {inp.name: inp for inp in self._session.get_inputs()}
            inputs = {}

            # 4a: phone (ContentVec 特征) — 必须最先提取, pitch 依赖其帧数 T
            if "phone" in input_meta:
                phone = self._extract_content_features(audio_pad)
                if phone is None:
                    logger.warning("contentvec 特征提取失败, 回退 DSP 模拟")
                    return self._infer_simulate(audio, sr, f0)
                inputs["phone"] = phone

            # 4b: pitch / pitchf (依赖 phone 的帧数 T)
            phone_for_pitch = inputs.get("phone")
            if "pitch" in input_meta or "pitchf" in input_meta:
                pitch_int, pitch_f = self._f0_to_pitch(f0, sr, audio_pad.shape[0], phone_for_pitch)
                if "pitch" in input_meta:
                    inputs["pitch"] = pitch_int
                if "pitchf" in input_meta:
                    inputs["pitchf"] = pitch_f

            # 4c: phone_lengths
            if "phone_lengths" in input_meta:
                T = inputs.get("phone", np.zeros((1, 1, 256))).shape[1]
                inputs["phone_lengths"] = np.array([T], dtype=np.int64)

            # 4d: ds (speaker ID, 单 speaker 模型恒为 0)
            if "ds" in input_meta:
                inputs["ds"] = np.array([0], dtype=np.int64)

            # 4e: rnd (随机噪声: 1×192×T)
            if "rnd" in input_meta:
                T = inputs.get("phone", np.zeros((1, 1, 256))).shape[1]
                inputs["rnd"] = np.random.RandomState(0).randn(1, 192, T).astype(np.float32)

            # 4f: audio (少数模型直接接受原始音频)
            if "audio" in input_meta:
                inputs["audio"] = audio_pad[np.newaxis, :].astype(np.float32)

            # 4g: 其他未知输入
            for name, meta in input_meta.items():
                if name in inputs:
                    continue
                shape = meta.shape
                dtype = meta.type
                if np.prod(shape) == 1 and dtype.startswith("int"):
                    inputs[name] = np.array([0], dtype=np.int64)
                elif np.prod(shape) == 1 and dtype.startswith("float"):
                    inputs[name] = np.array([0.0], dtype=np.float32)
                else:
                    logger.warning(f"未知 ONNX 输入: {name} {shape}, 跳过")

            # Step 5: ONNX 推理
            output_names = [out.name for out in self._session.get_outputs()]
            results = self._session.run(output_names, inputs)
            audio_out = results[0]  # (1, n_samples) float32

            # Step 5b: 输出质量验证 — 检测全零/静音输出
            out_flat = audio_out.flatten()
            out_rms = np.sqrt(np.mean(out_flat ** 2))
            if out_rms < 1e-6:
                logger.warning(f"ONNX 输出全静音 (RMS={out_rms:.2e}), 回退 DSP 模拟")
                return self._infer_simulate(audio, sr, f0)
            if np.any(np.isnan(out_flat)) or np.any(np.isinf(out_flat)):
                logger.warning("ONNX 输出含 NaN/Inf, 回退 DSP 模拟")
                return self._infer_simulate(audio, sr, f0)

            # Step 6: 后处理 — 裁剪填充 + 重采样回原 sr
            out_16k = audio_out[0, pad_len:-pad_len] if audio_out.shape[1] > pad_len * 2 else audio_out[0]
            if sr != RVC_SAMPLE_RATE:
                out_final = librosa.resample(
                    out_16k, orig_sr=RVC_SAMPLE_RATE, target_sr=sr,
                )
            else:
                out_final = out_16k

            # 对齐长度
            target_len = len(audio)
            if len(out_final) > target_len:
                out_final = out_final[:target_len]
            elif len(out_final) < target_len:
                out_final = np.pad(out_final, (0, target_len - len(out_final)))

            logger.info(f"RVC ONNX 推理完成 ({len(out_final)} samples, RMS={out_rms:.4f})")
            return out_final.astype(np.float32)

        except Exception as e:
            logger.error(f"RVC ONNX 推理失败: {e}, 回退 DSP 模拟")
            return self._infer_simulate(audio, sr, f0)

    def _extract_content_features(self, audio_16k: np.ndarray) -> np.ndarray:
        """
        从 ContentVec/ContentVec ONNX 模型提取语义特征 (phone tensor)。

        策略:
          A. 使用已缓存的 self._vec_session
          B. 尝试加载 vec-256-layer-9.onnx / contentvec.onnx
          C. 如果 RVC 模型自身有 audio 输入 (content encoder 已 trace in) → 跳过

        音频输入应为 16kHz mono float32。
        """
        # 策略 A: 已缓存的 vec session
        session = self._vec_session
        if session is None:
            # 策略 B: 动态加载并缓存
            for name in ["vec-256-layer-9.onnx", "contentvec.onnx", "hubert_base.onnx"]:
                path = os.path.join(self.models_dir, name)
                if os.path.exists(path):
                    try:
                        session = self._ort.InferenceSession(
                            path, providers=["CPUExecutionProvider"],
                        )
                        self._vec_session = session
                        logger.info(f"ContentVec 动态加载成功: {path}")
                        break
                    except Exception as e:
                        logger.warning(f"加载 {name} 失败: {e}")

        if session is not None:
            try:
                # 构建所有输入 (ContentVec 通常有 3 个输入: source + 2 个 embed_dim)
                inputs = {}
                for inp in session.get_inputs():
                    shape = inp.shape
                    name = inp.name
                    if name == "source" or (len(shape) >= 2 and shape[-1] != 1):
                        # 音频输入: (1, 1, T) 或 (1, T)
                        if shape and len(shape) == 3:
                            inputs[name] = audio_16k[np.newaxis, np.newaxis, :].astype(np.float32)
                        else:
                            inputs[name] = audio_16k[np.newaxis, :].astype(np.float32)
                    else:
                        # 辅助输入 (embed_dim 等) — 填零
                        inputs[name] = np.zeros([1], dtype=np.float32)
                features = session.run(None, inputs)[0]
                logger.info(f"ContentVec 特征提取成功: {features.shape}")
                return features.astype(np.float32)
            except Exception as e:
                logger.warning(f"ContentVec 推理失败: {e}")
                self._vec_session = None

        logger.warning(
            "ContentVec 模型未找到 (vec-256-layer-9.onnx / contentvec.onnx)。\n"
            f"  请运行: python scripts/download_rvc_models.py\n"
            f"  或将 .onnx 模型放入 {self.models_dir}/\n"
            f"  回退到 DSP 模拟模式。"
        )
        return None

    def _f0_to_pitch(self, f0: np.ndarray, sr: int, target_samples: int, phone: np.ndarray = None) -> tuple:
        """
        F0 → coarse pitch (mel-scale 1~255) + fine pitch (Hz)

        phone 用于确定帧数 T, 否则从 target_samples 按 RVC_HOP_LENGTH 计算。
        """
        if phone is not None:
            T = phone.shape[1]
        else:
            T = max(1, target_samples // RVC_HOP_LENGTH)

        # 将 f0 重采样到 RVC 帧率
        n_f0_frames = len(f0)

        # F0 帧对齐
        if n_f0_frames != T:
            x_old = np.linspace(0, 1, n_f0_frames)
            x_new = np.linspace(0, 1, T)
            f0_aligned = np.interp(x_new, x_old, f0)
        else:
            f0_aligned = f0.copy()

        # mel-scale coarse pitch
        f0_mel_min = 1127.0 * np.log1p(RVC_F0_MIN / 700.0)
        f0_mel_max = 1127.0 * np.log1p(RVC_F0_MAX / 700.0)
        f0_clipped = np.clip(f0_aligned, RVC_F0_MIN, RVC_F0_MAX)
        f0_mel = 1127.0 * np.log1p(f0_clipped / 700.0)
        pitch_int = np.clip(
            np.round((f0_mel - f0_mel_min) / (f0_mel_max - f0_mel_min) * 254 + 1),
            1, 255,
        ).astype(np.int64)
        pitch_int[f0_aligned <= 1] = 1  # 静音帧 → 1

        pitch_tensor = pitch_int[np.newaxis, :]  # (1, T)
        pitchf_tensor = f0_aligned[np.newaxis, :].astype(np.float32)  # (1, T)

        return pitch_tensor, pitchf_tensor

    # ─── DSP 模拟回退 ─────────────────────────────────────

    def _infer_simulate(self, audio: np.ndarray, sr: int, f0: np.ndarray) -> np.ndarray:
        """
        DSP 模拟推理 (v7): 源-滤波语音转换模拟

        基于开源变声器的源-滤波模型:
          1. STFT → 倒谱包络 (声道滤波) + 精细结构 (声源激励)
          2. 声道滤波按目标角色扭曲/调整
          3. 声源激励变调到目标基频
          4. 重合成 + 混响润色

        角色特化参数表定义每个角色的 formant/pitch/reverb/EQ 组合。
        """
        from dsp.pitch_shift import pitch_shift
        from dsp.effects import comb_reverb, noise_gate
        from dsp.formant_shift import formant_shift
        from dsp.postprocessor import parametric_eq, normalize_loudness

        # 角色特化参数 (v8: 补齐 woman_1, 保留 formant 方向)
        model_effects = {
            "woman_1": {
                "formant": 1.15,
                "pitch": 3,
                "reverb": 0.1,
                "bass_boost": -2,
                "treble_boost": 2,
            },
            "kobe": {
                "formant": 0.88,
                "pitch": -3,
                "reverb": 0.25,
                "bass_boost": 4,
                "treble_boost": -2,
            },
            "spongebob": {
                "formant": 1.35,
                "pitch": 7,
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

        # Step 2: 共振峰移位 (声道滤波改变 → 音色/体型感)
        if effects["formant"] != 1.0:
            output = formant_shift(output, sr, effects["formant"])
        else:
            output = output.copy()

        # Step 3: 角色 EQ 塑形
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

        # Step 7: LUFS 响度归一化 (块级门控, 不放大静音段)
        output = normalize_loudness(output, sr)

        return output.astype(np.float32)

    # ─── 模型信息 ──────────────────────────────────────────

    def get_model_info(self) -> dict:
        """获取当前模型信息 (用于 UI 显示)"""
        available_models = self.list_models()
        mode = "DSP模拟"
        if self.current_model:
            m = self.current_model.get("mode")
            if m == "onnx" and self._session is not None:
                mode = "RVC (ONNX)"
            elif m == "pytorch" and self._net_g is not None:
                mode = "RVC (PyTorch)"
            else:
                mode = "DSP模拟"

        return {
            "name": self.current_model_name,
            "loaded": self.current_model is not None,
            "device": self._device,
            "rvc_available": self._rvc_available,
            "onnx_available": self._ort_available,
            "mode": mode,
            "model_file": self.current_model.get("path") if self.current_model else None,
            "available_models": [m["name"] + "." + ("onnx" if m["type"] == "ONNX" else "pth")
                                 for m in available_models],
        }
