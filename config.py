"""
声纹魔方 - 全局配置 (v2)
Voice Cube - Global Configuration
"""

# ==================== 音频参数 ====================
SAMPLE_RATE = 44100          # 标准采样率 (语音质量优先)
HOP_LENGTH = 512             # STFT 帧移
N_FFT = 2048                 # STFT 窗长
N_MELS = 128                 # Mel 频带数
WIN_LENGTH = 2048            # 窗长
WINDOW = "hann"              # 窗函数类型

# ==================== DSP 参数范围 ====================
PITCH_SHIFT_RANGE = (-12, 12)      # 半音范围
FORMANT_RATIO_RANGE = (0.4, 2.5)   # 共振峰缩放范围
REVERB_RANGE = (0.0, 1.0)          # 混响强度范围
RING_MOD_RANGE = (0.0, 1.0)        # 环形调制深度范围

# ==================== LPC 参数 ====================
LPC_ORDER = 46                # LPC 阶数 (fs/1000+4≈46 @44100Hz，共振峰分辨率更高)
LPC_PRE_EMPHASIS = 0.97       # LPC 预加重系数

# ==================== 谱减法参数 ====================
NOISE_FRAMES = 8              # 噪声估计帧数
NOISE_OVERSUBTRACTION = 1.5   # 过减因子 α (从2.0降到1.5，减少音乐噪声)
SPECTRAL_FLOOR = 0.04         # 谱下限 β (从0.01提到0.04，防止过度抑制)
SPECTRAL_SMOOTH_FRAMES = 3    # 频谱平滑帧数

# ==================== 滤波器参数 ====================
TELEPHONE_LOW = 300           # 电话带通低频 (Hz)
TELEPHONE_HIGH = 3400         # 电话带通高频 (Hz)
FILTER_ORDER = 127            # FIR 滤波器阶数 (从101提到127，更陡峭)

# ==================== 混响参数 ====================
REVERB_DELAYS_SEC = [0.0353, 0.0366, 0.0338, 0.0322]  # 梳状滤波器延迟 (秒)
REVERB_GAINS = [0.80, 0.82, 0.78, 0.76]               # 对应增益
ALLPASS_DELAYS_SEC = [0.0051, 0.0126]                  # 全通滤波器延迟 (秒)
ALLPASS_GAIN = 0.7                                      # 全通增益

# ==================== 压缩器参数 ====================
COMPRESSOR_THRESHOLD = -20    # 压缩阈值 (dB)
COMPRESSOR_RATIO = 4.0        # 压缩比
COMPRESSOR_ATTACK = 0.005     # 启动时间 (秒)
COMPRESSOR_RELEASE = 0.05     # 释放时间 (秒)
COMPRESSOR_KNEE = 6.0         # 软拐点 (dB)

# ==================== 均衡器参数 ====================
EQ_CROSSOVER_LOW = 250        # 低/中分频 (Hz)
EQ_CROSSOVER_HIGH = 4000      # 中/高分频 (Hz)

# ==================== 气声参数 ====================
BREATHINESS_BAND_LOW = 1500   # 气声噪声低频 (Hz)
BREATHINESS_BAND_HIGH = 8000  # 气声噪声高频 (Hz)

# ==================== 响度参数 ====================
TARGET_LUFS = -16.0           # 目标响度 (LUFS)
LOUDNESS_GATE = -40.0         # 响度门限 (dB)

# ==================== UI 配置 ====================
APP_TITLE = "声纹魔方 Voice Cube"
APP_DESCRIPTION = "基于 DSP 与混合 AI 的语音模仿秀系统"
THEME_PRIMARY = "#6C63FF"     # 主色调
THEME_SECONDARY = "#FF6584"   # 强调色
THEME_BG = "#0F0E17"          # 深色背景
THEME_SURFACE = "#1A1A2E"     # 卡片背景
THEME_TEXT = "#EAEAEA"        # 文字颜色

# ==================== 路径配置 ====================
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODELS_DIR = os.path.join(ASSETS_DIR, "models")
AVATARS_DIR = os.path.join(ASSETS_DIR, "avatars")
