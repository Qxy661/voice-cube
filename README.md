# 声纹魔方 Voice Cube

> 基于 DSP 与混合 AI 的语音模仿秀系统

一个支持麦克风实时录音或文件上传的变声工作台。系统提供两条核心变声链路：**纯 DSP 模仿经典音色** 和 **DSP+AI 克隆特定人物**，每一步处理都有可视化的波形和频谱反馈。

## 系统特色

- **基础模仿区（纯 DSP）**：8 种预设音色（低音炮男声、童声、机器人、小黄人、老式电话、巨人、耳语、金属音），纯算法实现，无深度学习黑盒
- **AI 克隆区（DSP+AI）**：RVC 音色克隆引擎，支持科比、海绵宝宝等热门人物一键变身
- **11 步 DSP 管线**：变调 → 共振峰 → 环调 → 电话滤波 → 梳状滤波 → 气声 → 激励器 → 压缩器 → 混响 → EQ → LUFS 归一化
- **声学分析看板**：原声 vs DSP 处理 vs AI 克隆的波形、频谱、语谱图、共振峰轨迹、质量指标仪表盘
- **完整闭环**：录音/上传 → 处理 → 播放/下载 → 可视化分析

## 技术架构

```
用户输入 → DC去除 → LUFS归一化 → DSP 处理管线(11步) → 可视化输出
                         │
                         ├─ 基础模仿：变调 + 共振峰 + 特效 + 激励器 + 压缩器 + 混响 + EQ
                         └─ AI 克隆：Wiener降噪 + F0提取 + RVC + 压缩器 + 混响 + EQ
```

## 快速开始

### 环境要求

- Python 3.9+
- 推荐使用 conda 或 venv 虚拟环境

### 安装

```bash
# 克隆项目
git clone https://github.com/Qxy661/voice-cube.git
cd voice-cube

# 安装依赖
pip install -r requirements.txt

# 启动
python app.py
```

浏览器访问 `http://localhost:7860` 即可使用。

### 可选依赖

```bash
# AI 克隆功能（需要 GPU 加速时安装）
pip install torch>=2.0.0 torchaudio>=2.0.0

# 开发工具
pip install -e ".[dev]"
```

### CJK 字体说明

可视化图表使用 matplotlib 渲染。系统已实现 CJK 字体自动检测，支持 Microsoft YaHei、SimHei、Noto Sans SC 等。如果中文标签仍显示为方块，请确认系统已安装上述字体之一。

## 项目结构

```
voice-cube/
├── app.py                  # Gradio 主界面入口
├── config.py               # 全局配置
├── requirements.txt        # 依赖清单
├── pyproject.toml          # 项目元数据
│
├── dsp/                    # 纯 DSP 引擎（核心算法）
│   ├── presets.py          # 模仿预设参数库（8个DSP + 2个AI）
│   ├── pitch_shift.py      # Phase Vocoder 变调（多相重采样）
│   ├── formant_shift.py    # LPC 共振峰移位（LPC-24 + 预加重）
│   ├── effects.py          # 环形调制/滤波器/混响/压缩器/激励器
│   ├── preprocessor.py     # Wiener滤波降噪/噪声门
│   ├── pitch_extract.py    # 基频提取 + F0-MIDI 转换
│   ├── postprocessor.py    # 参量EQ/混响/LUFS归一化
│   └── visualizer.py       # 9种图表（含共振峰轨迹/质量指标）
│
├── ai/                     # AI 克隆引擎
│   └── rvc_engine.py       # RVC 模型封装 + DSP模拟回退
│
├── ui/                     # 界面组件
│   └── recorder.py         # 音频输入处理/DC去除/临时文件管理
│
├── tests/                  # 单元测试（17个）
│   └── test_dsp.py         # DSP/AI/UI 全模块测试
│
├── assets/                 # 静态资源
│   ├── avatars/            # 人物头像
│   └── models/             # RVC 模型权重
│
└── docs/                   # 文档
    ├── technical.md        # 算法原理文档
    └── architecture.md     # 系统架构文档
```

## DSP 算法说明

### 1. Phase Vocoder 变调（变调不变速）

```
输入 → STFT分帧 → 相位累积修正 → 频率缩放 → 多相重采样 → 输出
```

利用短时傅里叶变换在频域对信号进行缩放，通过相位声码器修正相位不连续，使用 librosa.resample 多相重采样恢复原始时长。

### 2. LPC 共振峰移位（改变体型感）

```
输入 → 预加重 → LPC分析(24阶) → 求根找共振峰 → 移动极点 → 稳定性检查 → LPC合成 → 输出
```

通过线性预测编码提取声道共振峰频率，将极点在单位圆上按比例移动，改变声音的"体型"特征。

### 3. Wiener 滤波降噪

```
|S(ω)|² = G(ω)·|X(ω)|²,  G(ω) = max(1 - α·|N(ω)|²/|X(ω)|², β)
```

基于 Wiener 滤波器的谱减法，过减因子 α=1.5，谱下限 β=0.04，帧间平滑减少音乐噪声。

### 4. 软拐点压缩器

```
输入 → 包络跟随 → 增益计算(软拐点) → 增益平滑(attack/release) → 压缩输出
```

动态范围压缩，让安静部分相对变响，提升整体响度一致性。

### 5. 谐波激励器

```
输入 → 带通滤波(2-8kHz) → tanh软削波 → 混合(原声 + amount×谐波) → 输出
```

增加高频谐波，提升声音清晰度与存在感。

### 6. LUFS 响度归一化

基于 ITU-R BS.1770 简化实现，K-weighting 预滤波，目标 -16 LUFS。符合流媒体响度标准。

## AI 克隆流水线

```
原声 → [DC去除] → [Wiener降噪] → [F0提取] → [RVC音色重构] → [压缩器] → [混响] → [EQ] → [LUFS] → 成品
```

- **DSP 负责**：精准剥离特征与预/后处理
- **AI 负责**：非线性音色映射

无 PyTorch 时自动回退到 DSP 模拟模式，使用角色特化参数链实现变声效果。

## 测试

```bash
# 运行全部 17 个测试
python tests/test_dsp.py

# 使用 pytest
pytest tests/ -v
```

测试覆盖：变调、共振峰、特效、降噪、基频提取、后处理、预设库、可视化、压缩器、激励器、LUFS、噪声门、梳状滤波、共振峰提取、F0-MIDI、RVC引擎、录音组件。

## 课堂展示要点

1. **纯 DSP 能力**：8 种音色模仿全部由经典 DSP 算法实现，可展示频谱图共振峰平移
2. **AI 融合**：清晰的流水线展示"经典技术与前沿技术的强强联合"
3. **可视化闭环**：三列并排语谱图 + 共振峰轨迹 + 质量指标仪表盘
4. **工程完整度**：完整的 Web 应用，支持录音/上传/播放/下载，17 个单元测试

## License

MIT License
