# Changelog

## v3.0.0 (2026-05-17)

### 新增算法
- **软拐点压缩器** (`soft_knee_compressor`): 基于包络跟随器的动态范围压缩，支持 attack/release 参数
- **谐波激励器** (`harmonics_exciter`): 2-8kHz 带通 + tanh 软削波，增加高频清晰度与存在感
- **LUFS 响度归一化** (`normalize_loudness`): ITU-R BS.1770 简化实现，K-weighting 滤波
- **Wiener 滤波降噪**: 替代基础谱减法，帧间平滑 + 中值噪声估计
- **噪声门 v2**: attack/release 包络，避免"呼吸效应"
- **梳状滤波器金属音** (`comb_filter_metallic`): 可调频率的金属质感特效
- **F0 → MIDI 转换** (`f0_to_midi`): 基频到 MIDI 音符号映射

### 算法优化
- **变调**: Phase Vocoder 使用 `librosa.resample` 多相重采样替代线性插值，减少混叠
- **共振峰移位**: LPC 阶数 16→24，新增预加重 (0.97)、反射系数裁剪、滤波器稳定性检查
- **谱减法**: 过减因子 2.0→1.5，谱下限 0.01→0.04，增加帧间平滑
- **参量 EQ**: 使用正确的双二阶 shelving 滤波器替代 `iirpeak`
- **混响**: 梳状滤波器改为递归反馈结构 `y[n] = x[n] + g·y[n-D]`
- **气声**: 带通噪声 (1.5-8kHz) + attack/release 包络跟随器
- **FIR 滤波器**: 阶数 101→127

### RVC 引擎增强
- 模拟推理扩展到所有 AI 预设角色（kobe/spongebob）
- 新增压缩器、激励器、EQ 到模拟推理管线
- 未知模型使用通用变声链（formant=1.15, pitch=-1, compressor, exciter, reverb）

### Bug 修复
- `apply_dsp_preset` 错误路径返回值不一致导致解包崩溃
- `gr.update()` 在 Gradio 4.x 中已废弃，改用 `gr.File(visible=...)`
- 双重归一化冲突：移除 RMS 归一化，仅保留 LUFS
- 临时文件泄漏：`save_audio()` 新增模块级追踪与自动清理
- 移除未使用的 `pydub` 依赖

### 可视化改进
- CJK 字体自动检测（Microsoft YaHei / SimHei / Noto Sans SC）
- 新增共振峰轨迹图 (`plot_formant_trajectory`)
- 新增差分语谱图 (`plot_delta_spectrogram`)
- 新增音频质量指标仪表盘 (`plot_quality_metrics`)

### 测试
- 新增 9 个单元测试（压缩器/激励器/LUFS/噪声门/梳状滤波/共振峰提取/F0-MIDI/RVC引擎/录音组件）
- 总计 17 个测试，覆盖所有核心 DSP 模块
- 测试包含信号内容断言（动态范围压缩、频段能量变化等）

### 依赖变更
- 移除 `pydub`（未使用）
- `pyproject.toml` 版本升级至 3.0.0
- 新增 `[project.optional-dependencies] ai` 段落（torch/torchaudio）
- 新增 `[project.optional-dependencies] dev` 段落（pytest/ruff）

---

## v2.0.0 (2026-05-15)

### 音频质量优化（22 轮迭代）
- Phase Vocoder 变调算法优化，减少频谱泄漏
- LPC 共振峰移位精度提升
- Schroeder 混响改为递归反馈结构
- 谱减法降噪参数调优
- 参量均衡器改为 biquad shelving 实现

### 可视化增强
- 新增 1/3 倍频程频谱图
- 新增差分语谱图
- 新增音频质量指标仪表盘
- 共振峰轨迹可视化

---

## v1.0.0 (2026-05-14)

### 核心功能
- **基础模仿区**：8 种 DSP 预设音色（低音炮男声、童声、机器人、小黄人、老式电话、巨人、耳语、金属音）
- **AI 克隆区**：RVC 音色克隆引擎，支持科比、海绵宝宝等预设模型
- **声学分析看板**：波形对比、频谱对比、语谱图对比、基频轮廓线

### DSP 算法
- Phase Vocoder 变调（变调不变速）
- LPC 共振峰移位（改变体型感）
- 环形调制（机械音效果）
- FIR 带通滤波（老式电话效果）
- Schroeder 混响
- 谱减法降噪
- 自相关法/pYIN 基频提取
- 三段参量均衡器

### AI 功能
- RVC 模型加载与推理封装
- DSP 预处理 → AI 克隆 → DSP 后处理 完整流水线
- 模拟模式（无 RVC 依赖时使用 DSP 模拟效果）

### 可视化
- 时域波形对比图
- FFT 频谱对比图
- Mel 语谱图对比（支持 2/3 列并排）
- 基频 F0 轮廓线图
- 处理流水线状态图

### 界面
- Gradio Blocks 双 Tab 布局
- 模仿预设卡片选择
- 高级滑块微调
- 麦克风录音 + 文件上传
- 播放 + 下载
