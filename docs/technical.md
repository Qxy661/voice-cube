# 声纹魔方 — 算法原理技术文档

## 1. Phase Vocoder 变调算法

### 1.1 原理概述

Phase Vocoder（相位声码器）是一种在频域中改变音频信号时间尺度或音高的经典 DSP 算法。其核心思想是将时域信号通过 STFT 转换到频域，在频域中对频率轴进行缩放，然后通过 ISTFT 重建时域信号。

### 1.2 数学推导

**STFT 分析**：

$$X(k, n) = \sum_{m=0}^{N-1} x(nH + m) \cdot w(m) \cdot e^{-j2\pi km/N}$$

其中：
- $x(n)$ 为输入信号
- $w(m)$ 为窗函数（汉宁窗）
- $H$ 为帧移（hop length）
- $N$ 为 FFT 点数
- $k$ 为频率索引，$n$ 为帧索引

**音高缩放**：

音高偏移 $s$ 半音对应的频率缩放因子：

$$\alpha = 2^{s/12}$$

在频域中，将幅度谱的频率轴按 $\alpha$ 缩放：

$$|Y(k)| = |X(k/\alpha)|$$

**相位声码器修正**：

直接缩放频率轴会导致相位不连续。Phase Vocoder 通过累积相位来解决：

$$\phi_Y(k, n) = \phi_Y(k, n-1) + \frac{2\pi k \cdot H}{N} + \Delta\phi(k, n)$$

其中 $\Delta\phi$ 为瞬时相位偏差。

**ISTFT 重建**：

$$y(n) = \frac{1}{N} \sum_{k=0}^{N-1} Y(k, n) \cdot e^{j2\pi kn/N}$$

### 1.3 实现要点

- 帧长 N = 2048，帧移 H = 512
- 窗函数：汉宁窗
- 使用 `librosa.phase_vocoder()` 实现相位修正
- **v3 改进**：变调后使用 `librosa.resample()` 多相重采样恢复原始时长，替代线性插值，减少混叠伪影

---

## 2. LPC 共振峰移位算法

### 2.1 原理概述

线性预测编码（LPC）假设语音信号可以用一个全极点模型描述：

$$H(z) = \frac{1}{1 - \sum_{i=1}^{p} a_i z^{-i}}$$

其中 $a_i$ 为 LPC 系数，$p$ 为预测阶数。

LPC 多项式的根（极点）对应声道的共振峰频率。通过移动极点位置，可以改变声音的"体型"特征。

### 2.2 共振峰提取

1. 对语音帧做 LPC 分析，得到系数 $\{a_i\}$
2. 求 LPC 多项式的根：$1 - \sum a_i z^{-i} = 0$
3. 筛选单位圆内的共轭极点对
4. 极点的角频率对应共振峰频率：

$$F_k = \frac{\theta_k \cdot f_s}{2\pi}$$

### 2.3 共振峰移位

将每个极点的角度按比例缩放：

$$\theta_k' = \theta_k \cdot r$$

其中 $r$ 为缩放比例：
- $r < 1$：共振峰频率降低 → 小体型（小黄人）
- $r = 1$：不变
- $r > 1$：共振峰频率升高 → 大体型（巨人）

### 2.4 LPC 合成

用移位后的极点重建 LPC 系数，然后对原始激励信号（残差）进行滤波：

$$\hat{x}(n) = e(n) * h'(n)$$

其中 $e(n)$ 为 LPC 残差，$h'(n)$ 为新的声道脉冲响应。

### 2.5 v3 改进

- **LPC 阶数**：16 → 24，更高阶数捕捉更多共振峰细节
- **预加重**：0.97 系数高通滤波，补偿语音信号的 -6dB/oct 衰减
- **反射系数裁剪**：防止数值不稳定导致滤波器发散
- **稳定性检查**：重建滤波器后验证所有极点在单位圆内

---

## 3. 谱减法降噪

### 3.1 基础原理

谱减法假设噪声是加性的：

$$x(n) = s(n) + d(n)$$

在频域中：

$$|X(\omega)|^2 = |S(\omega)|^2 + |D(\omega)|^2$$

因此干净信号的功率谱估计为：

$$|\hat{S}(\omega)|^2 = |X(\omega)|^2 - \alpha \cdot |\hat{D}(\omega)|^2$$

其中 $\alpha$ 为过减因子，$|\hat{D}(\omega)|^2$ 为噪声功率谱估计。

### 3.2 半波整流与谱下限

为防止负值，引入谱下限 $\beta$：

$$|\hat{S}(\omega)|^2 = \max\left(|\hat{S}(\omega)|^2, \quad \beta \cdot |X(\omega)|^2\right)$$

### 3.3 Wiener 滤波改进（v3）

v3 使用 Wiener 滤波器增益替代简单的谱减法：

$$G(\omega) = \max\left(1 - \alpha \cdot \frac{|\hat{N}(\omega)|^2}{|X(\omega)|^2}, \quad \beta\right)$$

$$|\hat{S}(\omega)| = G(\omega) \cdot |X(\omega)|$$

改进点：
- **帧间平滑**：增益函数跨帧平滑，减少音乐噪声
- **中值噪声估计**：使用中值而非均值估计噪声功率谱，对突发噪声更鲁棒
- **参数调优**：过减因子 α=2.0→1.5（减少语音失真），谱下限 β=0.01→0.04（保留更多背景细节）

---

## 4. 环形调制（Ring Modulation）

### 4.1 原理

环形调制是一种幅度调制技术：

$$y(n) = x(n) \cdot \sin(2\pi f_m n / f_s)$$

展开后：

$$y(n) = \frac{1}{2}\left[X(f - f_m) + X(f + f_m)\right]$$

原始信号的频谱被搬移到 $f_m$ 两侧，产生边带频率，产生金属/机械质感。

### 4.2 改进版

为保留原始信号可辨识度，加入直流偏移：

$$y(n) = x(n) \cdot (1 - d + d \cdot \sin(2\pi f_m n / f_s))$$

其中 $d$ 为调制深度。

---

## 5. Schroeder 混响

### 5.1 结构

经典 Schroeder 混响由以下部分组成：

```
输入 ─┬─ 梳状滤波器1 ─┐
      ├─ 梳状滤波器2 ─┤
      ├─ 梳状滤波器3 ─┼─ 求和 → 全通滤波器1 → 全通滤波器2 → 输出
      └─ 梳状滤波器4 ─┘
```

### 5.2 梳状滤波器（递归反馈）

$$y(n) = x(n) + g \cdot y(n - M)$$

其中 $M$ 为延迟采样数，$g$ 为反馈增益。

**v3 改进**：改为递归反馈结构（IIR），非简单的 FIR 延迟叠加，产生更自然的混响尾音。

### 5.3 全通滤波器

$$y(n) = -g \cdot x(n) + x(n - M) + g \cdot y(n - M)$$

全通滤波器只改变相位响应，不影响幅度响应。

---

## 6. 基频提取（F0 Estimation）

### 6.1 自相关法

1. 分帧加窗
2. 计算自相关函数：

$$R(\tau) = \sum_{n=0}^{N-1} x(n) \cdot x(n + \tau)$$

3. 在合理延迟范围 $[\tau_{min}, \tau_{max}]$ 内找峰值
4. 峰值位置 $\tau_0$ 对应基频周期：

$$f_0 = \frac{f_s}{\tau_0}$$

### 6.2 概率 YIN 算法

`librosa.pyin()` 实现了 YIN 算法的改进版本，引入概率模型处理清浊音判断，鲁棒性更强。

### 6.3 F0 → MIDI 转换（v3 新增）

$$\text{MIDI} = 69 + 12 \cdot \log_2\left(\frac{f_0}{440}\right)$$

MIDI 0 对应 F0=0（静音），MIDI 69 对应 440Hz（A4）。

---

## 7. 软拐点压缩器（v3 新增）

### 7.1 原理

动态范围压缩器减小信号的动态范围，让安静部分相对变响，提升整体响度一致性。

### 7.2 包络跟随器

对信号幅度取包络，使用 attack/release 时间常数平滑：

$$\text{env}(n) = \begin{cases} \alpha_a \cdot \text{env}(n-1) + (1-\alpha_a) \cdot |x(n)| & \text{if } |x(n)| > \text{env}(n-1) \\ \alpha_r \cdot \text{env}(n-1) + (1-\alpha_r) \cdot |x(n)| & \text{otherwise} \end{cases}$$

其中 $\alpha_a$ 和 $\alpha_r$ 分别为 attack 和 release 的平滑系数。

### 7.3 软拐点增益计算

当信号幅度超过阈值 $T$ 时，按压缩比 $R$ 计算增益衰减：

$$G(\text{dB}) = \begin{cases} 0 & \text{if } \text{level} < T - W/2 \\ \frac{(\text{level} - T + W/2)^2}{2W} \cdot \left(\frac{1}{R} - 1\right) & \text{if } T - W/2 \leq \text{level} \leq T + W/2 \\ (T - \text{level}) \cdot \left(1 - \frac{1}{R}\right) & \text{if } \text{level} > T + W/2 \end{cases}$$

其中 $W$ 为软拐点宽度。

### 7.4 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| threshold_db | -20 dB | 压缩阈值 |
| ratio | 4:1 | 压缩比 |
| attack_ms | 10 ms | 启动时间 |
| release_ms | 100 ms | 释放时间 |
| knee_db | 6 dB | 软拐点宽度 |

---

## 8. 谐波激励器（v3 新增）

### 8.1 原理

谐波激励器通过在高频段产生谐波失真，增加声音的清晰度和"存在感"。

### 8.2 处理流程

```
输入 → 带通滤波(2-8kHz) → tanh软削波 → 混合(原声 + amount × 谐波) → 输出
```

### 8.3 tanh 软削波

$$y(n) = \tanh(g \cdot x(n))$$

其中 $g$ 为增益系数。tanh 函数在零点附近近似线性，在大幅度时产生平滑的饱和，产生偶次和奇次谐波。

### 8.4 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| amount | 0.0 - 1.0 | 激励量（混合比例） |
| freq_low | 2000 Hz | 带通下限 |
| freq_high | 8000 Hz | 带通上限 |

---

## 9. LUFS 响度归一化（v3 新增）

### 9.1 标准

基于 ITU-R BS.1770 响度测量标准。LUFS (Loudness Units relative to Full Scale) 是当前流媒体平台（Spotify、YouTube、Apple Music）统一使用的响度单位。

### 9.2 K-weighting 滤波

在测量响度前，对信号施加 K-weighting 滤波器，模拟人耳对不同频率的感知灵敏度：

1. **高频 shelving 滤波器**：+4dB @ 4kHz（补偿人耳高频灵敏度）
2. **高通滤波器**：截止频率 38Hz（忽略次声频段）

### 9.3 响度计算

将信号分为 400ms 帧，每帧重叠 75%，计算每帧的均方能量：

$$L_k = -0.691 + 10 \cdot \log_{10}\left(\sum_i w_i \cdot \text{MS}_i\right) \quad \text{[LUFS]}$$

### 9.4 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| target_lufs | -16 LUFS | 目标响度（流媒体标准） |
| gate_threshold | -70 LUFS | 静音门限 |

---

## 10. AI 克隆流水线设计

### 10.1 设计理念

**"DSP 负责精准剥离特征与预/后处理，AI 负责非线性映射"**

```
原声 → [DC去除] → [Wiener降噪] → [F0提取] → [RVC音色重构] → [压缩器] → [混响] → [EQ] → [LUFS] → 成品
```

### 10.2 各步骤作用

| 步骤 | 技术 | 作用 |
|------|------|------|
| DC去除 | 减均值 | 消除直流偏移，防止后续处理引入误差 |
| 降噪 | Wiener 滤波 | 去除环境底噪，防止"垃圾进垃圾出" |
| 基频提取 | 自相关/pYIN | 精确提取 F0 轮廓线，保留语调 |
| 音色重构 | RVC | 非线性音色映射，克隆目标音色 |
| 压缩器 | 软拐点压缩 | 统一动态范围，提升响度一致性 |
| 混响 | Schroeder | 让声音更自然，去除"干涩感" |
| EQ | 参量均衡 | 调整音色平衡 |
| LUFS | 响度归一化 | 符合流媒体响度标准 |

### 10.3 RVC (Retrieval-based Voice Conversion)

RVC 是一种基于检索的语音转换方法：
1. 使用 HuBERT 提取内容特征
2. 使用预训练的 speaker encoder 提取音色特征
3. 通过检索找到最匹配的训练样本
4. 使用 HiFi-GAN 解码器合成目标音色的语音

### 10.4 DSP 模拟回退

无 PyTorch 环境时，使用角色特化 DSP 参数链模拟变声效果：

| 角色 | 共振峰 | 变调 | 混响 | 压缩 | 激励 | EQ |
|------|--------|------|------|------|------|-----|
| 科比 | 1.2 | -2 | 0.3 | 开 | 0.12 | bass+4/treble-2 |
| 海绵宝宝 | 0.6 | +6 | 0.1 | 开 | 0.08 | bass-5/treble+3 |
| 通用 | 1.15 | -1 | 0.25 | 开 | 0.1 | bass+2/treble 0 |

---

## 11. 参量均衡器

### 11.1 三段均衡

使用双二阶 (biquad) shelving 滤波器实现：

- **低频 shelving**：提升/衰减低频段（< 300Hz）
- **中频**：保持不变
- **高频 shelving**：提升/衰减高频段（> 3000Hz）

### 11.2 双二阶滤波器

$$y(n) = b_0 x(n) + b_1 x(n-1) + b_2 x(n-2) - a_1 y(n-1) - a_2 y(n-2)$$

通过 RBJ Audio EQ Cookbook 计算 shelving 滤波器系数。

### 11.3 v3 改进

使用正确的 RBJ shelving 滤波器系数实现，替代之前错误使用的 `scipy.signal.iirpeak`（iirpeak 产生的是峰值 EQ，不是 shelving EQ）。
