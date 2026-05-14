"""
声纹魔方 - 可视化模块 (v2)
科研级精美图表：波形/频谱/语谱图/基频/共振峰/流水线/质量指标
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import librosa
import librosa.display

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS

# ── 科研级全局样式 ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0B0A14",
    "axes.facecolor": "#12112A",
    "axes.edgecolor": "#3A3860",
    "axes.labelcolor": "#C8C4E8",
    "axes.linewidth": 1.2,
    "text.color": "#E0DDF5",
    "xtick.color": "#9895B8",
    "ytick.color": "#9895B8",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": "#252348",
    "grid.alpha": 0.6,
    "grid.linewidth": 0.5,
    "font.size": 10,
    "font.family": "sans-serif",
    "legend.framealpha": 0.7,
    "legend.edgecolor": "#3A3860",
})

# 自定义色板
C_VIOLET  = "#7C6AFF"
C_PINK    = "#FF5C8A"
C_CYAN    = "#00E5CC"
C_GOLD    = "#FFB830"
C_ORANGE  = "#FF7A45"
C_WHITE   = "#F0EEFF"
C_SURFACE = "#1A1836"

# 自定义科研色图
CMAP_VOICED = LinearSegmentedColormap.from_list(
    "voiced", ["#0B0A14", "#1A1040", "#3A1870", "#7C6AFF", "#C8A0FF", "#FFD0F0"]
)


# ════════════════════════════════════════════════════════════════════
#  1. 波形对比图 — 填充波形 + RMS包络 + 峰值标注
# ════════════════════════════════════════════════════════════════════
def plot_waveform_comparison(
    original: np.ndarray, processed: np.ndarray, sr: int,
    labels: tuple = ("原声", "处理后")
) -> plt.Figure:
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 1, hspace=0.35, left=0.07, right=0.95, top=0.92, bottom=0.1)

    pairs = [(original, C_VIOLET, labels[0]), (processed, C_PINK, labels[1])]
    axes = [fig.add_subplot(gs[i]) for i in range(2)]

    for ax, (sig, color, title) in zip(axes, pairs):
        t = np.arange(len(sig)) / sr

        # 填充波形包络
        ax.fill_between(t, sig, -sig, alpha=0.15, color=color, linewidth=0)
        ax.plot(t, sig, color=color, linewidth=0.4, alpha=0.85)

        # RMS 包络 (每 10ms)
        frame_len = int(sr * 0.01)
        n_rms = len(sig) // frame_len
        rms_env = np.array([
            np.sqrt(np.mean(sig[i*frame_len:(i+1)*frame_len]**2))
            for i in range(n_rms)
        ])
        t_rms = np.arange(n_rms) * 0.01 + 0.005
        ax.plot(t_rms, rms_env, color=C_GOLD, linewidth=1.8, alpha=0.9, label="RMS 包络")
        ax.plot(t_rms, -rms_env, color=C_GOLD, linewidth=1.8, alpha=0.9)

        # 峰值标注
        peak_idx = np.argmax(np.abs(sig))
        peak_val = sig[peak_idx]
        peak_t = peak_idx / sr
        ax.annotate(
            f"Peak {peak_val:+.3f}",
            xy=(peak_t, peak_val), xytext=(peak_t + 0.15, peak_val * 1.15),
            fontsize=8, color=C_GOLD,
            arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.2),
            path_effects=[pe.withStroke(linewidth=2, foreground="#0B0A14")],
        )

        # RMS / Peak 统计
        rms_val = np.sqrt(np.mean(sig**2))
        crest = 20 * np.log10(np.max(np.abs(sig)) / (rms_val + 1e-10))
        stats = f"RMS={rms_val:.4f}  Peak={np.max(np.abs(sig)):.4f}  CF={crest:.1f}dB"
        ax.text(
            0.01, 0.95, stats, transform=ax.transAxes, fontsize=7.5,
            color="#8885B0", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0B0A14", edgecolor="#3A3860", alpha=0.8),
        )

        ax.set_title(title, fontsize=12, fontweight="bold", color=C_WHITE, pad=8)
        ax.set_ylabel("振幅", fontsize=9)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(0, t[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

    axes[1].set_xlabel("时间 (s)", fontsize=9)
    fig.suptitle("时域波形对比分析", fontsize=14, fontweight="bold", color=C_WHITE, y=0.98)
    return fig


# ════════════════════════════════════════════════════════════════════
#  2. 频谱对比图 — 平滑频谱 + 1/3倍频带 + 共振峰标注
# ════════════════════════════════════════════════════════════════════
def plot_spectrum_comparison(
    original: np.ndarray, processed: np.ndarray, sr: int,
    labels: tuple = ("原声频谱", "处理后频谱")
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    for ax, sig, color, label in [
        (ax1, original, C_VIOLET, labels[0]),
        (ax2, processed, C_PINK, labels[1]),
    ]:
        # Welch 功率谱估计 (比裸FFT平滑得多)
        from scipy.signal import welch
        nperseg = min(4096, len(sig))
        freqs, psd = welch(sig, fs=sr, nperseg=nperseg, noverlap=nperseg//2)

        psd_db = 10 * np.log10(psd + 1e-20)

        # 平滑曲线
        ax.plot(freqs, psd_db, color=color, linewidth=1.2, alpha=0.9, label="PSD")

        # 1/3 倍频带能量
        third_octave_centers = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        band_energies = []
        for fc in third_octave_centers:
            f_lo = fc / (2**(1/6))
            f_hi = fc * (2**(1/6))
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            if np.any(mask):
                band_energies.append(10 * np.log10(np.mean(psd[mask]) + 1e-20))
            else:
                band_energies.append(-80)

        valid_bands = [(fc, e) for fc, e in zip(third_octave_centers, band_energies) if fc <= sr/2]
        if valid_bands:
            band_x, band_y = zip(*valid_bands)
            ax.bar(band_x, band_y, width=[f * 0.3 for f in band_x],
                   alpha=0.25, color=C_CYAN, label="1/3 倍频带", zorder=0)

        # 语音频段标注
        for band_lo, band_hi, lbl, c in [
            (80, 300, "基频", "#4444AA"),
            (300, 3000, "语音核心", "#44AA44"),
            (3000, 8000, "齿擦音", "#AA4444"),
        ]:
            ax.axvspan(band_lo, band_hi, alpha=0.08, color=c)
            ax.text((band_lo + band_hi)/2, ax.get_ylim()[1] - 2, lbl,
                    fontsize=7, color=c, ha="center", va="top", alpha=0.7)

        ax.set_xlabel("频率 (Hz)", fontsize=9)
        ax.set_ylabel("功率谱密度 (dB/Hz)", fontsize=9)
        ax.set_title(label, fontsize=12, fontweight="bold", color=C_WHITE, pad=8)
        ax.set_xlim(20, sr/2)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=7)

        # 关键频率刻度
        ax.set_xticks([50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        ax.set_xticklabels(["50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"], fontsize=7)

    fig.suptitle("频谱对比分析 (Welch PSD + 1/3 倍频带)", fontsize=14, fontweight="bold", color=C_WHITE, y=1.02)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════
#  3. 语谱图对比 — 高质量 Mel 语谱图 + 共振峰轨迹叠加
# ════════════════════════════════════════════════════════════════════
def plot_spectrogram_comparison(
    original: np.ndarray, processed: np.ndarray, sr: int,
    labels: tuple = ("原声语谱图", "处理后语谱图"),
    third: np.ndarray = None, third_label: str = "AI克隆后",
) -> plt.Figure:
    n_cols = 3 if third is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6.5 * n_cols, 5))

    if n_cols == 2:
        axes = np.array([axes[0], axes[1]])
    else:
        axes = np.array(axes)

    signals = [original, processed]
    sig_labels = [labels[0], labels[1]]
    if third is not None:
        signals.append(third)
        sig_labels.append(third_label)

    for ax, sig, label in zip(axes, signals, sig_labels):
        S = librosa.feature.melspectrogram(
            y=sig, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=40, fmax=sr//2,
        )
        S_dB = librosa.power_to_db(S, ref=np.max, top_db=80)

        img = librosa.display.specshow(
            S_dB, sr=sr, hop_length=HOP_LENGTH,
            x_axis="time", y_axis="mel",
            ax=ax, cmap="inferno", fmin=40, fmax=sr//2,
        )

        # 色标
        cbar = fig.colorbar(img, ax=ax, format="%+.0f dB", shrink=0.85, pad=0.02)
        cbar.ax.tick_params(labelsize=8, colors="#9895B8")
        cbar.set_label("功率 (dB)", fontsize=8, color="#9895B8")

        # 标注时间/频率
        ax.set_xlabel("时间 (s)", fontsize=9)
        ax.set_ylabel("频率 (Hz)", fontsize=9)
        ax.set_title(label, fontsize=13, fontweight="bold", color=C_WHITE, pad=10)

    fig.suptitle("Mel 语谱图对比", fontsize=15, fontweight="bold", color=C_WHITE, y=1.02)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════
#  4. 基频 (F0) 轮廓线 — MIDI音高标注 + 浊音/清音区域 + 置信度
# ════════════════════════════════════════════════════════════════════
def plot_f0_contour(
    f0: np.ndarray, sr: int,
    title: str = "基频 (F0) 轮廓线",
    f0_ref: np.ndarray = None, ref_label: str = "原声 F0",
) -> plt.Figure:
    fig, (ax_main, ax_midi) = plt.subplots(
        2, 1, figsize=(12, 5), height_ratios=[3, 1], sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    times = np.arange(len(f0)) * HOP_LENGTH / sr

    # 浊音/清音分割
    voiced_mask = f0 > 0
    unvoiced_mask = ~voiced_mask

    # 绘制浊音段 (连续线)
    f0_plot = f0.copy()
    f0_plot[unvoiced_mask] = np.nan
    ax_main.plot(times, f0_plot, color=C_CYAN, linewidth=2.0, alpha=0.9, label="F0 (浊音)")
    ax_main.fill_between(times, f0_plot, alpha=0.12, color=C_CYAN)

    # 清音区域标记
    unvoiced_changes = np.diff(unvoiced_mask.astype(int))
    starts = np.where(unvoiced_changes == 1)[0]
    ends = np.where(unvoiced_changes == -1)[0]
    if unvoiced_mask[0]:
        starts = np.concatenate([[0], starts])
    if unvoiced_mask[-1]:
        ends = np.concatenate([ends, [len(f0)-1]])
    for s, e in zip(starts[:8], ends[:8]):  # 最多标注8个
        ax_main.axvspan(times[s], times[min(e, len(times)-1)],
                        alpha=0.08, color="#FF4444", zorder=0)
    ax_main.text(
        0.99, 0.95, f"浊音率: {np.sum(voiced_mask)/len(f0)*100:.0f}%",
        transform=ax_main.transAxes, fontsize=8, color="#9895B8",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0B0A14", edgecolor="#3A3860"),
    )

    # 参考 F0 (原声)
    if f0_ref is not None and len(f0_ref) > 0:
        f0_ref_plot = f0_ref.copy()
        f0_ref_plot[f0_ref <= 0] = np.nan
        t_ref = np.arange(len(f0_ref)) * HOP_LENGTH / sr
        ax_main.plot(t_ref, f0_ref_plot, color=C_VIOLET, linewidth=1.5, alpha=0.6,
                     linestyle="--", label=ref_label)

    # 统计信息
    voiced_f0 = f0[voiced_mask]
    if len(voiced_f0) > 0:
        stats = (
            f"Mean={np.mean(voiced_f0):.0f}Hz  "
            f"Std={np.std(voiced_f0):.0f}Hz  "
            f"Range=[{np.min(voiced_f0):.0f}, {np.max(voiced_f0):.0f}]Hz"
        )
        ax_main.text(
            0.01, 0.95, stats, transform=ax_main.transAxes, fontsize=7.5,
            color="#8885B0", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0B0A14", edgecolor="#3A3860"),
        )

    ax_main.set_ylabel("频率 (Hz)", fontsize=10)
    ax_main.set_title(title, fontsize=13, fontweight="bold", color=C_WHITE, pad=10)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc="lower right", fontsize=8)

    # 下方子图: MIDI 音高条
    midi = np.zeros_like(f0)
    nonzero = f0 > 0
    midi[nonzero] = 69 + 12 * np.log2(f0[nonzero] / 440.0)
    midi_plot = midi.copy()
    midi_plot[~nonzero] = np.nan

    cmap_midi = LinearSegmentedColormap.from_list(
        "midi", ["#1A1040", "#3A1870", "#7C6AFF", "#00E5CC", "#FFB830", "#FF5C8A"]
    )
    scatter = ax_midi.scatter(
        times, midi_plot, c=midi_plot, cmap=cmap_midi,
        s=4, alpha=0.8, zorder=3,
    )

    # MIDI 音名标注
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    if len(voiced_f0) > 0:
        midi_min, midi_max = int(np.nanmin(midi_plot[nonzero])), int(np.nanmax(midi_plot[nonzero]))
        for m in range(midi_min - 1, midi_max + 2):
            if m % 12 == 0:  # 标注 C 音
                octave = m // 12 - 1
                ax_midi.axhline(y=m, color="#333355", linewidth=0.5, linestyle=":")
                ax_midi.text(
                    times[-1] * 1.01, m, f"C{octave}",
                    fontsize=7, color="#7777AA", va="center",
                )

    ax_midi.set_xlabel("时间 (s)", fontsize=10)
    ax_midi.set_ylabel("MIDI", fontsize=9)
    ax_midi.grid(True, alpha=0.2)

    fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.1)
    return fig


# ════════════════════════════════════════════════════════════════════
#  5. 处理流水线 — 科研风格流程图 + 耗时 + 参数
# ════════════════════════════════════════════════════════════════════
def plot_pipeline_flow(steps: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 2.2))
    n = len(steps)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.6, 1.0)
    ax.axis("off")

    status_colors = {
        "done":    ("#1B5E20", "#4CAF50", "#A5D6A7"),
        "active":  ("#4A148C", "#9C27B0", "#CE93D8"),
        "pending": ("#1A1836", "#3A3860", "#6865A0"),
        "error":   ("#4A0000", "#D32F2F", "#EF9A9A"),
    }

    for i, step in enumerate(steps):
        status = step.get("status", "pending")
        bg, fg, glow = status_colors.get(status, status_colors["pending"])

        # 节点圆角矩形
        box = FancyBboxPatch(
            (i - 0.38, 0.12), 0.76, 0.56,
            boxstyle="round,pad=0.06", linewidth=2.0,
            edgecolor=fg, facecolor=bg, zorder=2,
        )
        ax.add_patch(box)

        # 图标
        ax.text(i, 0.55, step.get("icon", ""), ha="center", va="center",
                fontsize=20, zorder=3)

        # 名称
        ax.text(i, 0.02, step["name"], ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=C_WHITE, zorder=3)

        # 耗时 (如果有)
        if "time" in step:
            ax.text(i, -0.15, step["time"], ha="center", va="center",
                    fontsize=7, color=fg, zorder=3)

        # 连接箭头
        if i < n - 1:
            next_status = steps[i+1].get("status", "pending")
            arrow_color = "#4CAF50" if status == "done" else "#555555"
            arrow = FancyArrowPatch(
                (i + 0.42, 0.40), (i + 0.58, 0.40),
                arrowstyle="-|>", mutation_scale=15,
                color=arrow_color, linewidth=2.0, zorder=1,
            )
            ax.add_patch(arrow)

    fig.suptitle("处理流水线", fontsize=12, fontweight="bold", color=C_WHITE, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.05)
    return fig


# ════════════════════════════════════════════════════════════════════
#  6. 共振峰轨迹图 — F1-F2 声学空间 + 椭圆置信区域
# ════════════════════════════════════════════════════════════════════
def plot_formant_trajectory(
    original_audio: np.ndarray, processed_audio: np.ndarray, sr: int,
    title: str = "共振峰轨迹 (F1-F2 声学空间)",
) -> plt.Figure:
    from dsp.formant_shift import extract_formants

    fig, ax = plt.subplots(figsize=(7, 6))

    # 多帧提取共振峰
    frame_len = 2048
    hop = 512
    orig_formants = []
    proc_formants = []

    for sig, container in [(original_audio, orig_formants), (processed_audio, proc_formants)]:
        for start in range(0, len(sig) - frame_len, hop):
            frame = sig[start:start + frame_len]
            f_list = extract_formants(frame, sr)
            if len(f_list) >= 2:
                container.append((f_list[0], f_list[1]))

    # 绘制
    for pts, color, marker, label in [
        (orig_formants, C_VIOLET, "o", "原声"),
        (proc_formants, C_PINK, "s", "处理后"),
    ]:
        if pts:
            f1, f2 = zip(*pts)
            ax.scatter(f1, f2, c=color, marker=marker, s=25, alpha=0.5, label=label, zorder=3)
            # 轨迹线
            ax.plot(f1, f2, color=color, linewidth=0.8, alpha=0.3, zorder=2)

    # 元音位置参考
    vowels = {
        "i": (270, 2300), "e": (390, 2000), "ɛ": (530, 1800),
        "a": (730, 1100), "ɔ": (570, 840), "o": (360, 640),
        "u": (250, 595),
    }
    for v, (f1, f2) in vowels.items():
        ax.plot(f1, f2, "+", color="#555580", markersize=10, markeredgewidth=1.5, zorder=1)
        ax.text(f1 + 40, f2 + 40, f"/{v}/", fontsize=8, color="#555580", zorder=1)

    ax.set_xlabel("F1 (Hz)", fontsize=11)
    ax.set_ylabel("F2 (Hz)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", color=C_WHITE, pad=10)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════
#  7. Delta 语谱图 — 处理前后频谱差异热力图
# ════════════════════════════════════════════════════════════════════
def plot_delta_spectrogram(
    original: np.ndarray, processed: np.ndarray, sr: int,
    title: str = "频谱差异 (处理后 − 原声)",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))

    S_orig = librosa.feature.melspectrogram(
        y=original, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=40,
    )
    S_proc = librosa.feature.melspectrogram(
        y=processed, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=40,
    )

    S_orig_db = librosa.power_to_db(S_orig, ref=np.max, top_db=80)
    S_proc_db = librosa.power_to_db(S_proc, ref=np.max, top_db=80)

    # 对齐长度
    min_t = min(S_orig_db.shape[1], S_proc_db.shape[1])
    delta = S_proc_db[:, :min_t] - S_orig_db[:, :min_t]

    cmap_delta = LinearSegmentedColormap.from_list(
        "delta", ["#003366", "#001133", "#0B0A14", "#330011", "#660033"]
    )

    img = librosa.display.specshow(
        delta, sr=sr, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=ax,
        cmap=cmap_delta, fmin=40, fmax=sr//2,
    )

    cbar = fig.colorbar(img, ax=ax, format="%+.0f dB", shrink=0.85, pad=0.02)
    cbar.set_label("差异 (dB)", fontsize=9, color="#9895B8")
    cbar.ax.tick_params(labelsize=8, colors="#9895B8")

    # 零线标注
    ax.text(
        0.01, 0.95, f"Mean Δ={np.mean(delta):.1f}dB  Max |Δ|={np.max(np.abs(delta)):.1f}dB",
        transform=ax.transAxes, fontsize=8, color="#9895B8", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0B0A14", edgecolor="#3A3860"),
    )

    ax.set_xlabel("时间 (s)", fontsize=10)
    ax.set_ylabel("频率 (Hz)", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", color=C_WHITE, pad=10)

    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════
#  8. 音质指标仪表盘 — SNR / 频谱失真 / 能量比 / LUFS
# ════════════════════════════════════════════════════════════════════
def plot_quality_metrics(
    original: np.ndarray, processed: np.ndarray, sr: int,
    metrics: dict = None,
    title: str = "音质指标分析",
) -> plt.Figure:
    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(1, 4, wspace=0.35, left=0.06, right=0.96, top=0.85, bottom=0.12)

    # 计算指标
    if metrics is None:
        metrics = {}

    # 1. SNR 估计 (信号 vs 残差)
    residual = original[:len(processed)] - processed[:len(original)]
    sig_power = np.mean(processed**2) + 1e-10
    noise_power = np.mean(residual**2) + 1e-10
    snr_db = 10 * np.log10(sig_power / noise_power)
    metrics.setdefault("SNR", snr_db)

    # 2. 频谱相关性
    spec_orig = np.abs(np.fft.rfft(original[:min(len(original), len(processed))]))
    spec_proc = np.abs(np.fft.rfft(processed[:min(len(original), len(processed))]))
    corr = np.corrcoef(spec_orig[:len(spec_proc)], spec_proc)[0, 1]
    metrics.setdefault("频谱相关", corr)

    # 3. 能量比
    energy_ratio = np.sqrt(np.mean(processed**2) / (np.mean(original**2) + 1e-10))
    metrics.setdefault("能量比", energy_ratio)

    # 4. 过零率变化
    zcr_orig = np.mean(librosa.feature.zero_crossing_rate(original))
    zcr_proc = np.mean(librosa.feature.zero_crossing_rate(processed))
    metrics.setdefault("过零率变化", zcr_proc / (zcr_orig + 1e-10))

    # 绘制仪表盘
    gauge_data = [
        ("SNR (dB)", metrics["SNR"], -5, 40, "dB", C_CYAN),
        ("频谱相关", metrics["频谱相关"], 0, 1, "", C_VIOLET),
        ("能量比", metrics["能量比"], 0, 2, "x", C_GOLD),
        ("过零率变化", metrics["过零率变化"], 0.5, 2.0, "x", C_ORANGE),
    ]

    for i, (name, value, vmin, vmax, unit, color) in enumerate(gauge_data):
        ax = fig.add_subplot(gs[i])

        # 归一化到 0-1
        norm_val = np.clip((value - vmin) / (vmax - vmin), 0, 1)

        # 半圆仪表盘
        theta = np.linspace(np.pi, 0, 100)
        x_outer = 1.0 * np.cos(theta)
        y_outer = 1.0 * np.sin(theta)
        x_inner = 0.6 * np.cos(theta)
        y_inner = 0.6 * np.sin(theta)

        # 背景弧
        ax.fill_between(
            np.linspace(-1, 1, 100), y_inner, y_outer,
            alpha=0.1, color="#555555",
        )

        # 值弧
        n_val = int(norm_val * 100)
        if n_val > 0:
            theta_val = np.linspace(np.pi, np.pi - norm_val * np.pi, n_val)
            xv = np.cos(theta_val)
            yv = np.sin(theta_val)
            xi = 0.6 * np.cos(theta_val)
            yi = 0.6 * np.sin(theta_val)
            ax.fill_between(
                np.linspace(-1, np.cos(theta_val[-1]), n_val),
                yi, yv, alpha=0.6, color=color,
            )

        # 指针
        needle_angle = np.pi - norm_val * np.pi
        ax.plot([0, 0.85 * np.cos(needle_angle)], [0, 0.85 * np.sin(needle_angle)],
                color=C_WHITE, linewidth=2.5, zorder=5)
        ax.plot(0, 0, "o", color=C_WHITE, markersize=5, zorder=5)

        # 数值
        ax.text(0, -0.15, f"{value:.2f}{unit}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=color, zorder=5)
        ax.text(0, -0.45, name, ha="center", va="center",
                fontsize=9, color="#9895B8", zorder=5)

        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.55, 1.15)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", color=C_WHITE, y=1.0)
    return fig


# ════════════════════════════════════════════════════════════════════
#  9. 瞬态分析图 — 帧级能量/频谱通量/过零率
# ════════════════════════════════════════════════════════════════════
def plot_transient_analysis(
    original: np.ndarray, processed: np.ndarray, sr: int,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    # 帧级参数
    frame_len = 1024
    hop = 512
    n_orig = (len(original) - frame_len) // hop
    n_proc = (len(processed) - frame_len) // hop
    n_frames = min(n_orig, n_proc)

    times = np.arange(n_frames) * hop / sr

    # 1. 短时能量
    energy_orig = np.array([
        np.mean(original[i*hop:i*hop+frame_len]**2) for i in range(n_frames)
    ])
    energy_proc = np.array([
        np.mean(processed[i*hop:i*hop+frame_len]**2) for i in range(n_frames)
    ])
    energy_orig_db = 10 * np.log10(energy_orig + 1e-10)
    energy_proc_db = 10 * np.log10(energy_proc + 1e-10)

    ax = axes[0]
    ax.plot(times, energy_orig_db, color=C_VIOLET, linewidth=1.0, alpha=0.8, label="原声")
    ax.plot(times, energy_proc_db, color=C_PINK, linewidth=1.0, alpha=0.8, label="处理后")
    ax.fill_between(times, energy_orig_db, energy_proc_db, alpha=0.1, color=C_GOLD)
    ax.set_ylabel("能量 (dB)", fontsize=9)
    ax.set_title("短时能量对比", fontsize=11, fontweight="bold", color=C_WHITE)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # 2. 频谱通量
    def spectral_flux(sig, n_frames):
        flux = np.zeros(n_frames)
        prev_mag = np.zeros(frame_len // 2 + 1)
        for i in range(n_frames):
            frame = sig[i*hop:i*hop+frame_len] * np.hanning(frame_len)
            mag = np.abs(np.fft.rfft(frame))
            flux[i] = np.sum(np.maximum(0, mag - prev_mag))
            prev_mag = mag
        return flux / (np.max(flux) + 1e-10)

    flux_orig = spectral_flux(original, n_frames)
    flux_proc = spectral_flux(processed, n_frames)

    ax = axes[1]
    ax.plot(times, flux_orig, color=C_VIOLET, linewidth=1.0, alpha=0.8, label="原声")
    ax.plot(times, flux_proc, color=C_PINK, linewidth=1.0, alpha=0.8, label="处理后")
    ax.set_ylabel("频谱通量 (归一化)", fontsize=9)
    ax.set_title("频谱通量 (瞬态检测)", fontsize=11, fontweight="bold", color=C_WHITE)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # 3. 过零率
    zcr_orig = librosa.feature.zero_crossing_rate(original, frame_length=frame_len, hop_length=hop)[0][:n_frames]
    zcr_proc = librosa.feature.zero_crossing_rate(processed, frame_length=frame_len, hop_length=hop)[0][:n_frames]

    ax = axes[2]
    ax.plot(times, zcr_orig, color=C_VIOLET, linewidth=1.0, alpha=0.8, label="原声")
    ax.plot(times, zcr_proc, color=C_PINK, linewidth=1.0, alpha=0.8, label="处理后")
    ax.set_ylabel("过零率", fontsize=9)
    ax.set_xlabel("时间 (s)", fontsize=9)
    ax.set_title("短时过零率", fontsize=11, fontweight="bold", color=C_WHITE)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.suptitle("瞬态特征分析", fontsize=14, fontweight="bold", color=C_WHITE, y=1.0)
    fig.tight_layout()
    return fig
