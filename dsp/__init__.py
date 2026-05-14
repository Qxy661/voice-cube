"""
声纹魔方 - DSP 引擎模块 (v2)
纯数字信号处理算法实现，拒绝深度学习黑盒
"""

from .pitch_shift import pitch_shift, pitch_shift_psola
from .formant_shift import formant_shift, extract_formants
from .effects import (
    ring_modulate, telephone_filter, comb_reverb, add_breathiness,
    comb_filter_metallic, soft_knee_compressor, harmonics_exciter,
)
from .preprocessor import spectral_subtraction, noise_gate
from .pitch_extract import extract_f0, f0_to_midi
from .postprocessor import parametric_eq, apply_reverb, normalize_loudness
from .visualizer import (
    plot_waveform_comparison, plot_spectrum_comparison,
    plot_spectrogram_comparison, plot_f0_contour, plot_pipeline_flow,
    plot_formant_trajectory, plot_delta_spectrogram,
    plot_quality_metrics, plot_transient_analysis,
)
from .presets import get_preset, list_presets, DSP_PRESETS, AI_PRESETS

__all__ = [
    "pitch_shift", "pitch_shift_psola",
    "formant_shift", "extract_formants",
    "ring_modulate", "telephone_filter", "comb_reverb", "add_breathiness",
    "comb_filter_metallic", "soft_knee_compressor", "harmonics_exciter",
    "spectral_subtraction", "noise_gate",
    "extract_f0", "f0_to_midi",
    "parametric_eq", "apply_reverb", "normalize_loudness",
    "plot_waveform_comparison", "plot_spectrum_comparison",
    "plot_spectrogram_comparison", "plot_f0_contour", "plot_pipeline_flow",
    "plot_formant_trajectory", "plot_delta_spectrogram",
    "plot_quality_metrics", "plot_transient_analysis",
    "get_preset", "list_presets", "DSP_PRESETS", "AI_PRESETS",
]
