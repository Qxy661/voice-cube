"""
声纹魔方 - DSP 引擎模块
纯数字信号处理算法实现，拒绝深度学习黑盒
"""

from .pitch_shift import pitch_shift
from .formant_shift import formant_shift
from .effects import ring_modulate, telephone_filter, comb_reverb, add_breathiness
from .preprocessor import spectral_subtraction
from .pitch_extract import extract_f0
from .postprocessor import parametric_eq, apply_reverb
from .visualizer import plot_waveform_comparison, plot_spectrum_comparison, plot_spectrogram_comparison
from .presets import get_preset, list_presets, DSP_PRESETS

__all__ = [
    "pitch_shift", "formant_shift",
    "ring_modulate", "telephone_filter", "comb_reverb", "add_breathiness",
    "spectral_subtraction", "extract_f0",
    "parametric_eq", "apply_reverb",
    "plot_waveform_comparison", "plot_spectrum_comparison", "plot_spectrogram_comparison",
    "get_preset", "list_presets", "DSP_PRESETS",
]
