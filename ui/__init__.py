"""
声纹魔方 - UI 界面组件模块
"""

from .recorder import process_audio_input, save_audio, normalize_audio, remove_dc_offset, cleanup_temp_files

__all__ = ["process_audio_input", "save_audio", "normalize_audio", "remove_dc_offset", "cleanup_temp_files"]
