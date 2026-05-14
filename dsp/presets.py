"""
声纹魔方 - 模仿预设参数库 (v2)
优化后的参数：更自然的人声，更少的伪影
"""

# ==================== 基础区：DSP 模仿经典音色 ====================
DSP_PRESETS = {
    "deep_male": {
        "name": "低音炮男声",
        "icon": "🎤",
        "description": "浑厚磁性的成熟男声",
        "category": "basic",
        "params": {
            "pitch_shift": -3,
            "formant_ratio": 1.10,
            "ring_mod": 0.0,
            "ring_freq": 0,
            "telephone": False,
            "reverb": 0.08,
            "eq_bass_boost": 4,
            "eq_treble_boost": -2,
            "breathiness": 0.0,
            "comb_filter": False,
            "compressor": True,
            "exciter": 0.1,
        }
    },
    "child": {
        "name": "童声",
        "icon": "👶",
        "description": "天真可爱的儿童声音",
        "category": "basic",
        "params": {
            "pitch_shift": 5,
            "formant_ratio": 0.78,
            "ring_mod": 0.0,
            "ring_freq": 0,
            "telephone": False,
            "reverb": 0.0,
            "eq_bass_boost": -3,
            "eq_treble_boost": 2,
            "breathiness": 0.0,
            "comb_filter": False,
            "compressor": True,
            "exciter": 0.05,
        }
    },
    "robot": {
        "name": "机器人",
        "icon": "🤖",
        "description": "金属质感的机械音",
        "category": "basic",
        "params": {
            "pitch_shift": 0,
            "formant_ratio": 1.0,
            "ring_mod": 0.7,
            "ring_freq": 50,
            "telephone": False,
            "reverb": 0.25,
            "eq_bass_boost": 0,
            "eq_treble_boost": 3,
            "breathiness": 0.0,
            "comb_filter": False,
            "compressor": False,
            "exciter": 0.0,
        }
    },
    "minion": {
        "name": "小黄人",
        "icon": "🟡",
        "description": "高亢搞笑的小黄人音色",
        "category": "basic",
        "params": {
            "pitch_shift": 7,
            "formant_ratio": 0.65,
            "ring_mod": 0.0,
            "ring_freq": 0,
            "telephone": False,
            "reverb": 0.08,
            "eq_bass_boost": -5,
            "eq_treble_boost": 3,
            "breathiness": 0.0,
            "comb_filter": False,
            "compressor": True,
            "exciter": 0.08,
        }
    },
    "old_telephone": {
        "name": "老式电话",
        "icon": "📞",
        "description": "带底噪的复古电话音效",
        "category": "basic",
        "params": {
            "pitch_shift": 0,
            "formant_ratio": 1.0,
            "ring_mod": 0.0,
            "ring_freq": 0,
            "telephone": True,
            "reverb": 0.0,
            "eq_bass_boost": 0,
            "eq_treble_boost": 0,
            "breathiness": 0.0,
            "comb_filter": False,
            "compressor": False,
            "exciter": 0.0,
        }
    },
    "giant": {
        "name": "巨人",
        "icon": "👹",
        "description": "低沉威严的巨大生物",
        "category": "basic",
        "params": {
            "pitch_shift": -6,
            "formant_ratio": 1.5,
            "ring_mod": 0.0,
            "ring_freq": 0,
            "telephone": False,
            "reverb": 0.4,
            "eq_bass_boost": 8,
            "eq_treble_boost": -3,
            "breathiness": 0.0,
            "comb_filter": False,
            "compressor": True,
            "exciter": 0.05,
        }
    },
    "whisper": {
        "name": "耳语",
        "icon": "🤫",
        "description": "轻柔的气声低语",
        "category": "basic",
        "params": {
            "pitch_shift": 0,
            "formant_ratio": 1.0,
            "ring_mod": 0.0,
            "ring_freq": 0,
            "telephone": False,
            "reverb": 0.3,
            "eq_bass_boost": -6,
            "eq_treble_boost": 2,
            "breathiness": 0.5,
            "comb_filter": False,
            "compressor": False,
            "exciter": 0.0,
        }
    },
    "metallic": {
        "name": "金属音",
        "icon": "⚙️",
        "description": "冰冷的金属回响质感",
        "category": "basic",
        "params": {
            "pitch_shift": -2,
            "formant_ratio": 1.0,
            "ring_mod": 0.35,
            "ring_freq": 200,
            "telephone": False,
            "reverb": 0.5,
            "eq_bass_boost": 0,
            "eq_treble_boost": 4,
            "breathiness": 0.0,
            "comb_filter": True,
            "compressor": False,
            "exciter": 0.0,
        }
    },
}

# ==================== 进阶区：AI 模仿特定人物 ====================
AI_PRESETS = {
    "kobe": {
        "name": "科比",
        "icon": "🏀",
        "description": "科比·布莱恩特的标志性嗓音",
        "category": "ai",
        "model_path": "assets/models/kobe.pth",
    },
    "spongebob": {
        "name": "海绵宝宝",
        "icon": "🧽",
        "description": "海绵宝宝的经典尖细声",
        "category": "ai",
        "model_path": "assets/models/spongebob.pth",
    },
    "custom": {
        "name": "自定义",
        "icon": "📁",
        "description": "上传自定义 RVC 模型",
        "category": "ai",
        "model_path": None,
    },
}


def get_preset(name: str) -> dict:
    """获取指定预设的参数配置"""
    if name in DSP_PRESETS:
        return DSP_PRESETS[name]
    if name in AI_PRESETS:
        return AI_PRESETS[name]
    raise ValueError(f"未知预设: {name}，可用预设: {list(DSP_PRESETS.keys()) + list(AI_PRESETS.keys())}")


def list_presets(category: str = None) -> list:
    """列出所有预设，可按类别筛选"""
    if category == "basic":
        return [{"key": k, **v} for k, v in DSP_PRESETS.items()]
    if category == "ai":
        return [{"key": k, **v} for k, v in AI_PRESETS.items()]
    return [{"key": k, **v} for k, v in {**DSP_PRESETS, **AI_PRESETS}.items()]


def get_default_params() -> dict:
    """获取默认的 DSP 参数（无任何效果）"""
    return {
        "pitch_shift": 0,
        "formant_ratio": 1.0,
        "ring_mod": 0.0,
        "ring_freq": 0,
        "telephone": False,
        "reverb": 0.0,
        "eq_bass_boost": 0,
        "eq_treble_boost": 0,
        "breathiness": 0.0,
        "comb_filter": False,
        "compressor": False,
        "exciter": 0.0,
    }
