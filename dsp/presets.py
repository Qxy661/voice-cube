"""
声纹魔方 - 模仿预设参数库 (v4)
更自然的参数：减少过度处理，保留人声质感
"""

# ==================== 基础区：DSP 模仿经典音色 ====================
DSP_PRESETS = {
    "deep_male": {
        "name": "低音炮男声",
        "icon": "🎤",
        "description": "浑厚磁性的成熟男声",
        "category": "basic",
        "params": {
            "pitch_shift": -2,
            "formant_ratio": 0.92,      # 压低共振峰 → 大体型感 → 深沉
            "ring_mod": 0.0,
            "ring_freq": 0,
            "telephone": False,
            "reverb": 0.05,
            "eq_bass_boost": 3,
            "eq_treble_boost": -1,
            "breathiness": 0.0,
            "comb_filter": False,
        }
    },
    "child": {
        "name": "童声",
        "icon": "👶",
        "description": "天真可爱的儿童声音",
        "category": "basic",
        "params": {
            "pitch_shift": 4,
            "formant_ratio": 1.22,       # 抬高共振峰 → 小体型感 → 明亮
            "ring_mod": 0.0,
            "ring_freq": 0,
            "telephone": False,
            "reverb": 0.0,
            "eq_bass_boost": -2,
            "eq_treble_boost": 1,
            "breathiness": 0.0,
            "comb_filter": False,
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
            "reverb": 0.15,
            "eq_bass_boost": 0,
            "eq_treble_boost": 2,
            "breathiness": 0.0,
            "comb_filter": False,
        }
    },
    "minion": {
        "name": "小黄人",
        "icon": "🟡",
        "description": "高亢搞笑的小黄人音色",
        "category": "basic",
        "params": {
            "pitch_shift": 6,
            "formant_ratio": 1.40,       # 大幅抬高共振峰 → 小体型 → 高亢
            "ring_mod": 0.0,
            "ring_freq": 0,
            "telephone": False,
            "reverb": 0.05,
            "eq_bass_boost": -4,
            "eq_treble_boost": 2,
            "breathiness": 0.0,
            "comb_filter": False,
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
        }
    },
    "giant": {
        "name": "巨人",
        "icon": "👹",
        "description": "低沉威严的巨大生物",
        "category": "basic",
        "params": {
            "pitch_shift": -5,
            "formant_ratio": 0.70,       # 压低共振峰 → 大体型 → 深沉威严
            "ring_mod": 0.0,
            "ring_freq": 0,
            "telephone": False,
            "reverb": 0.3,
            "eq_bass_boost": 6,
            "eq_treble_boost": -2,
            "breathiness": 0.0,
            "comb_filter": False,
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
            "reverb": 0.2,
            "eq_bass_boost": -4,
            "eq_treble_boost": 1,
            "breathiness": 0.5,
            "comb_filter": False,
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
            "reverb": 0.3,
            "eq_bass_boost": 0,
            "eq_treble_boost": 3,
            "breathiness": 0.0,
            "comb_filter": True,
        }
    },
    # ── 卡通角色 DSP 模拟 ──
    "lazy_goat": {
        "name": "懒羊羊",
        "icon": "🐑",
        "description": "懒羊羊的懒散可爱声线 (DSP 模拟)",
        "category": "basic",
        "params": {
            "pitch_shift": 3,
            "formant_ratio": 1.25,      # 抬高共振峰 → 小体型 → 可爱
            "ring_mod": 0.0,
            "ring_freq": 0,
            "telephone": False,
            "reverb": 0.08,
            "eq_bass_boost": -2,
            "eq_treble_boost": 2,
            "breathiness": 0.15,        # 轻微气声 → 懒散感
            "comb_filter": False,
        }
    },
    "bear_two": {
        "name": "熊二",
        "icon": "🐻",
        "description": "熊二的憨厚低沉声线 (DSP 模拟)",
        "category": "basic",
        "params": {
            "pitch_shift": -4,
            "formant_ratio": 0.85,      # 压低共振峰 → 大体型 → 憨厚
            "ring_mod": 0.0,
            "ring_freq": 0,
            "telephone": False,
            "reverb": 0.15,
            "eq_bass_boost": 5,
            "eq_treble_boost": -2,
            "breathiness": 0.0,
            "comb_filter": False,
        }
    },
    "belial": {
        "name": "贝利亚",
        "icon": "👹",
        "description": "贝利亚的威严邪恶声线 (DSP 模拟)",
        "category": "basic",
        "params": {
            "pitch_shift": -6,
            "formant_ratio": 0.78,      # 大幅压低共振峰 → 巨大体型 → 威严
            "ring_mod": 0.2,
            "ring_freq": 80,
            "telephone": False,
            "reverb": 0.35,
            "eq_bass_boost": 6,
            "eq_treble_boost": 1,
            "breathiness": 0.0,
            "comb_filter": False,
        }
    },
    "mambo": {
        "name": "曼波",
        "icon": "🐧",
        "description": "曼波的活泼尖细声线 (DSP 模拟)",
        "category": "basic",
        "params": {
            "pitch_shift": 5,
            "formant_ratio": 1.35,      # 高共振峰 → 小体型 → 活泼
            "ring_mod": 0.0,
            "ring_freq": 0,
            "telephone": False,
            "reverb": 0.05,
            "eq_bass_boost": -3,
            "eq_treble_boost": 3,
            "breathiness": 0.0,
            "comb_filter": False,
        }
    },
}

# ==================== 进阶区：AI 模仿特定人物 ====================
AI_PRESETS = {
    "woman_1": {
        "name": "通用女声",
        "icon": "👩",
        "description": "通用女声音色 (RVC ONNX 推理)",
        "category": "ai",
        "model_path": "assets/models/woman_1.onnx",
    },
    "kobe": {
        "name": "科比",
        "icon": "🏀",
        "description": "科比·布莱恩特的标志性嗓音 (DSP 模拟)",
        "category": "ai",
        "model_path": "assets/models/kobe.pth",
    },
    "spongebob": {
        "name": "海绵宝宝",
        "icon": "🧽",
        "description": "海绵宝宝的经典尖细声 (RVC PyTorch 推理)",
        "category": "ai",
        "model_path": "assets/models/spongebob.pth",
    },
    "lazy_goat_ai": {
        "name": "懒羊羊(AI)",
        "icon": "🐑",
        "description": "懒羊羊的可爱声线 (DSP 模拟, 待训练 RVC 模型)",
        "category": "ai",
        "model_path": None,
    },
    "bear_two_ai": {
        "name": "熊二(AI)",
        "icon": "🐻",
        "description": "熊二的憨厚声线 (DSP 模拟, 待训练 RVC 模型)",
        "category": "ai",
        "model_path": None,
    },
    "belial_ai": {
        "name": "贝利亚(AI)",
        "icon": "👹",
        "description": "贝利亚的威严声线 (DSP 模拟, 待训练 RVC 模型)",
        "category": "ai",
        "model_path": None,
    },
    "mambo_ai": {
        "name": "曼波(AI)",
        "icon": "🐧",
        "description": "曼波的活泼声线 (DSP 模拟, 待训练 RVC 模型)",
        "category": "ai",
        "model_path": None,
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
    }
