"""
声纹魔方 - RVC ONNX 模型下载脚本

从 Hugging Face 下载预训练 RVC ONNX 模型 (MIT License, ozada/onnx_rvc)。
下载内容:
  - vec-256-layer-9.onnx  (293 MB) — ContentVec 语义特征提取器
  - rmvpe.onnx            (362 MB) — RMVPE 基频提取器
  - woman_1.onnx          (111 MB) — 女声 RVC 音色模型 (演示用)

模型会放入 assets/models/ 目录。
"""

import os
import sys
import requests
from tqdm import tqdm

BASE_URL = "https://huggingface.co/ozada/onnx_rvc/resolve/main"
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "models")

REQUIRED_MODELS = {
    "vec-256-layer-9.onnx": {
        "url": f"{BASE_URL}/vec-256-layer-9.onnx",
        "size_mb": 293,
        "description": "ContentVec 语义特征提取器 (RVC v1, 256-dim)",
    },
    "vec-768-layer-12.onnx": {
        "url": f"{BASE_URL}/vec-768-layer-12.onnx",
        "size_mb": 360,
        "description": "ContentVec 语义特征提取器 (RVC v2, 768-dim)",
    },
    "rmvpe.onnx": {
        "url": f"{BASE_URL}/rmvpe.onnx",
        "size_mb": 362,
        "description": "RMVPE 基频提取器 (高精度 F0)",
    },
    "woman_1.onnx": {
        "url": f"{BASE_URL}/woman_1.onnx",
        "size_mb": 111,
        "description": "女声 RVC 音色模型 (演示用)",
    },
}

OPTIONAL_MODELS = {
    "sabrina.onnx": {
        "url": f"{BASE_URL}/sabrina.onnx",
        "size_mb": 111,
        "description": "Sabrina Carpenter 音色 (示例人物)",
    },
    "drake.onnx": {
        "url": f"{BASE_URL}/drake.onnx",
        "size_mb": 111,
        "description": "Drake 音色 (示例人物)",
    },
}

# HuggingFace 社区 PyTorch 模型 (RVC v2, .pth)
COMMUNITY_MODELS = {
    "spongebob.pth": {
        "url": "https://huggingface.co/abus/aiconverter/resolve/main/spongebob.pth",
        "size_mb": 54,
        "description": "海绵宝宝 RVC v2 音色 (PyTorch, 48kHz)",
    },
    "mambo.pth": {
        "url": "https://huggingface.co/juzi45/RVC_Matikanetannhauser/resolve/main/uma-Matikane_Tannhauser.pth",
        "size_mb": 55,
        "description": "曼波 RVC v2 音色 (赛马娘 Matikane Tannhauser, PyTorch, 48kHz)",
    },
}


def download_file(url: str, dest: str, desc: str = "") -> bool:
    """下载文件，显示进度条"""
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        session = requests.Session()
        session.trust_env = False  # 绕过 Windows 系统代理
        resp = session.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            desc=desc or os.path.basename(dest),
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [FAIL] {os.path.basename(dest)}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def main():
    print("=" * 60)
    print("  声纹魔方 - RVC ONNX 模型下载")
    print("=" * 60)
    print(f"  目标目录: {MODELS_DIR}")
    print(f"  来源: ozada/onnx_rvc (MIT License)")
    print("=" * 60)

    # 检查现有文件
    existing = {f for f in os.listdir(MODELS_DIR)} if os.path.exists(MODELS_DIR) else set()
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Step 1: 下载必需模型
    print("\n[Step 1/2] 下载必需模型 (共 3 个, ~766 MB)")
    print("-" * 40)
    all_ok = True
    for name, info in REQUIRED_MODELS.items():
        dest = os.path.join(MODELS_DIR, name)
        if name in existing:
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"  [SKIP] {name} ({size_mb:.0f} MB) — 已存在")
            continue
        print(f"  [下载] {name} ({info['size_mb']} MB) — {info['description']}")
        ok = download_file(info["url"], dest, desc=name)
        if not ok:
            all_ok = False
        existing.add(name)

    # Step 2: 可选模型
    print("\n[Step 2/3] 可选模型 (按需下载)")
    print("-" * 40)
    for name, info in OPTIONAL_MODELS.items():
        dest = os.path.join(MODELS_DIR, name)
        if name in existing:
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"  [SKIP] {name} ({size_mb:.0f} MB) — 已存在")
            continue
        print(f"  [选项] {name} ({info['size_mb']} MB) — {info['description']}")
        # 可选模型不自动下载，提示用户

    # Step 3: 社区 PyTorch 模型
    print("\n[Step 3/3] 社区 RVC 模型 (HuggingFace)")
    print("-" * 40)
    for name, info in COMMUNITY_MODELS.items():
        dest = os.path.join(MODELS_DIR, name)
        if name in existing:
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"  [SKIP] {name} ({size_mb:.0f} MB) — 已存在")
            continue
        print(f"  [下载] {name} ({info['size_mb']} MB) — {info['description']}")
        ok = download_file(info["url"], dest, desc=name)
        if not ok:
            all_ok = False
        existing.add(name)

    print("\n" + "=" * 60)
    if all_ok:
        print("  模型下载完成! 可用人物:")
        print("    基础区 (DSP 模拟):")
        print("      deep_male, child, robot, minion, giant, whisper, metallic")
        print("      lazy_goat, bear_two, belial, mambo (卡通角色)")
        print("    进阶区 (AI 推理):")
        print("      woman_1 — 通用女声 (ONNX)")
        print("      spongebob — 海绵宝宝 (PyTorch)")
        print("      mambo — 曼波/赛马娘 (PyTorch)")
        print("      kobe — 科比 (DSP 模拟)")
        print("    待训练:")
        print("      lazy_goat_ai, bear_two_ai, belial_ai")
        print("      (可从 klrvc.com 获取或自行训练 RVC 模型)")
    else:
        print("  部分模型下载失败, 请重试")
    print("=" * 60)


if __name__ == "__main__":
    main()
