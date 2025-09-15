import os
from pathlib import Path

# 跟你的 main.py 同目录
cache_files = Path("yoloobb_3class").rglob("*.cache")
for f in cache_files:
    print(f"Removing cache: {f}")
    os.remove(f)
