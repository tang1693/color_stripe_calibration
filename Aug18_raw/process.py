from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import csv

import cv2
import numpy as np
from tqdm import tqdm

CFG: Dict[str, Any] = {
    "MARGIN": 25,          # R 必须分别比 G 和 B 大多少
    "ALPHA": 0.35,         # 叠加透明度 (0..1)
    "SAVE_MASK": True,     # 是否另存二值掩码
    "MASK_SUFFIX": "_mask.png",
    "CSV_NAME": "summary.csv",
}

def read_image_keep_alpha(p: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """读图并保留 alpha。返回(BGR, alpha or None)。"""
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read: {p}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img, None
    if img.shape[2] == 4:
        return img[..., :3], img[..., 3]
    return img, None

def save_with_optional_alpha(out_bgr: np.ndarray, alpha: Optional[np.ndarray], out_path: Path):
    if alpha is not None:
        rgba = np.dstack([out_bgr, alpha])
        cv2.imwrite(str(out_path), rgba)
    else:
        cv2.imwrite(str(out_path), out_bgr)

def highlight_and_measure(bgr: np.ndarray, margin: int, alpha_overlay: float):
    """
    高亮红色主导像素，并计算灰度均值/总和（仅在高亮区域内）。
    返回 (overlay_bgr, mask_u8, count, min_gray, sum_gray)
    """
    # 分通道并用 int16 防止减法溢出
    b = bgr[..., 0].astype(np.int16)
    g = bgr[..., 1].astype(np.int16)
    r = bgr[..., 2].astype(np.int16)

    mask = ((r - g) > margin) & ((r - b) > margin)
    mask_u8 = (mask.astype(np.uint8) * 255)

    # 半透明红色叠加
    vis = bgr.astype(np.float32).copy()
    overlay = np.zeros_like(vis, dtype=np.float32)
    overlay[..., 2] = 255.0  # 纯红(BGR)
    m3 = mask[..., None].astype(np.float32)
    vis = np.where(m3, (1 - alpha_overlay) * vis + alpha_overlay * overlay, vis)
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    # 灰度（仅在高亮区域）
    # OpenCV是BGR：Y = 0.114*B + 0.587*G + 0.299*R
    gray = (0.05 * b.astype(np.float32) +
            0.05 * g.astype(np.float32) +
            0.9 * r.astype(np.float32))

    if mask.any():
        gray_in = gray[mask]
        min_gray = float(gray_in.min())
        sum_gray = float(gray_in.sum())
        count = int(gray_in.size)
    else:
        min_gray = float("nan")
        sum_gray = 0.0
        count = 0

    return vis, mask_u8, count, min_gray, sum_gray

def process_one(in_path: Path, out_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    bgr, a = read_image_keep_alpha(in_path)
    vis, mask, cnt, min_gray, sum_gray = highlight_and_measure(
        bgr, cfg["MARGIN"], cfg["ALPHA"]
    )

    out_img = out_dir / in_path.name
    save_with_optional_alpha(vis, a, out_img)

    if cfg.get("SAVE_MASK", False):
        cv2.imwrite(str(out_dir / (in_path.stem + cfg["MASK_SUFFIX"])), mask)

    h, w = bgr.shape[:2]
    return {
        "file": in_path.name,
        "H": h,
        "W": w,
        "highlighted_pixels": cnt,
        "min_gray_in_highlight": min_gray,
        "sum_gray_in_highlight": sum_gray,
    }

def main(input_dir: str, output_dir: str, cfg: Dict[str, Any] = None):
    cfg = {**CFG, **(cfg or {})}
    in_dir, out_dir = Path(input_dir), Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg"}
    imgs = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not imgs:
        print(f"No images found in {in_dir}")
        return

    rows: List[Dict[str, Any]] = []
    for p in tqdm(imgs, desc="Highlight + grayscale stats"):
        rows.append(process_one(p, out_dir, cfg))

    # 写 CSV
    csv_path = out_dir / cfg["CSV_NAME"]
    fields = ["file", "H", "W", "highlighted_pixels",
              "min_gray_in_highlight", "sum_gray_in_highlight"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    total = sum(r["highlighted_pixels"] for r in rows)
    print(f"Saved overlays to: {out_dir}")
    print(f"Saved CSV to: {csv_path}")
    print(f"Total highlighted pixels across images: {total}")

if __name__ == "__main__":
    main("cockroach", "cockroach_output", cfg={"MARGIN": 25, "ALPHA": 0.35, "SAVE_MASK": True})
    main("der_f", "der_f_output", cfg={"MARGIN": 25, "ALPHA": 0.35, "SAVE_MASK": True})
    main("der_p", "der_p_output", cfg={"MARGIN": 25, "ALPHA": 0.35, "SAVE_MASK": True})
