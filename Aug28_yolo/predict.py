"""
Predict OBBs for LFA stripes with YOLOv8 — labels drawn above the box, JSON saved.

Usage:
    python predict_obb_lfa.py
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from ultralytics import YOLO
from typing import List, Tuple

# -----------------------
# USER SETTINGS
# -----------------------
# Where to read images from (a single image path or a directory).
SOURCE = Path(r"raw\png")   # <-- CHANGE IF NEEDED
# Where to save visualized images + JSON outputs.
OUT_DIR = Path(r"pred")  # output folder (will be created)
# Trained weights (falls back to yolov8m-obb.pt if not found).
WEIGHTS = Path(r"runs_lfa_obb_wo_constrain/y8m_lfa2/weights/best.pt")

# Inference knobs
IMG_SIZE = 1536
CONF = 0.10
IOU = 0.50
DEVICE = 0   # GPU index; use -1 for CPU

# File types to scan when SOURCE is a directory
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# -----------------------
# Helpers
# -----------------------
def _color_for_class(cid: int) -> Tuple[int, int, int]:
    """Deterministic BGR color per class id."""
    rng = np.random.default_rng(seed=cid + 12345)
    return tuple(int(x) for x in rng.integers(60, 220, size=3))  # B, G, R


def _class_name(names, cid: int) -> str:
    """Safely fetch a class name from Ultralytics names (list or dict)."""
    if isinstance(names, dict):
        return names.get(int(cid), str(int(cid)))
    try:
        return names[int(cid)]
    except Exception:
        return str(int(cid))


def _draw_obb_with_label(img: np.ndarray, poly_xy: np.ndarray, label: str, color: Tuple[int, int, int]) -> None:
    """
    Draw a 4-point polygon and place label OUTSIDE & ABOVE the top-most edge.

    poly_xy: (4, 2) absolute pixel coords (x, y).
    """
    p = poly_xy.astype(int).reshape(-1, 2)

    # 1) draw polygon
    cv2.polylines(img, [p.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)

    # 2) position label above the top-most vertex; horizontally centered over the top-most points
    ys = p[:, 1]
    top_y = ys.min()
    top_mask = np.isclose(ys, top_y)
    top_points = p[top_mask]
    x_anchor = int(np.mean(top_points[:, 0])) if len(top_points) else int(np.mean(p[:, 0]))
    y_anchor = int(top_y - 6)  # a few px above

    h, w = img.shape[:2]
    y_anchor = max(0, y_anchor)

    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

    # center text horizontally; clamp to image bounds
    x_text = max(0, min(x_anchor - tw // 2, w - tw - 1))
    y_text = max(th + 2, y_anchor)

    # filled background for readability
    cv2.rectangle(
        img,
        (x_text - 2, y_text - th - 4),
        (x_text + tw + 2, y_text + baseline),
        color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    # black text over colored box
    cv2.putText(img, label, (x_text, y_text - 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _gather_images(src: Path) -> List[Path]:
    if src.is_file():
        return [src]
    files = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def _ensure_model(weights: Path) -> YOLO:
    if not weights.exists():
        print(f"[warn] {weights} not found; falling back to yolov8m-obb.pt")
        return YOLO("yolov8m-obb.pt")
    return YOLO(str(weights))


def _save_json(json_path: Path, image_name: str, w: int, h: int, detections: list) -> None:
    obj = {
        "image": image_name,
        "width": w,
        "height": h,
        "detections": detections
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# -----------------------
# Main prediction routine
# -----------------------
def main() -> None:
    src = SOURCE
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _ensure_model(WEIGHTS)

    img_paths = _gather_images(src)
    if not img_paths:
        raise FileNotFoundError(f"No images found under: {src}")

    pbar = tqdm(img_paths, desc="Predicting OBB", unit="img")
    for ip in pbar:
        # run model (we draw/save ourselves)
        results = model.predict(
            source=str(ip),
            imgsz=IMG_SIZE,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            save=False,
            verbose=False
        )

        res = results[0]
        names = res.names
        im_bgr = cv2.imread(str(ip))
        if im_bgr is None:
            print(f"[warn] Could not read image: {ip}")
            continue
        h, w = im_bgr.shape[:2]

        detections = []
        n_det = 0

        obb = getattr(res, "obb", None)
        if obb is not None and len(obb.data):
            # absolute pixel geometry
            polys = obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)   # (N,4,2)
            clss  = obb.cls.cpu().numpy().astype(int)              # (N,)
            confs = obb.conf.cpu().numpy()                         # (N,)
            xywhr = obb.xywhr.cpu().numpy()                        # (N,5) xc,yc,w,h,r(rad)

            n_det = len(polys)

            for i, poly in enumerate(polys):
                cid = int(clss[i])
                cname = _class_name(names, cid)
                c = float(confs[i])
                label = f"{cname} {c:.2f}"
                color = _color_for_class(cid)

                _draw_obb_with_label(im_bgr, poly, label, color)

                detections.append({
                    "class_id": cid,
                    "class_name": cname,
                    "confidence": c,
                    "polygon_xy": poly.tolist(),  # 4x2 absolute px
                    "xywhr": {
                        "xc": float(xywhr[i, 0]),
                        "yc": float(xywhr[i, 1]),
                        "w":  float(xywhr[i, 2]),
                        "h":  float(xywhr[i, 3]),
                        "radian": float(xywhr[i, 4]),
                        "degree": float(np.degrees(xywhr[i, 4]))
                    }
                })

        # write outputs
        out_img = out_dir / f"{ip.stem}_obb.jpg"
        out_json = out_dir / f"{ip.stem}_obb.json"
        cv2.imwrite(str(out_img), im_bgr)
        _save_json(out_json, ip.name, w, h, detections)

        pbar.set_postfix_str(f"det={n_det}")

    print(f"[done] Saved images + JSON to: {out_dir.resolve()}")


if __name__ == "__main__":
    # Windows-safe; harmless on other OSes
    try:
        import multiprocessing
        multiprocessing.freeze_support()
    except Exception:
        pass
    main()
