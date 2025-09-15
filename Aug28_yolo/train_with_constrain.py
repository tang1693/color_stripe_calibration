# Fine-tune YOLOv8 OBB for LFA stripes — all in Python, no CLI.
# Prereqs (one-time): pip install ultralytics==8.3.0 opencv-python numpy

import os, random, shutil, math
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

# -----------------------
# USER SETTINGS
# -----------------------
# Point this to your YOLOv8 OBB export directory from Label Studio.
# That export should contain images/ and labels/ with .txt files in OBB format.
DATASET_DIR = Path(r"yoloobb_3class")   # <-- CHANGE THIS
NAMES = ["Control", "Stripe", "Test"]   # your three classes (index order matters)
IMG_SIZE = 1536                         # 1280–1536 is good for thin stripes
EPOCHS = 200                            # small dataset → more epochs
BATCH = 4                               # adjust to your GPU
DEVICE = 0                              # -1 for CPU

# -----------------------
# LFA CONSTRAINTS (tunable)
# -----------------------
class Cfg:
    # Angle: align Control/Test to Stripe within tolerance (deg). We'll snap to Stripe angle in post.
    ANGLE_TOL_DEG = 12.0

    # Aspect ratio for Control/Test (H:W ≈ 1:3) → longer/shorter ≈ 3.0
    CT_TARGET_AR = 3.0
    CT_AR_TOL = 0.6  # allow ±0.6 around 3.0

    # Control & Test should be same size (within tolerance)
    SAME_SIZE_TOL = 0.25  # relative (e.g., 0.25 means ±25%)

    # Ensure CT lie inside Stripe polygon (allow small margin)
    INSIDE_MARGIN = 1.02  # >1 expands stripe very slightly to be lenient

    # When adjusting to fit inside stripe, shrink until it fits
    FIT_SHRINK_STEP = 0.98
    FIT_MAX_STEPS = 20

    # Stripe elongation: require very elongated stripe (long/short >= 10)
    STRIPE_MIN_LONG_SHORT_AR = 10.0

    # Selectors
    REQUIRE_CT_INSIDE_STRIPE = True

CFG = Cfg()

# -----------------------
# Helpers: dataset prep
# -----------------------
def has_split(root: Path) -> bool:
    return (root/"images/train").exists() and (root/"images/val").exists() \
        and (root/"labels/train").exists() and (root/"labels/val").exists()

def make_split_if_missing(root: Path, val_ratio=0.15):
    """
    If dataset is flat:
      images/*.jpg|png
      labels/*.txt   (YOLOv8 OBB format: cls x1 y1 x2 y2 x3 y3 x4 y4 [0-1 normalized])
    Create images/train,val and labels/train,val with a simple split.
    """
    img_dir = root/"images"
    lab_dir = root/"labels"
    if has_split(root):
        print("[info] Train/val split already present.")
        return

    images = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff"):
        images.extend(list(img_dir.glob(ext)))
    images = sorted(images)

    random.seed(0)
    random.shuffle(images)
    k = max(1, int(len(images) * val_ratio))
    val_imgs = set(images[:k])
    train_imgs = set(images[k:])

    for split in ("train","val"):
        (root/f"images/{split}").mkdir(parents=True, exist_ok=True)
        (root/f"labels/{split}").mkdir(parents=True, exist_ok=True)

    def copy_one(img_path: Path, split: str):
        dst_img = root/f"images/{split}/{img_path.name}"
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        cand = lab_dir/(img_path.stem + ".txt")
        if cand.exists():
            dst_lab = root/f"labels/{split}/{cand.name}"
            if not dst_lab.exists():
                shutil.copy2(cand, dst_lab)

    for p in train_imgs: copy_one(p, "train")
    for p in val_imgs:   copy_one(p, "val")
    print(f"[info] Created split: train={len(train_imgs)} images, val={len(val_imgs)} images")

def write_yaml(root: Path, names):
    names_list = "[" + ", ".join(names) + "]"
    yaml_text = f"""# Auto-generated for YOLOv8-OBB
path: {root.as_posix()}
train: images/train
val: images/val
test: images/val
names: {names_list}
"""
    yaml_path = root/"lfa_obb.yaml"
    yaml_path.write_text(yaml_text)
    return yaml_path

def clean_ultra_caches(root: Path):
    """Remove stale Ultralytics *.cache files to avoid numpy._core pickle errors."""
    removed = 0
    for f in root.rglob("*.cache"):
        try:
            os.remove(f)
            removed += 1
        except OSError:
            pass
    if removed:
        print(f"[info] Removed {removed} stale cache file(s) that could break numpy pickle loading.")

# -----------------------
# Helpers: geometry (OBB)
# -----------------------
def angle_norm_deg(a):
    # normalize to [0, 180)
    a = a % 180.0
    if a < 0: a += 180.0
    return a

def angle_diff_deg(a, b):
    # small unsigned difference modulo 180
    d = abs(angle_norm_deg(a) - angle_norm_deg(b))
    return min(d, 180.0 - d)

def rect_xywhr_to_poly(cx, cy, w, h, angle_deg):
    """Return 4x2 polygon for an oriented rectangle (OpenCV-style angle relative to +x axis)."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx, dy = w/2, h/2
    pts = np.array([[-dx,-dy],[ dx,-dy],[ dx, dy],[-dx, dy]], dtype=np.float32)
    R = np.array([[ca,-sa],[sa, ca]], dtype=np.float32)
    rot = pts @ R.T
    rot[:,0] += cx
    rot[:,1] += cy
    return rot

def poly_edge_lengths(poly4):
    p = np.array(poly4, dtype=np.float32)
    L = []
    for i in range(4):
        j = (i+1) % 4
        L.append(np.linalg.norm(p[j]-p[i]))
    return L

def poly_angle_long_axis(poly4):
    p = np.array(poly4, dtype=np.float32)
    v01 = p[1]-p[0]
    v12 = p[2]-p[1]
    l01 = np.linalg.norm(v01)
    l12 = np.linalg.norm(v12)
    v = v01 if l01 >= l12 else v12
    ang = math.degrees(math.atan2(v[1], v[0]))
    return angle_norm_deg(ang)

def poly_wh(poly4):
    """Return (w,h) where w = longer side length, h = shorter side length."""
    L = poly_edge_lengths(poly4)
    w = max(L[0], L[1])
    h = min(L[0], L[1])
    return float(w), float(h)

def point_in_convex_quad(pt, quad):
    p = np.array(pt, dtype=np.float32)
    q = np.array(quad, dtype=np.float32)
    signs = []
    for i in range(4):
        a = q[i]; b = q[(i+1)%4]
        edge = b - a
        vp = p - a
        cross = edge[0]*vp[1] - edge[1]*vp[0]
        signs.append(cross)
    eps = 1e-5
    pos = sum(s >  eps for s in signs)
    neg = sum(s < -eps for s in signs)
    return (pos == 4) or (neg == 4)

def poly_contains_poly(inner4, outer4, margin_scale=1.0):
    outer = np.array(outer4, dtype=np.float32)
    c = outer.mean(axis=0)
    scaled = (outer - c) * margin_scale + c
    for k in range(4):
        if not point_in_convex_quad(inner4[k], scaled):
            return False
    return True

def resize_rect_keep_center(poly4, scale_w=1.0, scale_h=1.0):
    p = np.array(poly4, dtype=np.float32)
    c = p.mean(axis=0)
    ang = poly_angle_long_axis(p)
    w, h = poly_wh(p)
    new_w = w * scale_w
    new_h = h * scale_h
    return rect_xywhr_to_poly(c[0], c[1], new_w, new_h, ang)

# -----------------------
# Helpers: Ultralytics result parsing
# -----------------------
def extract_obb_dets(result, names):
    """
    Return list of dicts: {cls, conf, poly(4x2), w, h, ang}
    Works for YOLOv8 OBB results (xywhr or xyxyxyxy).
    """
    dets = []
    obb = getattr(result, "obb", None)
    if obb is None:
        return dets

    cls = obb.cls.detach().cpu().numpy() if hasattr(obb, "cls") else None
    conf = obb.conf.detach().cpu().numpy() if hasattr(obb, "conf") else None

    polys = None
    if hasattr(obb, "xywhr") and obb.xywhr is not None:
        arr = obb.xywhr.detach().cpu().numpy()  # [N,5]
        polys = []
        for cx, cy, w, h, a in arr:
            poly = rect_xywhr_to_poly(float(cx), float(cy), float(w), float(h), math.degrees(float(a)))
            polys.append(poly)
    elif hasattr(obb, "xyxyxyxy") and obb.xyxyxyxy is not None:
        arr = obb.xyxyxyxy.detach().cpu().numpy()  # [N,8]
        polys = [arr[i].reshape(4,2) for i in range(arr.shape[0])]

    if polys is None:
        return dets

    for i, poly in enumerate(polys):
        w, h = poly_wh(poly)
        ang = poly_angle_long_axis(poly)
        dets.append(dict(
            cls=int(cls[i]) if cls is not None else -1,
            conf=float(conf[i]) if conf is not None else 0.0,
            poly=poly.astype(np.float32),
            w=float(w), h=float(h), ang=float(ang)
        ))
    return dets

# -----------------------
# Constraint solver
# -----------------------
def select_stripe(dets, stripe_idx, cfg: Cfg):
    """Pick best Stripe that satisfies elongation (long/short >= threshold)."""
    candidates = []
    for d in dets:
        if d["cls"] != stripe_idx:
            continue
        ar_ls = d["w"] / max(d["h"], 1e-6)  # longer/shorter >= 1
        if ar_ls >= cfg.STRIPE_MIN_LONG_SHORT_AR:
            candidates.append(d)
    if not candidates:
        return None
    candidates.sort(key=lambda d: (d["conf"], d["w"]*d["h"]), reverse=True)
    return candidates[0]

def filter_ct_by_stripe(cands, stripe, want_class_idx, cfg: Cfg):
    out = []
    if stripe is None:
        return out
    sang = stripe["ang"]
    for d in cands:
        # angle close to stripe
        if angle_diff_deg(d["ang"], sang) > cfg.ANGLE_TOL_DEG:
            continue
        # aspect ratio near 3.0 (long/short)
        w, h = d["w"], d["h"]
        ar = w / max(h, 1e-6)  # long/short
        if abs(ar - cfg.CT_TARGET_AR) > cfg.CT_AR_TOL:
            continue
        # inside stripe?
        if cfg.REQUIRE_CT_INSIDE_STRIPE:
            if not poly_contains_poly(d["poly"], stripe["poly"], margin_scale=cfg.INSIDE_MARGIN):
                continue
        out.append(d)
    out.sort(key=lambda x: x["conf"], reverse=True)
    return out

def enforce_same_size_and_angle(c, t, stripe_ang, cfg: Cfg):
    """Snap Control/Test to same angle, same size, keep centers, and maintain 1:3 AR (long/short)."""
    if c is None and t is None:
        return c, t

    def snap_ar(det):
        if det is None: return None
        w, h = det["w"], det["h"]      # w >= h by construction
        target_w = CFG.CT_TARGET_AR * h
        cx, cy = det["poly"].mean(axis=0)
        new_poly = rect_xywhr_to_poly(float(cx), float(cy), float(target_w), float(h), stripe_ang)
        return dict(det, poly=new_poly, w=float(target_w), h=float(h), ang=float(stripe_ang))

    c2 = snap_ar(c)
    t2 = snap_ar(t)

    # decide common size (average)
    sizes = []
    if c2 is not None: sizes.append((c2["w"], c2["h"]))
    if t2 is not None: sizes.append((t2["w"], t2["h"]))
    if not sizes:
        return c2, t2
    avg_w = float(np.mean([s[0] for s in sizes]))
    avg_h = float(np.mean([s[1] for s in sizes]))

    def resize_to(det):
        if det is None: return None
        cx, cy = det["poly"].mean(axis=0)
        poly = rect_xywhr_to_poly(float(cx), float(cy), avg_w, avg_h, stripe_ang)
        return dict(det, poly=poly, w=avg_w, h=avg_h, ang=float(stripe_ang))

    c3 = resize_to(c2)
    t3 = resize_to(t2)
    return c3, t3

def shrink_to_fit(inner, stripe, cfg: Cfg):
    """Shrink inner box uniformly in W/H until fully inside stripe."""
    if inner is None or stripe is None: return inner
    poly = inner["poly"].copy()
    ok = poly_contains_poly(poly, stripe["poly"], margin_scale=cfg.INSIDE_MARGIN)
    steps = 0
    w, h = inner["w"], inner["h"]
    cx, cy = poly.mean(axis=0)
    while not ok and steps < cfg.FIT_MAX_STEPS:
        w *= cfg.FIT_SHRINK_STEP
        h *= cfg.FIT_SHRINK_STEP
        poly = rect_xywhr_to_poly(float(cx), float(cy), w, h, stripe["ang"])
        ok = poly_contains_poly(poly, stripe["poly"], margin_scale=cfg.INSIDE_MARGIN)
        steps += 1
    return dict(inner, poly=poly, w=w, h=h, ang=stripe["ang"])

def refine_lfa_predictions(dets, idx_control, idx_stripe, idx_test, cfg: Cfg):
    """
    dets: all predicted OBB detections in an image (dicts)
    Returns refined list [Stripe?, Control?, Test?] (some may be None).
    """
    stripe = select_stripe(dets, idx_stripe, cfg)

    ctrl_cands = [d for d in dets if d["cls"] == idx_control]
    test_cands = [d for d in dets if d["cls"] == idx_test]

    ctrl_valid = filter_ct_by_stripe(ctrl_cands, stripe, idx_control, cfg)
    test_valid = filter_ct_by_stripe(test_cands, stripe, idx_test, cfg)

    ctrl = ctrl_valid[0] if ctrl_valid else None
    test = test_valid[0] if test_valid else None

    if stripe is not None:
        ctrl, test = enforce_same_size_and_angle(ctrl, test, stripe["ang"], cfg)
        ctrl = shrink_to_fit(ctrl, stripe, cfg)
        test = shrink_to_fit(test, stripe, cfg)

    return stripe, ctrl, test

# -----------------------
# Visualization (labels above & outside bbox)
# -----------------------
def draw_poly(img, poly, color, thickness=2):
    p = np.array(poly, dtype=np.int32).reshape(-1,1,2)
    cv2.polylines(img, [p], isClosed=True, color=color, thickness=thickness)

def top_edge_center(poly):
    """Approximate top-edge center by averaging the two vertices with smallest y."""
    p = np.array(poly, dtype=np.float32)
    idx = np.argsort(p[:,1])[:2]  # two smallest y (topmost in image coords)
    top2 = p[idx]
    c = top2.mean(axis=0)
    return float(c[0]), float(c[1])

def put_label_outside(img, text, poly, color, scale=0.6, thickness=1, pad=6, leader=True):
    """Place label above the OBB, outside the box, centered on its top edge."""
    H, W = img.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    tx, ty = top_edge_center(poly)

    # Anchor above the top edge center
    y_text_bottom = max(th + pad, int(ty) - pad)  # ensure visible
    x_text_left = int(tx - tw/2)

    # Clamp horizontally
    x_text_left = max(2, min(W - tw - 2, x_text_left))
    y_text_bottom = max(th + 2, min(H - 2, y_text_bottom))

    # Optional: leader line from top edge center to label bottom-center
    if leader:
        p1 = (int(tx), int(ty))
        p2 = (int(x_text_left + tw/2), int(y_text_bottom + 2))
        cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)

    # Text background (small white box for readability, outside the bbox)
    bg_tl = (x_text_left - 3, y_text_bottom - th - 3)
    bg_br = (x_text_left + tw + 3, y_text_bottom + 3)
    cv2.rectangle(img, bg_tl, bg_br, (255,255,255), thickness=-1)
    cv2.putText(img, text, (x_text_left, y_text_bottom),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def visualize_refined(image_bgr, refined, names):
    out = image_bgr.copy()
    stripe, ctrl, test = refined
    if stripe is not None:
        draw_poly(out, stripe["poly"], (0,255,255), 3)  # yellow
        put_label_outside(out, names[stripe["cls"]], stripe["poly"], (0,160,160), 0.7, 2)
    if ctrl is not None:
        draw_poly(out, ctrl["poly"], (0,255,0), 3)  # green
        put_label_outside(out, names[ctrl["cls"]], ctrl["poly"], (0,140,0), 0.7, 2)
    if test is not None:
        draw_poly(out, test["poly"], (255,0,0), 3)  # blue
        put_label_outside(out, names[test["cls"]], test["poly"], (140,0,0), 0.7, 2)
    return out

# -----------------------
# Run
# -----------------------
def main():
    global DATASET_DIR
    DATASET_DIR = DATASET_DIR.resolve()
    assert (DATASET_DIR/"images").exists() and (DATASET_DIR/"labels").exists(), \
        f"Expected images/ and labels/ under {DATASET_DIR}"

    # kill stale caches (fixes: ModuleNotFoundError: numpy._core)
    clean_ultra_caches(DATASET_DIR)

    make_split_if_missing(DATASET_DIR)
    yaml_path = write_yaml(DATASET_DIR, NAMES)
    print(f"[info] Wrote dataset yaml → {yaml_path}")

    # Train YOLOv8 OBB
    model = YOLO("yolov8m-obb.pt")
    _ = model.train(
        data=str(yaml_path),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        project="runs_lfa_obb",
        name="y8m_lfa",
        optimizer="auto",
        lr0=0.0015,
        cos_lr=True,
        patience=70,
        amp=True,
        mosaic=0.30,         # keep structure
        mixup=0.0,
        copy_paste=0.0,
        degrees=5.0,
        translate=0.04,
        scale=0.08,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.02, hsv_s=0.30, hsv_v=0.30,
        fliplr=0.5, flipud=0.0,
        pretrained=True,
        seed=0,
        workers=0,           # Windows-safe
    )

    # Validate
    model.val(data=str(yaml_path), imgsz=IMG_SIZE, device=DEVICE)

    # -------- Prediction + Constraint Refinement on val --------
    pred_dir = DATASET_DIR/"images/val"
    raw_results = model.predict(
        source=str(pred_dir),
        imgsz=IMG_SIZE,
        conf=0.10,       # adjust later if you want fewer FPs
        iou=0.5,
        device=DEVICE,
        save=False,      # we'll save our own refined visualizations
        verbose=False
    )

    save_dir = Path("runs_lfa_obb/preds/val_vis_refined")
    save_dir.mkdir(parents=True, exist_ok=True)

    # class indices
    name_to_idx = {n.lower(): i for i, n in enumerate(NAMES)}
    idx_control = name_to_idx.get("control", 0)
    idx_stripe  = name_to_idx.get("stripe", 1)
    idx_test    = name_to_idx.get("test", 2)

    images_seen = 0
    have_stripe = have_ctrl = have_test = 0

    for r in raw_results:
        dets = extract_obb_dets(r, NAMES)
        stripe, ctrl, test = refine_lfa_predictions(dets, idx_control, idx_stripe, idx_test, CFG)
        img = r.orig_img if hasattr(r, "orig_img") else cv2.imread(str(r.path))
        vis = visualize_refined(img, (stripe, ctrl, test), NAMES)
        out_path = save_dir / Path(r.path).name
        cv2.imwrite(str(out_path), vis)

        images_seen += 1
        if stripe is not None: have_stripe += 1
        if ctrl   is not None: have_ctrl   += 1
        if test   is not None: have_test   += 1

    print(f"[info] Refined predictions saved under: {save_dir}")
    if images_seen:
        print(f"[stats] images={images_seen} | stripe={have_stripe} | control={have_ctrl} | test={have_test}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
