# Fine-tune YOLOv8 OBB for LFA stripes — all in Python, no CLI.
# Prereqs (one-time): pip install ultralytics==8.3.0

import os, random, shutil
from pathlib import Path
from torch import mps
from ultralytics import YOLO

# -----------------------
# USER SETTINGS
# -----------------------
# Point this to your YOLOv8 OBB export directory from Label Studio.
# That export should contain images/ and labels/ with .txt files in OBB format.
DATASET_DIR = Path(r"YOLOOBB")   # <-- CHANGE THIS
NAMES = ["Stripe"]             # your three classes
IMG_SIZE = 1536                                   # 1280–1536 is good for thin stripes
EPOCHS = 200                                      # small dataset → more epochs
BATCH = 4                                         # adjust to your GPU
DEVICE = 'mps'                                        # -1 for CPU

# -----------------------
# Helpers
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

    # gather image files that have a matching label file (or negatives without labels)
    images = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff"):
        images.extend(list(img_dir.glob(ext)))
    images = sorted(images)

    # Shuffle & split
    random.seed(0)
    random.shuffle(images)
    k = max(1, int(len(images) * val_ratio))
    val_imgs = set(images[:k])
    train_imgs = set(images[k:])

    # prepare dirs
    for split in ("train","val"):
        (root/f"images/{split}").mkdir(parents=True, exist_ok=True)
        (root/f"labels/{split}").mkdir(parents=True, exist_ok=True)

    # distribute
    def move_one(img_path: Path, split: str):
        # move/copy image
        dst_img = root/f"images/{split}/{img_path.name}"
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        # move/copy label if exists (same stem)
        cand = lab_dir/(img_path.stem + ".txt")
        if cand.exists():
            dst_lab = root/f"labels/{split}/{cand.name}"
            if not dst_lab.exists():
                shutil.copy2(cand, dst_lab)

    for p in train_imgs: move_one(p, "train")
    for p in val_imgs:   move_one(p, "val")
    print(f"[info] Created split: train={len(train_imgs)} images, val={len(val_imgs)} images")

def write_yaml(root: Path, names):
    # 写成列表形式：names: [Stripe]
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


# -----------------------
# Prepare dataset
# -----------------------
DATASET_DIR = DATASET_DIR.resolve()
assert (DATASET_DIR/"images").exists() and (DATASET_DIR/"labels").exists(), \
    f"Expected images/ and labels/ under {DATASET_DIR}"

make_split_if_missing(DATASET_DIR)
yaml_path = write_yaml(DATASET_DIR, NAMES)
print(f"[info] Wrote dataset yaml → {yaml_path}")

# -----------------------
# Train YOLOv8 OBB
# -----------------------
# Good starting checkpoint: small model with OBB head.
model = YOLO("yolov8m-obb.pt")

results = model.train(
    data=str(yaml_path),
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=BATCH,
    device=DEVICE,
    project="runs_lfa_obb",
    name="y8m_lfa",
    optimizer="auto",
    lr0=0.0015,          # 稍降一点，更稳
    cos_lr=True,
    patience=70,         # ↑ 别过早停
    amp=True,
    mosaic=0.8,          # 0.5~1.0 之间试试；太强会破坏细条带
    degrees=10.0,        # 细目标，角度轻一点更稳
    translate=0.04,
    scale=0.05,
    shear=0.0,
    perspective=0.0,
    hsv_h=0.02, hsv_s=0.35, hsv_v=0.35,
    fliplr=0.5, flipud=0.0,
    pretrained=True,
    seed=0
)

# -----------------------
# Validate
# -----------------------
model.val(data=str(yaml_path), imgsz=IMG_SIZE, device=DEVICE)

# -----------------------
# Quick prediction demo
# -----------------------
pred_dir = DATASET_DIR/"images/val"
save_dir = Path("runs_lfa_obb/preds")
save_dir.mkdir(parents=True, exist_ok=True)

pred = model.predict(
    source=str(pred_dir),
    imgsz=IMG_SIZE,
    conf=0.10,          # 0.05~0.15 之间找一个你的数据最合适的
    iou=0.5,
    device=DEVICE,
    save=True,
    project=str(save_dir),
    name="val_vis",
    verbose=False
)

print(f"[info] Sample predictions saved under: {save_dir/'val_vis'}")
