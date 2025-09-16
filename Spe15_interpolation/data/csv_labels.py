import cv2
import numpy as np
import pandas as pd
import os
import re

def extract_full_features(img):
    """Extract extended color features from a stripe image."""
    # Convert to RGB and HSV
    rgb = img
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

    # Split channels
    R, G, B = cv2.split(rgb)
    H, S, V = cv2.split(hsv)

    # Convert to 1D for sorting and stats
    R_flat = R.flatten()
    G_flat = G.flatten()
    B_flat = B.flatten()

    # Mean
    mean_R = np.mean(R_flat)
    mean_G = np.mean(G_flat)
    mean_B = np.mean(B_flat)

    # Sum (accumulate)
    sum_R = np.sum(R_flat)
    sum_G = np.sum(G_flat)
    sum_B = np.sum(B_flat)

    # Max
    max_R = np.max(R_flat)
    max_G = np.max(G_flat)
    max_B = np.max(B_flat)

    # Min
    min_R = np.min(R_flat)
    min_G = np.min(G_flat)
    min_B = np.min(B_flat)

    # Median
    median_R = np.median(R_flat)
    median_G = np.median(G_flat)
    median_B = np.median(B_flat)

    # Std HSV
    std_HSV = np.std(H) + np.std(S) + np.std(V)

    # R/G ratio
    rg_ratio = mean_R / (mean_G + 1e-6)

    # Max grayscale intensity
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    max_intensity = np.max(gray)

    return {
        "mean_R": mean_R,
        "mean_G": mean_G,
        "mean_B": mean_B,
        "sum_R": sum_R,
        "sum_G": sum_G,
        "sum_B": sum_B,
        "max_R": max_R,
        "max_G": max_G,
        "max_B": max_B,
        "min_R": min_R,
        "min_G": min_G,
        "min_B": min_B,
        "median_R": median_R,
        "median_G": median_G,
        "median_B": median_B,
        "std_HSV": std_HSV,
        "R/G": rg_ratio,
        "max_intensity": max_intensity
    }

# Re-run processing with extended features
def main_extended():
    input_folder = "resized"
    output_csv = "label.csv"

    image_extensions = (".png", ".jpg", ".jpeg")
    records = []

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(image_extensions):
            continue

        fpath = os.path.join(input_folder, fname)
        match = re.match(r"(?P<ng>\d+\.?\d*)ng_(?P<source>[A-Za-z]+)_(?P<type>ctrl|test)_img(?P<id>\d+).png", fname)
        if not match:
            continue

        img = cv2.imread(fpath)
        if img is None:
            continue

        meta = match.groupdict()
        meta["filename"] = fname
        meta["file_path"] = fpath
        meta["ng"] = float(meta["ng"])
        meta["id"] = int(meta["id"])
        meta["class"] = None

        # Use extended features
        features = extract_full_features(img)
        meta.update(features)

        records.append(meta)

    df = pd.DataFrame(records)

    if "ng" in df.columns:
        df["ng"] = pd.to_numeric(df["ng"], errors='coerce')
        df["id"] = pd.to_numeric(df["id"], errors='coerce')
        ng_to_class = {ng: idx for idx, ng in enumerate(sorted(df["ng"].dropna().unique()))}
        df["class"] = df["ng"].map(ng_to_class)

    df.to_csv(output_csv, index=False)
    return output_csv


main_extended()