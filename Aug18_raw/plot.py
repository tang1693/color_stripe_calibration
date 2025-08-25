import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- helpers (unchanged) ---
def _parse_idx_from_filename(s: str) -> float:
    """
    con_01.png -> 0.1
    test_8.png -> 8
    test_10.png -> 10
    """
    import re
    m = re.search(r'_(\d+)\.', s)
    if not m:
        raise ValueError(f"Cannot parse index from filename: {s}")
    token = m.group(1)
    return float(token) / 10.0 if (len(token) > 1 and token.startswith("0")) else float(token)

def f_inv_piecewise_builder(xs: np.ndarray, ys: np.ndarray):
    """
    Build inverse for a piecewise-linear (xs, ys) polyline (xs strictly increasing).
    Returns f_inv(y) -> x, clamping y >= 0.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    assert np.all(np.diff(xs) > 0), "xs must be strictly increasing"

    def inv(y_star: float) -> float:
        y_star = max(0.0, float(y_star))
        cand_x, cand_mid = [], []
        for i in range(len(xs) - 1):
            x0, x1 = xs[i], xs[i+1]
            y0, y1 = ys[i], ys[i+1]
            lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
            if y1 != y0 and lo <= y_star <= hi:
                t = (y_star - y0) / (y1 - y0)
                x_star = x0 + t * (x1 - x0)
                cand_x.append(x_star)
                cand_mid.append(abs(((x0 + x1) / 2.0) - x_star))
        if cand_x:
            return float(cand_x[int(np.argmin(cand_mid))])
        i_best = int(np.argmin(np.abs(ys - y_star)))
        return float(xs[i_best])

    return inv

# --- 1) Normalize ONLY (reads summary.csv and writes summary_scaled.csv) ---
def normalize_csv(csv_in: str, out_csv: str = "summary_scaled.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_in)

    # parse index & group
    df["idx"] = df["file"].apply(_parse_idx_from_filename)
    df["group"] = np.where(df["file"].str.startswith("con"), "con", "test")

    con = df[df["group"] == "con"].set_index("idx")
    test = df[df["group"] == "test"].set_index("idx")

    merged = con[["sum_gray_in_highlight"]].rename(columns={"sum_gray_in_highlight": "con_sum"}).join(
        test[["sum_gray_in_highlight"]].rename(columns={"sum_gray_in_highlight": "test_sum"}),
        how="outer"
    )

    merged["factor"] = np.where(merged["con_sum"] > 0, 1.0 / merged["con_sum"], np.nan)
    merged["test_scaled"] = (merged["test_sum"] * merged["factor"]).clip(lower=0)

    merged = merged.sort_index()
    merged.to_csv(out_csv, index_label="idx")
    print(f"Saved scaled results to {out_csv}")
    return merged

# --- 2) Plot & inverse ONLY (reads the scaled CSV produced above) ---
def plot_and_build_inverse_from_scaled(scaled_csv: str, plot_title: str = ""):
    merged = pd.read_csv(scaled_csv).set_index("idx").sort_index()
    valid = merged.dropna(subset=["test_scaled"])

    xs = valid.index.values
    ys = valid["test_scaled"].values

    # Plot: points + straight lines, clamp >=0 already ensured
    plt.figure(figsize=(8,5))
    plt.scatter(xs, ys, color="blue", label="Test (scaled points)")
    plt.plot(xs, ys, color="orange", label="Linear interpolation")
    plt.axhline(1.0, linestyle="--", linewidth=1, label="Con = 1 (ref)")
    plt.xlabel("Test index (e.g., 0.1, 1, 2, 4, 6, 8, 10)")
    plt.ylabel("Scaled sum_gray_in_highlight (≥0)")
    plt.title(("Test values scaled relative to Con: " + plot_title).strip(": "))
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    # Build inverse: scale -> test index
    f_inv = f_inv_piecewise_builder(xs, ys)
    return merged, f_inv


if __name__ == "__main__":
    
    normalize_csv("cockroach_output/summary.csv", "cockroach_output/summary_scaled.csv")
    merged, f_inv = plot_and_build_inverse_from_scaled("cockroach_output/summary_scaled.csv", plot_title="Cockroach")

    # demo queries (scale -> test index)
    for y in [0.2, 0.5, 1.0]:
        print(f"scale={y}  =>  test≈{f_inv(y):.3f}")
