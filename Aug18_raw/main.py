import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from process import main as process_main
from plot import normalize_csv 
from plot import plot_and_build_inverse_from_scaled
import shutil

# -------------------- 线性拟合+打分 --------------------
def _line_fit_metrics(xs: np.ndarray, ys: np.ndarray):
    """
    对所有点做 y = a + b x 的最小二乘拟合，返回：
      dict(a, b, rmse, r2, angle_score, dyn_range, mean_y)
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    assert n >= 2, "Need at least 2 points for line fit"

    # 拟合
    b, a = np.polyfit(xs, ys, 1)  # 注意：np.polyfit 返回 [slope, intercept]
    y_hat = a + b * xs

    # RMSE
    rmse = float(np.sqrt(np.mean((ys - y_hat) ** 2)))

    # R^2
    ss_res = float(np.sum((ys - y_hat) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    # 角度得分：sin(2θ) = 2|b|/(1+b^2), 0/90° -> 0, 45° -> 1
    angle_score = float(2.0 * abs(b) / (1.0 + b * b))

    # 动态范围与均值
    dyn_range = float(ys.max() - ys.min())
    mean_y = float(ys.mean())

    return dict(a=float(a), b=float(b), rmse=rmse, r2=float(r2),
                angle_score=angle_score, dyn_range=dyn_range, mean_y=mean_y)

def _score_margin_by_linefit(xs: np.ndarray, ys: np.ndarray,
                             wA: float = 0.5, wR: float = 0.25, wE: float = 1.0):
    """
    复合评分（越大越好）：
      score = r2 + wA*angle_score + wR*(dyn_range / (mean_y+eps)) - wE*(rmse / (dyn_range+eps))
    若动态范围太小（~近乎直线/无变化），直接返回极低分。
    """
    eps = 1e-12
    m = _line_fit_metrics(xs, ys)

    # 范围保护：避免“纯直线无意义”
    if m["dyn_range"] < 1e-6:
        return -1e12, m

    range_norm = m["dyn_range"] / (m["mean_y"] + eps)
    rmse_norm  = m["rmse"] / (m["dyn_range"] + eps)

    score = (m["r2"]
             + wA * m["angle_score"]
             + wR * range_norm
             - wE * rmse_norm)

    return float(score), m

# -------------------- 仅返回最佳 margin；最终结果写入 final_out_dir --------------------
def tune_best_margin(
    input_dir: str,
    final_out_dir: str,
    margins=range(4, 31),
    alpha: float = 0.35,
    save_mask: bool = False,
    weights=(0.5, 0.25, 1.0),   # (wA, wR, wE)
):
    """
    扫描 MARGIN（4..30），每次：
      - 跑 process_main -> 生成 summary.csv
      - normalize_csv -> 得到 summary_scaled.csv
      - 读取 test_scaled; 用全局线性拟合评分，挑选分数最大者
    中间结果使用 temp 目录覆盖，不保留；最后用最佳 margin 再跑一次到 final_out_dir，并写 summary_scaled.csv。
    返回 (best_margin, best_score, best_fit_metrics_dict)。
    """
    wA, wR, wE = weights
    temp_dir = Path(str(final_out_dir) + "_tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    best_score = None
    best_margin = None
    best_metrics = None

    for m in margins:
        # 清空临时目录
        for p in temp_dir.iterdir():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)

        # 1) 跑像素高亮（只影响 summary.csv）
        process_main(
            input_dir,
            str(temp_dir),
            cfg={"MARGIN": int(m), "ALPHA": float(alpha), "SAVE_MASK": bool(save_mask)},
        )

        # 2) 归一化（写 summary_scaled.csv）
        scaled_csv = temp_dir / "summary_scaled.csv"
        merged = normalize_csv(str(temp_dir / "summary.csv"), str(scaled_csv))

        # 3) 读取有效 test_scaled 并打分
        valid = merged.dropna(subset=["test_scaled"])
        if valid.empty or len(valid) < 2:
            score, metrics = -1e12, None
        else:
            xs = valid.index.values.astype(float)
            ys = valid["test_scaled"].values.astype(float).clip(min=0.0)
            score, metrics = _score_margin_by_linefit(xs, ys, wA=wA, wR=wR, wE=wE)

        # 记录最优
        if (best_score is None) or (score > best_score):
            best_score = float(score)
            best_margin = int(m)
            best_metrics = metrics

        print(f"[MARGIN={m:2d}] score={score:.6f} -> best={best_margin} ({best_score:.6f})")

    # 4) 用最佳 margin 跑最终目录，并写归一化 CSV
    final_out = Path(final_out_dir)
    final_out.mkdir(parents=True, exist_ok=True)
    for p in final_out.iterdir():
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)

    process_main(
        input_dir,
        str(final_out),
        cfg={"MARGIN": int(best_margin), "ALPHA": float(alpha), "SAVE_MASK": bool(save_mask)},
    )
    normalize_csv(str(final_out / "summary.csv"), str(final_out / "summary_scaled.csv"))

    with open(final_out / "best_margin.txt", "w") as f:
        f.write(f"{best_margin}\n")

    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n==> Best MARGIN: {best_margin} | score={best_score:.6f} | "
          f"fit: b={best_metrics['b']:.4f}, r2={best_metrics['r2']:.4f}, "
          f"rmse={best_metrics['rmse']:.4f}, angle={best_metrics['angle_score']:.3f}, "
          f"range={best_metrics['dyn_range']:.4f}")
    return best_margin, best_score, best_metrics


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------ 小工具：计算 RMSE ------------
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return np.inf
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))

# ------------ 拟合并在 xs 上预测：多项式 ------------
def _fit_poly(xs, ys, deg):
    coeffs = np.polyfit(xs, ys, deg)   # 高次在前
    poly = np.poly1d(coeffs)
    yhat = poly(xs)
    return yhat, {"name": f"poly{deg}", "coeffs": coeffs, "predict": poly}

# ------------ 对数线性：y = a + b*ln(x), x>0 ------------
def _fit_loglin(xs, ys):
    m = xs > 0
    if np.sum(m) < 2:
        return None
    X = np.c_[np.ones(np.sum(m)), np.log(xs[m])]
    beta, *_ = np.linalg.lstsq(X, ys[m], rcond=None)
    a, b = beta
    def pred(x):
        out = np.full_like(x, np.nan, dtype=float)
        mm = x > 0
        out[mm] = a + b * np.log(x[mm])
        return out
    yhat = pred(xs)
    return yhat, {"name": "loglin", "a": float(a), "b": float(b), "predict": pred}

# ------------ 指数：y = A*exp(Bx) -> ln(y) = ln A + B x, y>0 ------------
def _fit_exp(xs, ys):
    m = ys > 0
    if np.sum(m) < 2:
        return None
    X = np.c_[np.ones(np.sum(m)), xs[m]]
    beta, *_ = np.linalg.lstsq(X, np.log(ys[m]), rcond=None)
    la, B = beta
    A = np.exp(la)
    def pred(x):
        return A * np.exp(B * x)
    yhat = pred(xs)
    return yhat, {"name": "exp", "A": float(A), "B": float(B), "predict": pred}

# ------------ 幂律：y = A*x^B -> ln(y)=ln A + B ln x, x>0,y>0 ------------
def _fit_power(xs, ys):
    m = (xs > 0) & (ys > 0)
    if np.sum(m) < 2:
        return None
    X = np.c_[np.ones(np.sum(m)), np.log(xs[m])]
    beta, *_ = np.linalg.lstsq(X, np.log(ys[m]), rcond=None)
    la, B = beta
    A = np.exp(la)
    def pred(x):
        out = np.full_like(x, np.nan, dtype=float)
        mm = x > 0
        out[mm] = A * (x[mm] ** B)
        return out
    yhat = pred(xs)
    return yhat, {"name": "power", "A": float(A), "B": float(B), "predict": pred}

# ------------ 倒数：y = a + b/x, x!=0 ------------
def _fit_recip(xs, ys):
    m = xs != 0
    if np.sum(m) < 2:
        return None
    X = np.c_[np.ones(np.sum(m)), 1.0 / xs[m]]
    beta, *_ = np.linalg.lstsq(X, ys[m], rcond=None)
    a, b = beta
    def pred(x):
        out = np.full_like(x, np.nan, dtype=float)
        mm = x != 0
        out[mm] = a + b / x[mm]
        return out
    yhat = pred(xs)
    return yhat, {"name": "recip", "a": float(a), "b": float(b), "predict": pred}

def save_plot_from_scaled(
    scaled_csv: str,
    plot_path: str,
    plot_title: str = "",
    poly_degrees=(2, 3, 4, 5),
    try_loglin=True,
    try_exp=True,
    try_power=True,
    try_recip=True,
):
    """
    从 summary_scaled.csv 读取数据，保存折线图到 plot_path。
    - 保留原始点与分段直线
    - 叠加直线拟合（并显示 RMSE）
    - 在多种曲线模型中选择 RMSE 最小的一条叠加（并在图例中标注）
    """
    merged = pd.read_csv(scaled_csv).set_index("idx").sort_index()
    valid = merged.dropna(subset=["test_scaled"])
    xs = valid.index.values.astype(float)
    ys = valid["test_scaled"].values.astype(float)
    ys = np.clip(ys, 0, None)  # scale >= 0

    plt.figure(figsize=(8, 5))
    # 原始散点 + 分段直线
    plt.scatter(xs, ys, label="Test (scaled points)")
    plt.plot(xs, ys, label="Line through points")

    # 直线拟合 + RMSE
    best_line_rmse = None
    if len(xs) >= 2:
        slope, intercept = np.polyfit(xs, ys, 1)
        y_line_pred = intercept + slope * xs
        best_line_rmse = _rmse(ys, y_line_pred)

        x_line = np.linspace(xs.min(), xs.max(), 400)
        y_line = intercept + slope * x_line
        y_line = np.clip(y_line, 0, None)
        plt.plot(x_line, y_line, linestyle="--", linewidth=1.5,
                 label=f"Linear interpolation (RMSE={best_line_rmse:.2g})")

    # 准备候选模型
    candidates = []
    for d in poly_degrees:
        if len(xs) >= d + 1:
            yh, info = _fit_poly(xs, ys, d)
            yh = np.clip(yh, 0, None)
            candidates.append(("poly", d, yh, info))
    if try_loglin:
        out = _fit_loglin(xs, ys)
        if out is not None:
            yh, info = out
            yh = np.clip(yh, 0, None)
            candidates.append(("loglin", None, yh, info))
    if try_exp:
        out = _fit_exp(xs, ys)
        if out is not None:
            yh, info = out
            yh = np.clip(yh, 0, None)
            candidates.append(("exp", None, yh, info))
    if try_power:
        out = _fit_power(xs, ys)
        if out is not None:
            yh, info = out
            yh = np.clip(yh, 0, None)
            candidates.append(("power", None, yh, info))
    if try_recip:
        out = _fit_recip(xs, ys)
        if out is not None:
            yh, info = out
            yh = np.clip(yh, 0, None)
            candidates.append(("recip", None, yh, info))

    # 选 RMSE 最小的模型
    best_name = None
    best_info = None
    best_rmse = None
    for name, deg, yhat, info in candidates:
        r = _rmse(ys, yhat)
        if (best_rmse is None) or (r < best_rmse):
            best_rmse = r
            best_name = f"{info['name']}" if "name" in info else (f"{name}{deg}" if deg else name)
            best_info = info

    # 画出最佳曲线（若存在）
    if best_info is not None:
        x_dense = np.linspace(xs.min(), xs.max(), 600)
        if "predict" in best_info:
            y_dense = best_info["predict"](x_dense)
        else:
            y_dense = np.poly1d(best_info["coeffs"])(x_dense)
        y_dense = np.clip(y_dense, 0, None)
        plt.plot(x_dense, y_dense, linewidth=1.7,
                 label=f"Best curve: {best_name} (RMSE={best_rmse:.2g})")

    # 参考线
    plt.axhline(1.0, linestyle="--", linewidth=1, color="gray", label="Con = 1 (ref)")

    # 样式与保存
    ttl = "Test values scaled relative to Con"
    if plot_title:
        ttl += f": {plot_title}"
    plt.title(ttl)
    plt.xlabel("Test index (e.g., 0.1, 1, 2, 4, 6, 8, 10)")
    plt.ylabel("Scaled sum_gray_in_highlight (≥0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":




    # process_main("cockroach", "cockroach_output", cfg={"MARGIN": 24, "ALPHA": 0.35, "SAVE_MASK": True}) 
    # normalize_csv("cockroach_output/summary.csv", "cockroach_output/summary_scaled.csv") 
    # merged, f_inv = plot_and_build_inverse_from_scaled("cockroach_output/summary_scaled.csv", plot_title="Cockroach MARGIN=24")
    # save_plot_from_scaled(
    #     scaled_csv="cockroach_output/summary_scaled.csv",
    #     plot_path="interpolation_cockroach.png",
    #     plot_title="cockroach",
    #     poly_degrees=(1,2),
    #     try_loglin=True,
    #     try_exp=True,
    #     try_power=True,
    #     try_recip=True,
    # )
    
    # process_main("der_f", "der_f_output", cfg={"MARGIN": 5, "ALPHA": 0.35, "SAVE_MASK": True}) 
    # normalize_csv("der_f_output/summary.csv", "der_f_output/summary_scaled.csv") 
    # merged, f_inv = plot_and_build_inverse_from_scaled("der_f_output/summary_scaled.csv", plot_title="der_f MARGIN=5")
    save_plot_from_scaled(
        scaled_csv="der_f_output/summary_scaled.csv",
        plot_path="interpolation_der_f_without_0.png",
        plot_title="der_f_without_0",
        poly_degrees=(1,2),
        try_loglin=True,
        try_exp=True,
        try_power=True,
        try_recip=True,
    )
    
    # process_main("der_p", "der_p_output", cfg={"MARGIN": 3, "ALPHA": 0.35, "SAVE_MASK": True}) 
    # normalize_csv("der_p_output/summary.csv", "der_p_output/summary_scaled.csv") 
    # merged, f_inv = plot_and_build_inverse_from_scaled("der_p_output/summary_scaled.csv", plot_title="der_p MARGIN=3")
    # save_plot_from_scaled(
    #     scaled_csv="der_p_output/summary_scaled.csv",
    #     plot_path="interpolation_der_p.png",
    #     plot_title="der_p",
    #     poly_degrees=(1,2),
    #     try_loglin=True,
    #     try_exp=True,
    #     try_power=True,
    #     try_recip=True,
    # )



    # # 1) 自动选出最佳 margin (效果还不太好)
    # best_m, best_s, metrics = tune_best_margin(
    #     input_dir="der_p",
    #     final_out_dir="der_p_output",
    #     margins=range(4, 31),
    #     alpha=0.35,
    #     save_mask=False,
    #     weights=(0.5, 0.3, 0.25),   # (wA, wR, wE) 可按需微调
    # )
    # print("Best MARGIN =", best_m, "| score =", best_s)