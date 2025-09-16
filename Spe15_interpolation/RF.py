import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

sns.set(style="whitegrid")


def ensure_output_folder():
    output_dir = Path("RF")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def evaluate_and_plot(y_true, y_pred, output_dir, prefix="rf"):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])

    # Save raw output
    df.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)

    # Scatter: predicted vs true
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x="y_true", y="y_pred", data=df)
    plt.plot([df.y_true.min(), df.y_true.max()], [df.y_true.min(), df.y_true.max()], "--r")
    plt.xlabel("True ng")
    plt.ylabel("Predicted ng")
    plt.title("RF Prediction vs True")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_scatter.png")
    plt.close()

    # Error by ng group
    plt.figure(figsize=(6, 5))
    sns.barplot(data=df, x="y_true", y="abs_error", errorbar=None)
    plt.xlabel("True ng")
    plt.ylabel("Absolute Error")
    plt.title("Error by ng Value")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_error_bar.png")
    plt.close()


def plot_feature_importance(model, feature_names, output_dir, prefix="rf"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_feature_importance.png")
    plt.close()


def train_rf_with_cv(csv_path: str, n_splits: int = 10, random_state: int = 42):
    output_dir = ensure_output_folder()
    df = pd.read_csv(csv_path)
    df = df[df["type"] == "test"].copy()

    exclude_cols = ["filename", "file_path", "ng", "class", "source", "type", "id"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].values
    y = df["ng"].values

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 5, 10]
    }

    best_score = -np.inf
    best_model = None
    best_params = None
    all_results = []

    pbar = tqdm(
        [(n, d) for n in param_grid["n_estimators"] for d in param_grid["max_depth"]],
        desc="Searching RF hyperparams"
    )

    for n_est, max_d in pbar:
        preds, truths = [], []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for train_idx, val_idx in kf.split(X):
            model = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=random_state)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            preds.extend(pred)
            truths.extend(y[val_idx])

        mse = mean_squared_error(truths, preds)
        mae = mean_absolute_error(truths, preds)
        r2 = r2_score(truths, preds)

        all_results.append({
            "n_estimators": n_est,
            "max_depth": max_d,
            "MSE": mse,
            "MAE": mae,
            "R2": r2
        })

        if r2 > best_score:
            best_score = r2
            best_model = model
            best_params = (n_est, max_d)

    # Save metrics table
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "rf_hyperparam_results.csv", index=False)

    print("\n✅ Best Model Found:")
    print(f"   - n_estimators: {best_params[0]}")
    print(f"   - max_depth   : {best_params[1]}")
    print(f"   - R2 Score    : {best_score:.4f}")

    # Save final model
    final_model_path = output_dir / "rf_model_best.pkl"
    joblib.dump(best_model, final_model_path)
    print(f"💾 Model saved to {final_model_path}")

    # Final predictions on whole set
    y_final_pred = best_model.predict(X)
    evaluate_and_plot(y, y_final_pred, output_dir, prefix="rf")
    plot_feature_importance(best_model, feature_cols, output_dir, prefix="rf")


if __name__ == "__main__":
    csv_path = "label.csv"
    train_rf_with_cv(csv_path)
