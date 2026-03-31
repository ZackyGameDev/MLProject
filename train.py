"""
Model Training & Evaluation Pipeline
=====================================
Runs 4 models (Ridge, SVR, Random Forest, XGBoost) with GroupKFold
cross-validation and produces:
  1. Console comparison table (RMSE, MAE, R2)
  2. model_comparison.png    — bar chart of model metrics
  3. feature_importance.png  — top features from XGBoost
  4. predicted_vs_actual.png — scatter plot for the best model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os
import joblib

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────
DATASET = "cf_ml_dataset.csv"
TARGET = "future_rating_delta_30d"
GROUP_COL = "handle"
N_SPLITS = 5
RANDOM_STATE = 42

# Plot style
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.family": "sans-serif",
    "font.size": 11,
})

# ─── Load Data ────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATASET)

groups = df[GROUP_COL]
feature_cols = [c for c in df.columns if c not in [TARGET, GROUP_COL]]
X = df[feature_cols].values
y = df[TARGET].values

print(f"  Samples: {len(df)}  |  Features: {len(feature_cols)}  |  Unique users: {groups.nunique()}")
print()

# ─── Define Models ────────────────────────────────────────────────────
models = {
    "Ridge": Ridge(alpha=10.0),
    "SVR (RBF)": SVR(kernel="rbf", C=100, epsilon=10, gamma="scale"),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=5,
        random_state=RANDOM_STATE, verbosity=0,
        device="cuda", tree_method="hist",
    ),
}

# ─── GroupKFold Cross-Validation ──────────────────────────────────────
gkf = GroupKFold(n_splits=N_SPLITS)

results = {}
all_predictions = {}

print(f"Running {N_SPLITS}-fold GroupKFold cross-validation...")
print("=" * 65)

for name, model in models.items():
    fold_metrics = {"rmse": [], "mae": [], "r2": []}
    y_true_all, y_pred_all = [], []
    t0 = time.time()

    needs_scaling = name in ["Ridge", "SVR (RBF)"]

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if needs_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fold_metrics["rmse"].append(np.sqrt(mean_squared_error(y_test, preds)))
        fold_metrics["mae"].append(mean_absolute_error(y_test, preds))
        fold_metrics["r2"].append(r2_score(y_test, preds))

        y_true_all.extend(y_test)
        y_pred_all.extend(preds)

    elapsed = time.time() - t0

    avg = {k: np.mean(v) for k, v in fold_metrics.items()}
    std = {k: np.std(v) for k, v in fold_metrics.items()}
    results[name] = {"avg": avg, "std": std, "time": elapsed}
    all_predictions[name] = (np.array(y_true_all), np.array(y_pred_all))

    print(f"\n  {name}")
    print(f"    RMSE : {avg['rmse']:.2f} ± {std['rmse']:.2f}")
    print(f"    MAE  : {avg['mae']:.2f} ± {std['mae']:.2f}")
    print(f"    R2   : {avg['r2']:.4f} ± {std['r2']:.4f}")
    print(f"    Time : {elapsed:.1f}s")

print("\n" + "=" * 65)

# ─── Summary Table ────────────────────────────────────────────────────
print("\n[RESULTS] Model Comparison Summary")
print("-" * 65)
print(f"  {'Model':<20} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'Time':>8}")
print("-" * 65)

for name, r in results.items():
    print(f"  {name:<20} {r['avg']['rmse']:>10.2f} {r['avg']['mae']:>10.2f} {r['avg']['r2']:>10.4f} {r['time']:>7.1f}s")

best_model = min(results, key=lambda k: results[k]["avg"]["rmse"])
print("-" * 65)
print(f"  >>> Best model: {best_model}")
print()

# ─── Save Models (retrain on full data) ───────────────────────────────
print("Retraining all models on full dataset and saving...")
os.makedirs("models", exist_ok=True)

# Scaler for models that need it
full_scaler = StandardScaler()
X_scaled = full_scaler.fit_transform(X)
joblib.dump(full_scaler, "models/scaler.joblib")
joblib.dump(feature_cols, "models/feature_columns.joblib")

saved_models = {}
for name, model in models.items():
    needs_scaling = name in ["Ridge", "SVR (RBF)"]
    X_fit = X_scaled if needs_scaling else X
    model.fit(X_fit, y)
    
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = f"models/{safe_name}.joblib"
    joblib.dump(model, path)
    saved_models[name] = path
    print(f"  [OK] Saved {path}")

print()

# ─── Plot 1: Model Comparison Bar Chart ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Comparison (GroupKFold CV)", fontsize=16, fontweight="bold", color="#58a6ff")

metrics_to_plot = [("rmse", "RMSE (lower=better)", "#f97583"), ("mae", "MAE (lower=better)", "#d2a8ff"), ("r2", "R2 (higher=better)", "#56d364")]
model_names = list(results.keys())

for ax, (metric, label, color) in zip(axes, metrics_to_plot):
    vals = [results[m]["avg"][metric] for m in model_names]
    errs = [results[m]["std"][metric] for m in model_names]
    bars = ax.barh(model_names, vals, xerr=errs, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=10, color="#c9d1d9")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("[OK] Saved model_comparison.png")

# ─── Plot 2: Feature Importance (XGBoost) ────────────────────────────
# Use the already-trained XGBoost from the save step
xgb_full = joblib.load("models/xgboost.joblib")

importances = xgb_full.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

# Color by factor
factor_colors = {}
upsolving = ["upsolve_count", "upsolve_ratio", "upsolve_difficulty_delta", "contest_to_practice_ratio"]
consistency = ["subs_90d", "ac_ratio", "active_days", "avg_gap_days", "max_gap_days", "weekly_consistency", "streak_max", "speed_vs_growth_ratio"]
iq = ["base_rating", "rating_delta_90d", "rating_volatility", "rating_trend_slope", "contest_rank_percentile_avg", "solve_speed_avg", "difficulty_ceiling", "problem_rating_vs_user_rating"]
diversity = ["tag_entropy", "unique_tags_count", "new_tags_explored", "tag_concentration_top3"]
efficiency = ["attempts_per_ac", "first_attempt_ac_rate", "avg_debug_time"]
contest = ["contest_count", "contest_problems_solved_avg"]

for f in upsolving: factor_colors[f] = "#f97583"
for f in consistency: factor_colors[f] = "#56d364"
for f in iq: factor_colors[f] = "#58a6ff"
for f in diversity: factor_colors[f] = "#d2a8ff"
for f in efficiency: factor_colors[f] = "#e3b341"
for f in contest: factor_colors[f] = "#79c0ff"

colors = [factor_colors.get(f, "#8b949e") for f in feat_imp.index]

fig, ax = plt.subplots(figsize=(10, 9))
ax.barh(feat_imp.index, feat_imp.values, color=colors, alpha=0.9, edgecolor="white", linewidth=0.3)
ax.set_title("Feature Importance (XGBoost)", fontsize=16, fontweight="bold", color="#58a6ff")
ax.set_xlabel("Importance (gain)", fontsize=12)
ax.grid(axis="x", alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#f97583", label="Upsolving"),
    Patch(facecolor="#56d364", label="Consistency"),
    Patch(facecolor="#58a6ff", label="IQ"),
    Patch(facecolor="#d2a8ff", label="Topic Diversity"),
    Patch(facecolor="#e3b341", label="Learning Efficiency"),
    Patch(facecolor="#79c0ff", label="Contest Behavior"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10,
          facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
print("[OK] Saved feature_importance.png")

# ─── Plot 3: Predicted vs Actual (Best Model) ────────────────────────
y_true, y_pred = all_predictions[best_model]

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_true, y_pred, alpha=0.4, s=20, c="#58a6ff", edgecolors="none")

# Perfect prediction line
lims = [min(y_true.min(), y_pred.min()) - 20, max(y_true.max(), y_pred.max()) + 20]
ax.plot(lims, lims, "--", color="#f97583", linewidth=1.5, alpha=0.8, label="Perfect prediction")

ax.set_xlabel("Actual Rating Delta", fontsize=12)
ax.set_ylabel("Predicted Rating Delta", fontsize=12)
ax.set_title(f"Predicted vs Actual — {best_model}", fontsize=16, fontweight="bold", color="#58a6ff")
ax.legend(fontsize=11, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
ax.grid(alpha=0.3)

r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
ax.text(0.05, 0.92, f"R² = {r2:.4f}\nRMSE = {rmse:.1f}",
        transform=ax.transAxes, fontsize=12, color="#56d364",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d1117", edgecolor="#30363d"))

plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=150, bbox_inches="tight")
print("[OK] Saved predicted_vs_actual.png")

print("\nDone! Check the generated plots.")
