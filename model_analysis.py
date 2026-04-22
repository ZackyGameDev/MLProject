"""
Model Accuracy Analysis by Rating Bucket
==========================================
Picks 50 random users from each rating bucket and analyzes which model
is most accurate for each user type, using GroupKFold out-of-sample predictions.

Outputs:
  - Console summary table
  - model_analysis.png (bar chart of model wins per bucket)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import joblib
import warnings

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────
DATASET = "cf_ml_dataset.csv"
TARGET = "future_rating_delta_30d"
GROUP_COL = "handle"
N_SPLITS = 5
RANDOM_STATE = 42
USERS_PER_BUCKET = 50

BUCKETS = {
    "Newbie (<1000)":     (0, 1000),
    "Pupil (1000-1400)":  (1000, 1400),
    "Specialist (1400-1800)": (1400, 1800),
    "Expert (1800-2200)": (1800, 2200),
    "Master+ (2200+)":    (2200, 10000),
}

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

# ─── Assign each user to a bucket based on their median base_rating ───
user_median_rating = df.groupby(GROUP_COL)["base_rating"].median()

def get_bucket(rating):
    for name, (lo, hi) in BUCKETS.items():
        if lo <= rating < hi:
            return name
    return "Master+ (2200+)"

user_bucket = user_median_rating.apply(get_bucket)

# Sample 50 users per bucket
np.random.seed(RANDOM_STATE)
selected_users = []
for bucket_name in BUCKETS:
    bucket_users = user_bucket[user_bucket == bucket_name].index.tolist()
    n_available = len(bucket_users)
    n_sample = min(USERS_PER_BUCKET, n_available)
    sampled = list(np.random.choice(bucket_users, size=n_sample, replace=False))
    selected_users.extend(sampled)
    print(f"  {bucket_name}: {n_available} users available, sampled {n_sample}")

print(f"\nTotal selected users: {len(selected_users)}")

# Filter dataset to selected users
mask = df[GROUP_COL].isin(selected_users)
df_selected = df[mask].copy()
X_sel = df_selected[feature_cols].values
y_sel = df_selected[TARGET].values
groups_sel = df_selected[GROUP_COL]

print(f"Total samples for selected users: {len(df_selected)}")

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

# ─── Run GroupKFold to get out-of-sample predictions ──────────────────
print("\nRunning GroupKFold cross-validation on selected users...")
gkf = GroupKFold(n_splits=N_SPLITS)

# Store per-sample predictions for each model
sample_predictions = {name: np.full(len(df_selected), np.nan) for name in models}

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_sel, y_sel, groups_sel)):
    X_train, X_test = X_sel[train_idx], X_sel[test_idx]
    y_train, y_test = y_sel[train_idx], y_sel[test_idx]

    # Fit scaler for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        needs_scaling = name in ["Ridge", "SVR (RBF)"]
        Xtr = X_train_scaled if needs_scaling else X_train
        Xte = X_test_scaled if needs_scaling else X_test

        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        sample_predictions[name][test_idx] = preds

    print(f"  Fold {fold + 1}/{N_SPLITS} done")

# Add ensemble predictions
sample_predictions["Ensemble"] = np.mean(
    [sample_predictions[name] for name in models], axis=0
)

# ─── Compute per-sample errors ────────────────────────────────────────
model_names = list(models.keys()) + ["Ensemble"]
errors = {}
for name in model_names:
    errors[name] = np.abs(sample_predictions[name] - y_sel)

# Find which model wins for each sample
best_model_per_sample = []
for i in range(len(df_selected)):
    best = min(model_names, key=lambda m: errors[m][i])
    best_model_per_sample.append(best)

df_selected = df_selected.copy()
df_selected["best_model"] = best_model_per_sample
df_selected["actual_delta"] = y_sel
df_selected["bucket"] = df_selected["base_rating"].apply(get_bucket)
df_selected["abs_delta"] = np.abs(y_sel)

# Also categorize by magnitude of change
def delta_category(delta):
    abs_d = abs(delta)
    if abs_d < 50:
        return "Small (|delta| < 50)"
    elif abs_d < 150:
        return "Medium (50-150)"
    else:
        return "Large (|delta| > 150)"

df_selected["delta_category"] = df_selected["actual_delta"].apply(delta_category)

# ─── Analysis 1: Model wins by rating bucket ─────────────────────────
print("\n" + "=" * 75)
print("ANALYSIS 1: Which model wins most often per rating bucket?")
print("=" * 75)

bucket_wins = df_selected.groupby(["bucket", "best_model"]).size().unstack(fill_value=0)
# Reorder columns
for m in model_names:
    if m not in bucket_wins.columns:
        bucket_wins[m] = 0
bucket_wins = bucket_wins[model_names]

# Reorder rows
bucket_order = list(BUCKETS.keys())
bucket_wins = bucket_wins.reindex([b for b in bucket_order if b in bucket_wins.index])

print("\nWin counts (how many samples each model was the most accurate for):\n")
print(bucket_wins.to_string())

print("\nWin percentages:\n")
bucket_pct = bucket_wins.div(bucket_wins.sum(axis=1), axis=0) * 100
print(bucket_pct.round(1).to_string())

# ─── Analysis 2: Model wins by magnitude of change ───────────────────
print("\n" + "=" * 75)
print("ANALYSIS 2: Which model wins by magnitude of actual rating change?")
print("=" * 75)

delta_wins = df_selected.groupby(["delta_category", "best_model"]).size().unstack(fill_value=0)
for m in model_names:
    if m not in delta_wins.columns:
        delta_wins[m] = 0
delta_wins = delta_wins[model_names]
delta_order = ["Small (|delta| < 50)", "Medium (50-150)", "Large (|delta| > 150)"]
delta_wins = delta_wins.reindex([d for d in delta_order if d in delta_wins.index])

print("\nWin counts:\n")
print(delta_wins.to_string())

print("\nWin percentages:\n")
delta_pct = delta_wins.div(delta_wins.sum(axis=1), axis=0) * 100
print(delta_pct.round(1).to_string())

# ─── Analysis 3: Mean absolute error per model per bucket ────────────
print("\n" + "=" * 75)
print("ANALYSIS 3: Mean Absolute Error (MAE) per model per bucket")
print("=" * 75)

for name in model_names:
    df_selected[f"error_{name}"] = errors[name]

mae_table = pd.DataFrame()
for name in model_names:
    mae_table[name] = df_selected.groupby("bucket")[f"error_{name}"].mean()
mae_table = mae_table.reindex([b for b in bucket_order if b in mae_table.index])

print("\n")
print(mae_table.round(1).to_string())

# Highlight the best model per bucket
print("\nBest model per bucket (lowest MAE):")
for bucket in mae_table.index:
    best = mae_table.loc[bucket].idxmin()
    val = mae_table.loc[bucket].min()
    print(f"  {bucket:<25s}: {best} (MAE = {val:.1f})")

# ─── Analysis 4: Mean absolute error by delta magnitude ──────────────
print("\n" + "=" * 75)
print("ANALYSIS 4: Mean Absolute Error (MAE) by magnitude of change")
print("=" * 75)

mae_delta = pd.DataFrame()
for name in model_names:
    mae_delta[name] = df_selected.groupby("delta_category")[f"error_{name}"].mean()
mae_delta = mae_delta.reindex([d for d in delta_order if d in mae_delta.index])

print("\n")
print(mae_delta.round(1).to_string())

print("\nBest model per magnitude:")
for cat in mae_delta.index:
    best = mae_delta.loc[cat].idxmin()
    val = mae_delta.loc[cat].min()
    print(f"  {cat:<30s}: {best} (MAE = {val:.1f})")

# ─── Plot: Model win rate by bucket ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Which Model is Best for Which Users?", fontsize=18, fontweight="bold", color="#58a6ff")

colors = {
    "Ridge": "#f97583",
    "SVR (RBF)": "#d2a8ff",
    "Random Forest": "#56d364",
    "XGBoost": "#58a6ff",
    "Ensemble": "#e3b341",
}

# Plot 1: Win % by rating bucket
ax = axes[0]
bucket_pct.plot(kind="barh", stacked=True, ax=ax, 
                color=[colors[m] for m in bucket_pct.columns],
                edgecolor="white", linewidth=0.3)
ax.set_title("Win Rate by Rating Bucket", fontsize=14, fontweight="bold")
ax.set_xlabel("% of samples where model had lowest error")
ax.legend(loc="lower right", fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
ax.grid(axis="x", alpha=0.3)

# Plot 2: MAE by rating bucket
ax = axes[1]
mae_table.plot(kind="bar", ax=ax,
               color=[colors[m] for m in mae_table.columns],
               edgecolor="white", linewidth=0.3)
ax.set_title("MAE by Rating Bucket", fontsize=14, fontweight="bold")
ax.set_ylabel("Mean Absolute Error")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(loc="upper right", fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("model_analysis.png", dpi=150, bbox_inches="tight")
print("\n[OK] Saved model_analysis.png")

# ─── Final Summary ───────────────────────────────────────────────────
print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)

overall_wins = df_selected["best_model"].value_counts()
print("\nOverall model win counts (across all samples):")
for name in model_names:
    count = overall_wins.get(name, 0)
    pct = count / len(df_selected) * 100
    print(f"  {name:<20s}: {count:5d} wins ({pct:.1f}%)")

print("\nDone!")
