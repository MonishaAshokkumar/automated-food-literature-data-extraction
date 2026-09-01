"""
08_visualisations.py
====================
Generates publication-ready figures for the dissertation.

  Figure 1 — Actual vs Predicted scatter plots (all 5 targets)
  Figure 2 — Feature importance / coefficient charts (all 5 models)
  Figure 3 — Meta-dataset variable coverage bar chart

All figures saved as high-resolution PNG to:
  D:\\Monisha\\UCC Project\\outputs\\figures\\

Run after 02_ml_model.py (needs ml_extracted_predictions.csv and .pkl models).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
from matplotlib.patches import Patch
from sklearn.metrics import r2_score, mean_absolute_error

sys.stdout.reconfigure(encoding="utf-8")

# ── Configuration ──────────────────────────────────────────────────────────────
OUTPUT_DIR  = r"D:\Monisha\UCC Project\outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

TARGETS = [
    "Viscosity_Pa_s",
    "d90_µm",
    "Sensory_thickness",
    "Sensory_creaminess",
    "Friction_coefficient",
]

TARGET_LABELS = {
    "Viscosity_Pa_s":       "Viscosity (Pa·s)",
    "d90_µm":              "d₉₀ (µm)",
    "Sensory_thickness":    "Sensory Thickness (/10)",
    "Sensory_creaminess":   "Sensory Creaminess (/10)",
    "Friction_coefficient": "Friction Coefficient",
}

SPLIT_TYPE = {
    "Viscosity_Pa_s":       "Holdout — 7 test papers",
    "d90_µm":              "Holdout — 5 test papers",
    "Sensory_thickness":    "Leave-One-Out CV",
    "Sensory_creaminess":   "Leave-One-Out CV",
    "Friction_coefficient": "Leave-One-Out CV",
}

INPUT_ML = [
    "Temperature_C",
    "Homogenization_rpm",
    "Pressure_MPa",
    "Fat_concentration_wt%",
    "Concentration_wt%",
    "Protein_type",
    "Oil_Fat_type",
    "Volume_mL",
]

OUTPUT_ML = [
    "Viscosity_Pa_s",
    "d90_µm",
    "Sensory_thickness",
    "Sensory_creaminess",
    "Friction_coefficient",
]

# ── Global plot style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
})

COL_BLUE  = "#2166ac"   # input features
COL_RED   = "#d6604d"   # outputs / perfect-prediction line
COL_GREEN = "#1a9850"   # ridge coefficients
COL_GREY  = "#555555"   # annotations


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Actual vs Predicted scatter plots
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("FIGURE 1: Actual vs Predicted scatter plots")
print("=" * 60)

preds_path = os.path.join(OUTPUT_DIR, "ml_extracted_predictions.csv")
if not os.path.exists(preds_path):
    print(f"  [ERROR] {preds_path} not found.")
    print("  Run 02_ml_model.py first, then re-run this script.")
    sys.exit(1)

preds_df = pd.read_csv(preds_path)
print(f"  Loaded {len(preds_df)} rows  |  targets: {sorted(preds_df['Target'].unique())}\n")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes_flat = axes.flatten()

for i, target in enumerate(TARGETS):
    ax = axes_flat[i]
    df_t = preds_df[preds_df["Target"] == target].copy()

    if df_t.empty:
        ax.set_visible(False)
        print(f"  {target:30s}  [no data — skipped]")
        continue

    actual    = df_t["Actual"].values
    predicted = df_t["Predicted"].values

    r2  = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)

    # Scatter points
    ax.scatter(actual, predicted,
               color=COL_BLUE, edgecolors="white", s=80,
               linewidths=0.8, zorder=3, label="Papers")

    # 1:1 perfect prediction line
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    margin = (hi - lo) * 0.1 if hi != lo else 1.0
    lo -= margin
    hi += margin
    ax.plot([lo, hi], [lo, hi], color=COL_RED, linestyle="--",
            linewidth=1.5, label="Perfect prediction", zorder=2)

    ax.set_xlabel(f"Actual — {TARGET_LABELS[target]}")
    ax.set_ylabel(f"Predicted — {TARGET_LABELS[target]}")
    ax.set_title(
        f"{TARGET_LABELS[target]}\n"
        f"R² = {r2:.3f}  ·  MAE = {mae:.3f}  ·  {SPLIT_TYPE[target]}",
        fontsize=10
    )
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    print(f"  {target:30s}  R²={r2:7.3f}  MAE={mae:.4f}  n={len(actual)}")

# Hide the unused 6th subplot
axes_flat[5].set_visible(False)

fig.suptitle("Actual vs Predicted Values — All Target Variables",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout(pad=2.0)

out1 = os.path.join(FIGURES_DIR, "fig1_actual_vs_predicted.png")
fig.savefig(out1, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\n  → Saved: {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Feature importance / Ridge coefficients
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("FIGURE 2: Feature importance / Ridge coefficient charts")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes_flat = axes.flatten()

for i, target in enumerate(TARGETS):
    ax = axes_flat[i]

    model_path = os.path.join(OUTPUT_DIR, f"model_{target}.pkl")
    if not os.path.exists(model_path):
        ax.set_visible(False)
        print(f"  [{target}] model file not found — skipping")
        continue

    pkg        = joblib.load(model_path)
    model      = pkg["model"]
    feat_cols  = pkg["feature_cols"]
    model_name = type(model).__name__

    # Extract importance values
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        x_label     = "Feature Importance (Gini)"
        bar_color   = COL_BLUE
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        total       = importances.sum()
        importances = importances / total if total > 0 else importances
        x_label     = "|Coefficient| (normalised)"
        bar_color   = COL_GREEN
    else:
        ax.set_visible(False)
        continue

    # Sort descending
    order       = np.argsort(importances)[::-1]
    sorted_imp  = importances[order]
    sorted_feat = [feat_cols[j] for j in order]

    # Readable labels
    display_labels = [
        f.replace("_wt%", " (wt%)")
         .replace("_rpm", " (rpm)")
         .replace("_MPa", " (MPa)")
         .replace("_mL",  " (mL)")
         .replace("_C",   " (°C)")
         .replace("_",    " ")
        for f in sorted_feat
    ]

    # Plot horizontal bars (ascending so longest bar is at top)
    y_pos = range(len(sorted_imp))
    ax.barh(list(y_pos), sorted_imp[::-1],
            color=bar_color, edgecolor="white", linewidth=0.4)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(display_labels[::-1], fontsize=8)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_title(f"{TARGET_LABELS[target]}\n({model_name})", fontsize=10)

    # Value annotations
    for j, val in enumerate(sorted_imp[::-1]):
        ax.text(val + 0.002, j, f"{val:.3f}",
                va="center", fontsize=7, color=COL_GREY)

    ax.set_xlim(0, sorted_imp.max() * 1.25)

    print(f"  {target:30s}  model={model_name:25s}  top={sorted_feat[0]}")

axes_flat[5].set_visible(False)

fig.suptitle("Feature Importance by Target Variable",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout(pad=2.0)

out2 = os.path.join(FIGURES_DIR, "fig2_feature_importance.png")
fig.savefig(out2, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\n  → Saved: {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Meta-dataset variable coverage
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("FIGURE 3: Meta-dataset variable coverage")
print("=" * 60)

xl_path = os.path.join(OUTPUT_DIR, "extended_with_ocr.xlsx")
xl      = pd.ExcelFile(xl_path)
frames  = [pd.read_excel(xl, sheet_name=s) for s in xl.sheet_names]
all_df  = pd.concat(frames, ignore_index=True)
all_df["Value"] = pd.to_numeric(all_df["Value"], errors="coerce")

wide = (
    all_df[all_df["Value"].notna()]
    .groupby(["Paper", "Normalized_Variable"])["Value"]
    .median()
    .unstack()
)
wide = wide.drop(columns=["nan"], errors="ignore")

n_papers = len(wide)

vars_to_show = [v for v in INPUT_ML + OUTPUT_ML if v in wide.columns]
coverage_pct = [(wide[v].notna().sum() / n_papers) * 100 for v in vars_to_show]
bar_colors   = [COL_BLUE if v in INPUT_ML else COL_RED for v in vars_to_show]

display_labels = [
    v.replace("_wt%", " (wt%)")
     .replace("_rpm", " (rpm)")
     .replace("_MPa", " (MPa)")
     .replace("_mL",  " (mL)")
     .replace("_C",   " (°C)")
     .replace("_",    " ")
    for v in vars_to_show
]

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = range(len(vars_to_show))
bars  = ax.barh(list(y_pos), coverage_pct,
                color=bar_colors, edgecolor="white", linewidth=0.4)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(display_labels, fontsize=9)
ax.set_xlabel(f"Coverage across {n_papers}-paper corpus (%)", fontsize=10)
ax.set_title("Variable Coverage in the Meta-Dataset",
             fontsize=13, fontweight="bold", pad=12)

# 50% reference line
ax.axvline(x=50, color=COL_GREY, linestyle=":", linewidth=1.2, label="50% coverage")
ax.set_xlim(0, 120)

# Value labels on bars
for bar, val in zip(bars, coverage_pct):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}%", va="center", fontsize=8, color=COL_GREY)

# Separator line between inputs and outputs
n_inputs = sum(1 for v in vars_to_show if v in INPUT_ML)
ax.axhline(y=n_inputs - 0.5, color=COL_GREY, linestyle="-", linewidth=0.8, alpha=0.4)

# Legend
legend_elements = [
    Patch(facecolor=COL_BLUE, label="Input features"),
    Patch(facecolor=COL_RED,  label="Output targets"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="lower right", frameon=False)

fig.tight_layout()

out3 = os.path.join(FIGURES_DIR, "fig3_coverage.png")
fig.savefig(out3, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Variables plotted: {len(vars_to_show)}")
print(f"  → Saved: {out3}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("All figures saved to:")
print(f"  {FIGURES_DIR}")
print("=" * 60)
print(f"  fig1_actual_vs_predicted.png")
print(f"  fig2_feature_importance.png")
print(f"  fig3_coverage.png")
print("\nDone.")
