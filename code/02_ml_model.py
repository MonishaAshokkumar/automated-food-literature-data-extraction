import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import joblib
import warnings
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

EXTRACTED_XLSX = r"D:\Monisha\UCC Project\outputs\extended_with_ocr.xlsx"
OUTPUT_DIR     = r"D:\Monisha\UCC Project\outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 7 test papers chosen at mid-range feature coverage
# (common / representative — not too sparse, not the richest training papers)
TEST_PAPERS = [
    "J Sci Food Agric - 1999 - Moore - Effect of emulsifier type on sensory properties of oil‐in‐water emulsions",
    "1-s2.0-S0268005X04001675-main",
    "1-s2.0-S0268005X0500189X-main",
    "1-s2.0-S2665927124001321-main",
    "1-s2.0-S0260877412000362-main",
    "1-s2.0-S0963996911002845-main",
    "1-s2.0-S0268005X10002304-main",
]

# Input features extracted from text
INPUT_VARS = [
    "Temperature_C",
    "Homogenization_rpm",
    "Pressure_MPa",
    "Fat_concentration_wt%",
    "Concentration_wt%",
    "Protein_type",
    "Oil_Fat_type",
    "Volume_mL",
]

# Target variables — one model per target
TARGETS = {
    "Viscosity_Pa_s":      {"min_papers": 15, "unit": "Pa.s"},
    "d90_µm":             {"min_papers": 15, "unit": "µm"},
    "Sensory_thickness":   {"min_papers": 5,  "unit": "/10"},
    "Sensory_creaminess":  {"min_papers": 5,  "unit": "/10"},
    "Friction_coefficient":{"min_papers": 5,  "unit": "-"},
}

# ── Step 1: Load and pivot extracted data ─────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading NLP-extracted data")
print("=" * 60)

xl     = pd.ExcelFile(EXTRACTED_XLSX)
frames = [pd.read_excel(xl, sheet_name=s) for s in xl.sheet_names]
all_df = pd.concat(frames, ignore_index=True)
all_df["Value"] = pd.to_numeric(all_df["Value"], errors="coerce")
all_df["Normalized_Variable"] = all_df["Normalized_Variable"].astype(str)

print(f"  Total rows:   {len(all_df):,}")
print(f"  Papers:       {all_df['Paper'].nunique()}")

# Pivot: one row per paper, median value per variable
wide = (
    all_df[all_df["Value"].notna()]
    .groupby(["Paper", "Normalized_Variable"])["Value"]
    .median()
    .unstack()
)
wide = wide.drop(columns=["nan"], errors="ignore")

# ── Sensory scale normalisation: unify /100 → /10 ────────────────────────────
# Some papers report sensory scores on a 0–100 scale, others on 0–10.
# Any value > 10 is treated as a /100-scale score and divided by 10.
SENSORY_VARS = ["Sensory_thickness", "Sensory_creaminess", "Sensory_smoothness"]
for col in SENSORY_VARS:
    if col in wide.columns:
        wide[col] = wide[col].apply(
            lambda v: round(v / 10, 3) if pd.notna(v) and v > 10 else v
        )

feature_cols = [c for c in INPUT_VARS if c in wide.columns]

print(f"  Wide format:  {wide.shape[0]} papers x {wide.shape[1]} variables")
print(f"  Features available: {feature_cols}")

# ── Step 2: Train one model per target ────────────────────────────────────────

all_results = []

for target, cfg in TARGETS.items():
    if target not in wide.columns:
        print(f"\n  [{target}] not found in extracted data — skipping")
        continue

    # Papers that have this target variable extracted
    df_t = wide[wide[target].notna()].copy()
    n_papers = len(df_t)

    if n_papers < cfg["min_papers"]:
        print(f"\n  [{target}] only {n_papers} papers — too few for 7/20 split, skipping")
        continue

    print("\n" + "=" * 60)
    print(f"TARGET: {target}  ({n_papers} papers with this variable)")
    print("=" * 60)

    # Fill missing features: NaN means ingredient/condition not mentioned → use median
    df_t[feature_cols] = df_t[feature_cols].apply(lambda col: col.fillna(col.median()))

    X = df_t[feature_cols].values
    y = np.log1p(df_t[target].values)   # log-transform to handle wide value ranges

    # ── Split strategy ────────────────────────────────────────────────────────
    test_mask  = df_t.index.isin(TEST_PAPERS)
    train_mask = ~test_mask

    n_test  = test_mask.sum()
    n_train = train_mask.sum()

    if n_test >= 5 and n_train >= 10:
        # Enough for a proper 7/19 holdout split
        split_method = f"Holdout ({n_train} train / {n_test} test papers)"
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
        y_test_orig = df_t.loc[test_mask,  target].values
        test_papers_used = df_t.index[test_mask].tolist()
    else:
        # Fall back to Leave-One-Out when test papers don't overlap with this target
        split_method = "Leave-One-Out CV (not enough test papers have this target)"
        X_train, y_train = X, y
        X_test,  y_test  = X, y
        y_test_orig = df_t[target].values
        test_papers_used = df_t.index.tolist()

    print(f"  Split: {split_method}")

    # ── Train models ──────────────────────────────────────────────────────────
    models = {
        "Random Forest":     RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                       max_depth=3, random_state=42),
        "Ridge Regression":  Ridge(alpha=10.0),
    }

    model_scores = {}

    for name, model in models.items():
        if split_method.startswith("Holdout"):
            model.fit(X_train, y_train)
            log_preds = model.predict(X_test)
        else:
            log_preds = cross_val_predict(model, X_train, y_train, cv=LeaveOneOut())

        preds   = np.expm1(log_preds)
        actuals = y_test_orig

        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mae  = mean_absolute_error(actuals, preds)
        r2   = r2_score(actuals, preds)
        model_scores[name] = r2

        print(f"\n  {name}:")
        print(f"    RMSE: {rmse:.4f} {cfg['unit']}")
        print(f"    MAE : {mae:.4f} {cfg['unit']}")
        print(f"    R²  : {r2:.3f}")

    # ── Best model — per-paper predictions ────────────────────────────────────
    best_name  = max(model_scores, key=model_scores.get)
    best_model = models[best_name]

    if split_method.startswith("Holdout"):
        best_model.fit(X_train, y_train)
        best_preds = np.expm1(best_model.predict(X_test))
        pred_papers = test_papers_used
        actuals_for_df = y_test_orig
    else:
        log_preds  = cross_val_predict(best_model, X, y, cv=LeaveOneOut())
        best_preds = np.expm1(log_preds)
        pred_papers = df_t.index.tolist()
        actuals_for_df = df_t[target].values

    print(f"\n  Best model: {best_name}  (R²={model_scores[best_name]:.3f})")
    print(f"\n  Per-paper predictions:")

    pred_df = pd.DataFrame({
        "Paper":      pred_papers,
        "Actual":     actuals_for_df,
        "Predicted":  best_preds,
        "Abs_Error":  np.abs(actuals_for_df - best_preds),
    }).sort_values("Abs_Error")

    for _, row in pred_df.iterrows():
        paper_short = str(row["Paper"])[-40:]
        print(f"    {paper_short:40s}  actual={row['Actual']:8.3f}  pred={row['Predicted']:8.3f}  err={row['Abs_Error']:8.3f}")

    # ── Refit on all data + save model for new-paper prediction ─────────────
    best_model.fit(X, y)
    model_path = os.path.join(OUTPUT_DIR, f"model_{target}.pkl")
    joblib.dump({"model": best_model, "feature_cols": feature_cols}, model_path)
    print(f"\n  Model saved: model_{target}.pkl")

    # ── Feature importance ────────────────────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print(f"\n  Feature importance ({best_name}):")
        for feat, imp in fi.items():
            bar = "#" * int(imp * 200)
            print(f"    {imp:.4f}  {bar:<40}  {feat}")

    # ── Save predictions ──────────────────────────────────────────────────────
    pred_df["Target"] = target
    pred_df["Best_Model"] = best_name
    all_results.append(pred_df)

# ── Final summary ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY — Best R² per target")
print("=" * 60)

if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, "ml_extracted_predictions.csv"), index=False)

    for res in all_results:
        t = res["Target"].iloc[0]
        m = res["Best_Model"].iloc[0]
        r2 = r2_score(res["Actual"], res["Predicted"])
        mae = mean_absolute_error(res["Actual"], res["Predicted"])
        print(f"  {t:30s}  R²={r2:6.3f}  MAE={mae:.4f}  ({m})")

print(f"\nAll predictions saved to: {OUTPUT_DIR}/ml_extracted_predictions.csv")
print("\nDone.")
