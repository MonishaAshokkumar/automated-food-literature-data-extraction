"""
10_ground_truth_evaluation.py
==============================
Evaluates the accuracy of the extraction pipeline by comparing
extracted values against the manually compiled ExampleData.xlsx
(ground truth compiled by the UCC research team).

Matches papers by DOI -> pipeline filename mapping.
Computes Precision, Recall, F1 and match rate for each variable.

Variables compared:
  d90_µm            ↔  D(90)_(µm)              [same units]
  Viscosity_Pa_s    ↔  median viscosity columns  [mPa.s → Pa.s conversion]
  Sensory_thickness ↔  Mean_thickness_(/10)     [same units]
  Sensory_creaminess↔  Mean_creaminess_(/10)    [same units]
  Temperature_C     ↔  Sample_preparation_temperature_(°C) [same units]

Saves Figure 7: Precision / Recall / F1 bar chart.
Saves evaluation_results.xlsx with per-paper comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ── Configuration ──────────────────────────────────────────────────────────────
OUTPUT_DIR  = r"D:\Monisha\UCC Project\outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
GT_PATH     = r"D:\Monisha\UCC Project\ExampleData.xlsx"
PIPELINE_XL = os.path.join(OUTPUT_DIR, "extended_with_ocr.xlsx")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Tolerance for a value to count as "correctly extracted"
RELATIVE_TOLERANCE = 0.20   # 20% relative error allowed
SENSORY_ABS_TOL    = 1.0    # ±1 point on /10 scale (absolute)

# ── Study_ID → pipeline paper filename mapping (derived from DOI matching) ─────
STUDY_TO_PAPER = {
    "Akhtar2005":      "1-s2.0-S0268005X04001675-main",
    "Akhtar2006":      "1-s2.0-S0268005X0500189X-main",
    "Araiza2024":      "1-s2.0-S2665927124001321-main",
    "Arancibia2011":   "1-s2.0-S0963996911002845-main",
    "Bak2024":         "s11483-024-09844-8",
    "Buffo2001":       "1-s2.0-S0268005X00000503-main",
    "Chojnicka2012":   "1-s2.0-S0958694612000969-main",
    "Chung2013":       "1-s2.0-S0268005X13000234-main",
    "Floury2000":      "1-s2.0-S1466856400000126-main",
    "Kasprzak2023":    "foods-12-02288",
    "Kuhn2012":        "1-s2.0-S0260877412000362-main",
    "Martin2009":      "modeling-the-relationship-between-the-main-emulsion-components-and-stability-viscosity-fluid-behavior-ζ-potential-and",
    "Matsuyama2021":   "1-s2.0-S0268005X20308067-main",
    "Mirhosseini2009": "1-s2.0-S0268005X08000428-main",
    "Moore1998":       "J Sci Food Agric - 1999 - Moore - Effect of emulsifier type on sensory properties of oil\u2010in\u2010water emulsions",
    "Pal1998":         "1-s2.0-S0927775797003749-main",
    "Schadle2022":     "foods-11-00820-v2",
    "Souza2017":       "s11483-017-9469-4",
    "Taherian2006":    "1-s2.0-S0260877405005364-main",
    "Umana2022":       "foods-11-03750-v2",
    "Vingerhoeds2008": "1-s2.0-S0268005X07000689-mainext",
    "Wang2021":        "foods-10-03024",
    "Zhang2024":       "1-s2.0-S0260877423004624-main",
    "vanAken2011":     "1-s2.0-S0268005X10002304-main",
}

# ── Variable comparison spec ───────────────────────────────────────────────────
# (pipeline_variable, gt_column(s), unit_conversion_factor, tolerance_type)
# tolerance_type: "relative" = RELATIVE_TOLERANCE  |  "absolute" = SENSORY_ABS_TOL
VAR_SPEC = [
    ("d90_µm",             ["D(90)_(µm)"],
     1.0,        "relative"),

    ("Viscosity_Pa_s",     ["Viscosity_at_0.1s-1_(mPa.s)",
                             "Viscosity_at_1s-1_(mPa.s)",
                             "Viscosity_at_10s-1_(mPa.s)",
                             "Viscosity_at_50s-1_(mPa.s)",
                             "Viscosity_at_100s-1_(mPa.s)",
                             "Viscosity__at_100rpm_(mPa.s)",
                             "Viscosity__at_60rpm_(mPa.s)"],
     0.001,      "relative"),    # mPa.s → Pa.s (÷ 1000)

    ("Sensory_thickness",  ["Mean_thickness_(/10)"],
     1.0,        "absolute"),

    ("Sensory_creaminess", ["Mean_creaminess_(/10)"],
     1.0,        "absolute"),

    ("Temperature_C",      ["Sample_preparation_temperature_(°C)",
                             "Viscosity_measurement_temperature_(°C)"],
     1.0,        "relative"),
]

# Colours
COL_BLUE  = "#2166ac"
COL_RED   = "#d6604d"
COL_GREEN = "#1a9850"
COL_GREY  = "#555555"

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load ground truth (ExampleData)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading ground truth (ExampleData.xlsx)")
print("=" * 60)

raw = pd.read_excel(GT_PATH, header=None)
gt  = raw.iloc[7:].copy()
gt.columns = raw.iloc[6].tolist()
gt  = gt.reset_index(drop=True)

# Convert all output columns to numeric
for col in gt.columns[3:]:
    gt[col] = pd.to_numeric(gt[col], errors="coerce")

# Per-study medians
gt_median = gt.groupby("Study_ID").median(numeric_only=True)

print(f"  Studies in ground truth : {gt_median.shape[0]}")
print(f"  Matched to pipeline     : {len(STUDY_TO_PAPER)}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load pipeline extracted data (wide format)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Loading pipeline extracted data")
print("=" * 60)

xl     = pd.ExcelFile(PIPELINE_XL)
frames = [pd.read_excel(xl, sheet_name=s) for s in xl.sheet_names]
all_df = pd.concat(frames, ignore_index=True)
all_df["Value"] = pd.to_numeric(all_df["Value"], errors="coerce")

pipeline_wide = (
    all_df[all_df["Value"].notna()]
    .groupby(["Paper", "Normalized_Variable"])["Value"]
    .median()
    .unstack()
)
pipeline_wide = pipeline_wide.drop(columns=["nan"], errors="ignore")

# Sensory normalisation
for col in ["Sensory_thickness", "Sensory_creaminess"]:
    if col in pipeline_wide.columns:
        pipeline_wide[col] = pipeline_wide[col].apply(
            lambda v: round(v / 10, 3) if pd.notna(v) and v > 10 else v
        )

print(f"  Pipeline papers : {pipeline_wide.shape[0]}")
print(f"  Pipeline vars   : {pipeline_wide.shape[1]}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Compare per variable
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Comparing extracted values vs ground truth")
print("=" * 60)

results      = []   # one row per variable
detail_rows  = []   # one row per paper × variable

for pipe_var, gt_cols, factor, tol_type in VAR_SPEC:

    if pipe_var not in pipeline_wide.columns:
        print(f"\n  [{pipe_var}] not in pipeline — skipping")
        continue

    # Ground truth column(s) — take median across shear rates/conditions
    gt_avail = [c for c in gt_cols if c in gt_median.columns]
    if not gt_avail:
        print(f"\n  [{pipe_var}] no matching GT column — skipping")
        continue

    gt_series = gt_median[gt_avail].median(axis=1) * factor

    TP = FP = FN = 0
    errors    = []
    rel_errs  = []
    matched_papers = []

    print(f"\n  {pipe_var}  (GT cols: {gt_avail}  ×{factor})")
    print(f"  {'Study_ID':20s}  {'GT':>10}  {'Pipeline':>10}  {'Err%':>8}  Match")
    print("  " + "-" * 60)

    for study_id, paper_name in STUDY_TO_PAPER.items():
        gt_val   = gt_series.get(study_id, np.nan)
        pipe_val = pipeline_wide.loc[paper_name, pipe_var] \
                   if paper_name in pipeline_wide.index else np.nan

        # ── Classify ────────────────────────────────────────────────────────
        if pd.isna(gt_val):
            # Ground truth has no value → can't evaluate
            continue

        if pd.isna(pipe_val):
            # GT has value, pipeline missed it → False Negative
            FN += 1
            match = "MISS"
            err_pct = np.nan
        else:
            abs_err = abs(pipe_val - gt_val)
            err_pct = abs_err / abs(gt_val) * 100 if gt_val != 0 else np.nan

            if tol_type == "relative":
                hit = (abs_err / abs(gt_val)) <= RELATIVE_TOLERANCE if gt_val != 0 else False
            else:   # absolute
                hit = abs_err <= SENSORY_ABS_TOL

            if hit:
                TP += 1
                match = "✓"
            else:
                FP += 1
                match = "✗"

            errors.append(abs_err)
            if err_pct is not np.nan:
                rel_errs.append(err_pct)

        matched_papers.append(study_id)
        print(f"  {study_id:20s}  {gt_val if not pd.isna(gt_val) else '—':>10.3f}"
              f"  {pipe_val if not pd.isna(pipe_val) else '—':>10.3f}"
              f"  {err_pct:>7.1f}%  {match}"
              if not pd.isna(gt_val) and not pd.isna(pipe_val)
              else
              f"  {study_id:20s}  {gt_val if not pd.isna(gt_val) else '—':>10}  {'—':>10}  {'—':>8}  {match}")

        detail_rows.append({
            "Variable":   pipe_var,
            "Study_ID":   study_id,
            "GT_value":   round(gt_val, 4)   if not pd.isna(gt_val)   else None,
            "Pipeline_value": round(pipe_val, 4) if not pd.isna(pipe_val) else None,
            "Abs_error":  round(abs_err, 4)  if not pd.isna(pipe_val) else None,
            "Rel_error_%": round(err_pct, 2) if err_pct is not np.nan else None,
            "Match":      match,
        })

    # ── Metrics ─────────────────────────────────────────────────────────────
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0.0
    mae       = np.mean(errors)  if errors  else np.nan
    med_rel   = np.median(rel_errs) if rel_errs else np.nan

    print(f"\n  → TP={TP}  FP={FP}  FN={FN}")
    print(f"     Precision={precision:.2f}  Recall={recall:.2f}  F1={f1:.2f}")
    print(f"     MAE={mae:.3f}  Median relative error={med_rel:.1f}%")

    results.append({
        "Variable":         pipe_var,
        "TP": TP, "FP": FP, "FN": FN,
        "Precision":        round(precision, 3),
        "Recall":           round(recall, 3),
        "F1":               round(f1, 3),
        "MAE":              round(mae, 4) if not np.isnan(mae) else None,
        "Median_rel_err_%": round(med_rel, 1) if not np.isnan(med_rel) else None,
        "Tolerance":        f"±{int(RELATIVE_TOLERANCE*100)}% rel" if tol_type == "relative"
                            else f"±{SENSORY_ABS_TOL} abs",
    })


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Save results to Excel
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Saving results")
print("=" * 60)

results_df = pd.DataFrame(results)
detail_df  = pd.DataFrame(detail_rows)

out_xl = os.path.join(OUTPUT_DIR, "evaluation_results.xlsx")
with pd.ExcelWriter(out_xl, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="Summary",    index=False)
    detail_df.to_excel( writer, sheet_name="Per_Paper",  index=False)

print(f"  Saved: {out_xl}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Figure 7: Precision / Recall / F1 bar chart
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Figure 7 — Evaluation metrics bar chart")
print("=" * 60)

if results_df.empty:
    print("  No results to plot.")
else:
    vars_plot = results_df["Variable"].tolist()
    x         = np.arange(len(vars_plot))
    width     = 0.25

    short_labels = [
        v.replace("_Pa_s", "\n(Pa·s)")
         .replace("Sensory_", "Sensory\n")
         .replace("_C", "\n(°C)")
         .replace("_µm", "\n(µm)")
        for v in vars_plot
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars_p = ax.bar(x - width, results_df["Precision"], width,
                    label="Precision", color=COL_BLUE,
                    edgecolor="white", linewidth=0.5)
    bars_r = ax.bar(x,          results_df["Recall"],    width,
                    label="Recall",    color=COL_GREEN,
                    edgecolor="white", linewidth=0.5)
    bars_f = ax.bar(x + width, results_df["F1"],         width,
                    label="F1",        color=COL_RED,
                    edgecolor="white", linewidth=0.5)

    # Value labels
    for bar in list(bars_p) + list(bars_r) + list(bars_f):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=8, color=COL_GREY)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel("Score (0 – 1)", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Extraction Pipeline Evaluation — Precision, Recall, F1\n"
        f"(tolerance: ±{int(RELATIVE_TOLERANCE*100)}% for numeric · ±{SENSORY_ABS_TOL} for sensory)",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.axhline(y=0.5, color=COL_GREY, linestyle=":", linewidth=1, alpha=0.5)
    ax.legend(fontsize=10, frameon=False, loc="upper right")

    fig.tight_layout()
    out7 = os.path.join(FIGURES_DIR, "fig7_evaluation.png")
    fig.savefig(out7, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {out7}")

# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"\n  {'Variable':25s}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'MAE':>8}  {'Med%Err':>8}")
print("  " + "-" * 70)
for _, row in results_df.iterrows():
    mae_str = f"{row['MAE']:.3f}" if row["MAE"] is not None else "  —"
    mer_str = f"{row['Median_rel_err_%']:.1f}%" if row["Median_rel_err_%"] is not None else "  —"
    print(f"  {row['Variable']:25s}  {row['Precision']:>6.2f}  "
          f"{row['Recall']:>6.2f}  {row['F1']:>6.2f}  "
          f"{mae_str:>8}  {mer_str:>8}")

print("\nDone.")
