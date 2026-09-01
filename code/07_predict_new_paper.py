"""
07_predict_new_paper.py
=======================
End-to-end demonstration: given a NEW PDF the pipeline has never seen,
  1. Extract formulation and process variables using regex patterns
  2. Load the trained ML models
  3. Predict emulsion output properties
  4. Report side-by-side: extracted inputs → predicted outputs

Usage:
    python 07_predict_new_paper.py "path/to/paper.pdf"

If no path is given, runs on all 7 held-out test papers as a batch demo.
"""

import fitz
import re
import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR  = r"D:\Monisha\UCC Project\outputs"
PAPERS_DIR  = r"D:\Monisha\UCC Project\papers"

# The 7 papers held out from ML training — used as the batch demo
TEST_PAPERS = [
    "J Sci Food Agric - 1999 - Moore - Effect of emulsifier type on sensory properties of oil‐in‐water emulsions",
    "1-s2.0-S0268005X04001675-main",
    "1-s2.0-S0268005X0500189X-main",
    "1-s2.0-S2665927124001321-main",
    "1-s2.0-S0260877412000362-main",
    "1-s2.0-S0963996911002845-main",
    "1-s2.0-S0268005X10002304-main",
]

# Features the ML models were trained on — must match 02_ml_model.py INPUT_VARS
FEATURE_ORDER = [
    "Temperature_C",
    "Homogenization_rpm",
    "Pressure_MPa",
    "Fat_concentration_wt%",
    "Concentration_wt%",
    "Protein_type",
    "Oil_Fat_type",
    "Volume_mL",
]

TARGETS = [
    "Viscosity_Pa_s",
    "d90_µm",
    "Sensory_thickness",
    "Sensory_creaminess",
    "Friction_coefficient",
]

TARGET_UNITS = {
    "Viscosity_Pa_s":      "Pa·s",
    "d90_µm":             "µm",
    "Sensory_thickness":   "/10",
    "Sensory_creaminess":  "/10",
    "Friction_coefficient":"—",
}

# ── Regex patterns for feature extraction ─────────────────────────────────────
# Each tuple: (feature_name, compiled_pattern, value_group_index)
EXTRACT_PATTERNS = [
    ("Temperature_C",        re.compile(r'(\d+\.?\d*)\s*°C',            re.I), 1),
    ("Homogenization_rpm",   re.compile(r'(\d[\d,]*)\s*rpm',            re.I), 1),
    ("Pressure_MPa",         re.compile(r'(\d+\.?\d*)\s*MPa',           re.I), 1),
    ("Fat_concentration_wt%",re.compile(r'(\d+\.?\d*)\s*wt\.?%',        re.I), 1),
    ("Concentration_wt%",    re.compile(r'(\d+\.?\d*)\s*%\b',           re.I), 1),
    ("Volume_mL",            re.compile(r'(\d+\.?\d*)\s*mL',            re.I), 1),
]

# Keyword detection for categorical features
PROTEIN_KEYWORDS = [
    "whey protein", "whey", "casein", "sodium caseinate", "soy protein",
    "pea protein", "milk protein", "β-lactoglobulin", "lactalbumin",
    "ovalbumin", "gelatin", "protein",
]
OIL_KEYWORDS = [
    "sunflower oil", "rapeseed oil", "canola oil", "olive oil", "soybean oil",
    "corn oil", "fish oil", "palm oil", "MCT oil", "triglyceride", "oil", "fat",
    "lipid",
]

def extract_features_from_text(text):
    """
    Extract ML input features from raw PDF text using regex and keyword matching.
    Returns a dict: feature_name → median extracted value (or None).
    """
    features = {f: [] for f in FEATURE_ORDER}

    # Numerical features via regex
    for feat, pattern, grp in EXTRACT_PATTERNS:
        for match in pattern.finditer(text):
            try:
                val = float(match.group(grp).replace(",", ""))
                features[feat].append(val)
            except (ValueError, IndexError):
                continue

    # Protein_type: binary — 1 if paper mentions protein, 0 otherwise
    text_lower = text.lower()
    features["Protein_type"] = [
        1.0 if any(kw in text_lower for kw in PROTEIN_KEYWORDS) else 0.0
    ]

    # Oil_Fat_type: binary — 1 if paper mentions oil/fat, 0 otherwise
    features["Oil_Fat_type"] = [
        1.0 if any(kw in text_lower for kw in OIL_KEYWORDS) else 0.0
    ]

    # Take median of each feature list
    result = {}
    for feat, vals in features.items():
        result[feat] = float(np.median(vals)) if vals else np.nan

    return result


def read_pdf_text(pdf_path):
    """Extract all selectable text from a PDF."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    return "\n".join(pages)


def load_model(target):
    """Load a saved model from disk. Returns None if not found."""
    model_path = os.path.join(OUTPUT_DIR, f"model_{target}.pkl")
    if not os.path.exists(model_path):
        return None, None
    pkg = joblib.load(model_path)
    return pkg["model"], pkg["feature_cols"]


def predict_paper(pdf_path, paper_name=None):
    """
    Full pipeline for one paper:
      PDF → text → features → predictions
    Returns (features_dict, predictions_dict)
    """
    if paper_name is None:
        paper_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Step 1: Read PDF
    text = read_pdf_text(pdf_path)

    # Step 2: Extract features
    features = extract_features_from_text(text)

    # Step 3: Predict each target
    predictions = {}
    for target in TARGETS:
        model, feat_cols = load_model(target)
        if model is None:
            predictions[target] = None
            continue

        # Build feature vector in the order the model was trained on
        x = np.array([[
            features.get(f, np.nan) for f in feat_cols
        ]])

        # Fill any missing features with 0 (same as training median fallback)
        x = np.nan_to_num(x, nan=0.0)

        log_pred = model.predict(x)[0]
        predictions[target] = float(np.expm1(log_pred))

    return features, predictions


def print_paper_report(paper_name, features, predictions, actual=None):
    """Print a formatted extraction + prediction report for one paper."""
    print("\n" + "=" * 65)
    print(f"PAPER: {paper_name[:62]}")
    print("=" * 65)

    print("\n  ── Extracted input features ──────────────────────────────")
    for feat in FEATURE_ORDER:
        val = features.get(feat, np.nan)
        if np.isnan(val):
            print(f"    {feat:<35}  [not found in text]")
        else:
            print(f"    {feat:<35}  {val:.4g}")

    print("\n  ── Predicted output properties ───────────────────────────")
    print(f"    {'Target':<30} {'Predicted':>12}  {'Actual':>12}  {'Error':>10}")
    print("    " + "-" * 68)
    for target in TARGETS:
        pred = predictions.get(target)
        unit = TARGET_UNITS.get(target, "")
        if pred is None:
            pred_str = "  model missing"
        else:
            pred_str = f"{pred:>10.3f} {unit}"

        if actual and target in actual:
            act_val  = actual[target]
            act_str  = f"{act_val:>10.3f} {unit}"
            if pred is not None:
                err_str = f"{abs(act_val - pred):>9.3f}"
            else:
                err_str = "         —"
        else:
            act_str = "      —"
            err_str = "         —"

        print(f"    {target:<30} {pred_str:>14}  {act_str:>14}  {err_str}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Load the previously extracted dataset so we can show actual vs predicted
    print("Loading extracted dataset for actual-value lookup...")
    try:
        xl = pd.ExcelFile(
            r"D:\Monisha\UCC Project\outputs\extended_with_ocr.xlsx"
        )
        frames = [pd.read_excel(xl, sheet_name=s) for s in xl.sheet_names]
        all_df = pd.concat(frames, ignore_index=True)
        all_df["Value"] = pd.to_numeric(all_df["Value"], errors="coerce")

        wide = (
            all_df[all_df["Value"].notna()]
            .groupby(["Paper", "Normalized_Variable"])["Value"]
            .median()
            .unstack()
        )
        wide = wide.drop(columns=["nan"], errors="ignore")

        # Sensory normalisation
        for col in ["Sensory_thickness", "Sensory_creaminess"]:
            if col in wide.columns:
                wide[col] = wide[col].apply(
                    lambda v: round(v / 10, 3) if pd.notna(v) and v > 10 else v
                )
        have_actuals = True
    except Exception as e:
        print(f"  Could not load actuals: {e}")
        wide = None
        have_actuals = False

    # Determine which papers to run
    if len(sys.argv) > 1:
        # Single paper passed as argument
        pdf_path   = sys.argv[1]
        paper_name = os.path.splitext(os.path.basename(pdf_path))[0]
        papers_to_run = [(pdf_path, paper_name)]
    else:
        # Batch demo on the 7 held-out test papers
        print("No PDF path given — running batch demo on 7 held-out test papers.\n")
        papers_to_run = []
        for pname in TEST_PAPERS:
            pdf_path = os.path.join(PAPERS_DIR, pname + ".pdf")
            if os.path.exists(pdf_path):
                papers_to_run.append((pdf_path, pname))
            else:
                print(f"  [SKIP] Not found: {pname}.pdf")

    print(f"\nRunning predictions for {len(papers_to_run)} paper(s)...\n")

    all_rows = []

    for pdf_path, paper_name in papers_to_run:
        # Actual values from extracted dataset
        actual = None
        if have_actuals and paper_name in wide.index:
            actual = wide.loc[paper_name, [t for t in TARGETS if t in wide.columns]].to_dict()

        features, predictions = predict_paper(pdf_path, paper_name)
        print_paper_report(paper_name, features, predictions, actual)

        # Collect for summary table
        row = {"Paper": paper_name}
        for f, v in features.items():
            row[f"feat_{f}"] = round(v, 4) if not np.isnan(v) else None
        for t, p in predictions.items():
            row[f"pred_{t}"] = round(p, 4) if p is not None else None
            if actual and t in actual:
                row[f"actual_{t}"] = round(actual[t], 4)
        all_rows.append(row)

    # ── Save summary to Excel ─────────────────────────────────────────────────
    if all_rows:
        out_path = os.path.join(OUTPUT_DIR, "new_paper_predictions.xlsx")
        pd.DataFrame(all_rows).to_excel(out_path, index=False)
        print(f"\n\nSummary saved to: {out_path}")

    print("\nDone.")
