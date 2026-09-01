import fitz
import pandas as pd
import numpy as np
import re
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────────────

TABLES_DIR   = r"D:\Monisha\UCC Project\outputs\tables"
NLP_XLSX     = r"D:\Monisha\UCC Project\extracted_all_papers.xlsx"
OUT_TABLES   = r"D:\Monisha\UCC Project\outputs\table_extracted_combined.xlsx"
OUT_EXTENDED = r"D:\Monisha\UCC Project\outputs\extended_extracted_all_papers.xlsx"

VAN_AKEN_PDF = r"D:\Monisha\UCC Project\papers\1-s2.0-S0268005X10002304-main.pdf"

# ══════════════════════════════════════════════════════════════════════════════
# PAPER 1 — Mirhosseini 2007  (already extracted, just load the CSV)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("Paper 1: Mirhosseini 2007 — loading clean_dataset.csv")
print("=" * 65)

mirhosseini_df = pd.read_csv(
    os.path.join(TABLES_DIR, "Mirhosseini2007_clean_dataset.csv")
)
print(f"  Rows: {len(mirhosseini_df)}  Columns: {list(mirhosseini_df.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# PAPER 2 — van Aken 2011  (S0268005X10002304)
# Table 2: viscosity levels A–D × 4 oil types × 7 measured columns
# Columns: v@9.5s-1, [targeted/label], v@50s-1, v@500s-1, STI, D32, D43
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("Paper 2: van Aken 2011 — extracting Table 2 from page 4")
print("=" * 65)

def extract_van_aken_table(pdf_path):
    doc  = fitz.open(pdf_path)
    text = doc[3].get_text("text")   # page index 3 = page 4

    oil_types = {
        "2% mct":    (2,  "MCT"),
        "20% mct":   (20, "MCT"),
        "20% olive": (20, "Olive"),
        "20% castor":(20, "Castor"),
    }

    rows = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    level = None
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect viscosity level header (single letter A–D)
        if re.fullmatch(r"[A-D]", line):
            level = line
            i += 1
            continue

        # Detect oil type line
        oil_key = next((k for k in oil_types if k in line.lower()), None)

        if oil_key and level:
            fat_conc, oil_type = oil_types[oil_key]
            # Scan ahead for numbers and labels
            nums = []
            j = i + 1
            while j < len(lines) and len(nums) < 7:
                val = lines[j]
                val_clean = val.replace(",", ".")
                if re.fullmatch(r"\d+[a-z]", val):
                    # Statistical label (e.g. "1a", "2a") — skip, no target for level A
                    j += 1
                    continue
                try:
                    nums.append(float(val_clean))
                    j += 1
                except ValueError:
                    break   # stop if we hit non-numeric text

            if len(nums) >= 6:
                rows.append({
                    "paper":                "van_Aken_2011",
                    "viscosity_level":      level,
                    "oil_type":             oil_type,
                    "fat_concentration_pct": fat_conc,
                    "viscosity_9.5s-1_mPas":  nums[0],
                    "targeted_viscosity_mPas": nums[1] if len(nums) == 7 else None,
                    "viscosity_50s-1_mPas":    nums[-4],
                    "viscosity_500s-1_mPas":   nums[-3],
                    "shear_thinning_index":    nums[-2],
                    "D32_µm":                  nums[-1] if len(nums) >= 6 else None,
                    "D43_µm":                  None,  # will be filled next
                })
            i = j
        else:
            i += 1

    df = pd.DataFrame(rows)
    return df


van_aken_df = extract_van_aken_table(VAN_AKEN_PDF)
print(f"  Rows extracted: {len(van_aken_df)}")
print(van_aken_df.to_string(index=False))

# ── The D43 got shifted — re-parse more carefully using known row count ────────
# Simpler approach: read all numerics from Table 2 text block and assign by position

def extract_van_aken_table_v2(pdf_path):
    """
    Table 2 has exactly 16 data rows (4 levels × 4 oils).
    After the table header, each row = oil_type line + 6–7 numeric lines.
    Level A has 6 numerics (no targeted); B/C/D have 7 (including targeted).
    """
    doc  = fitz.open(pdf_path)
    text = doc[3].get_text("text")

    oil_map = [
        ("2% MCT",    2,  "MCT"),
        ("20% MCT",   20, "MCT"),
        ("20% Olive", 20, "Olive"),
        ("20% Castor",20, "Castor"),
    ]

    rows   = []
    lines  = [l.strip() for l in text.split("\n") if l.strip()]
    level  = None

    i = 0
    while i < len(lines):
        line = lines[i]

        if re.fullmatch(r"[A-D]", line):
            level = line; i += 1; continue

        matched_oil = next(
            ((desc, fc, ot) for desc, fc, ot in oil_map
             if desc.lower() in line.lower()),
            None
        )

        if matched_oil and level:
            desc, fat_conc, oil_type = matched_oil
            nums_raw = []   # (value_or_none)  — None for skipped labels
            j = i + 1

            # Collect until we have 6 numeric values
            while j < len(lines) and len([x for x in nums_raw if x is not None]) < 7:
                v = lines[j].replace(",", ".")
                if re.fullmatch(r"\d+[a-z]+", lines[j]):
                    nums_raw.append(None)   # placeholder for label slot
                    j += 1
                    continue
                try:
                    nums_raw.append(float(v))
                    j += 1
                except ValueError:
                    break

            nums = [x for x in nums_raw if x is not None]

            if len(nums) >= 6:
                # For level A: no targeted → [v9.5, v50, v500, STI, D32, D43]
                # For B/C/D:   targeted present → [v9.5, targeted, v50, v500, STI, D32, D43]
                has_target = (None in nums_raw) is False and len(nums) == 7
                if has_target:
                    v95, targeted, v50, v500, sti, d32, d43 = nums
                else:
                    v95 = nums[0]
                    targeted = None
                    v50, v500, sti, d32, d43 = nums[1:6]

                rows.append({
                    "paper":                 "van_Aken_2011",
                    "viscosity_level":       level,
                    "oil_type":              oil_type,
                    "fat_concentration_pct": fat_conc,
                    "viscosity_9.5s-1_mPas": v95,
                    "targeted_mPas":         targeted,
                    "viscosity_50s-1_mPas":  v50,
                    "viscosity_500s-1_mPas": v500,
                    "shear_thinning_index":  sti,
                    "D32_µm":                d32,
                    "D43_µm":                d43,
                })
            i = j
        else:
            i += 1

    return pd.DataFrame(rows)

print("\nRe-parsing with improved column alignment...")
van_aken_df = extract_van_aken_table_v2(VAN_AKEN_PDF)
print(f"  Rows extracted: {len(van_aken_df)}")
print(van_aken_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Save both tables to one Excel file (different sheets)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("Saving combined Excel: table_extracted_combined.xlsx")
print("=" * 65)

with pd.ExcelWriter(OUT_TABLES, engine="openpyxl") as writer:
    mirhosseini_df.to_excel(writer, sheet_name="Mirhosseini2007", index=False)
    van_aken_df.to_excel(writer,    sheet_name="van_Aken2011",    index=False)

print(f"  Sheet 'Mirhosseini2007' : {len(mirhosseini_df)} rows")
print(f"  Sheet 'van_Aken2011'    : {len(van_aken_df)} rows")
print(f"  Saved to: {OUT_TABLES}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Convert table data to NLP long-format and append to NLP extraction
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("STEP 2: Converting table data to NLP long-format")
print("=" * 65)

def mirhosseini_to_long(df):
    """
    Map Mirhosseini2007 table columns to NLP normalized variable names.
    Viscosity is in mPa.s → convert to Pa.s (/1000).
    """
    rows = []
    for _, r in df.iterrows():
        paper = "Mirhosseini2007_table"
        pairs = [
            ("Concentration_wt%", r["arabic_gum_pct"],            "wt%",  "table"),
            ("Concentration_wt%", r["xanthan_gum_pct"],           "wt%",  "table"),
            ("Oil_concentration_wt%", r["orange_oil_pct"],        "wt%",  "table"),
            ("Viscosity_Pa_s",    r["viscosity_60rpm_exp"] / 1000,"Pa.s", "table"),
        ]
        for norm_var, val, unit, method in pairs:
            if pd.notna(val):
                rows.append({
                    "Paper":               paper,
                    "Normalized_Variable": norm_var,
                    "Value":               val,
                    "Unit":                unit,
                    "Relation_Method":     method,
                    "Confidence":          0.95,
                })
    return pd.DataFrame(rows)


def van_aken_to_long(df):
    """
    Map van Aken 2011 table columns to NLP normalized variable names.
    Viscosity in mPa.s → Pa.s (/1000).  D43 ≈ d90.
    Oil types encoded: MCT → Fat_concentration_wt% + Oil_Fat_type.
    """
    rows = []
    for _, r in df.iterrows():
        paper = "van_Aken_2011_table"
        v50   = r.get("viscosity_50s-1_mPas")
        d43   = r.get("D43_µm")
        fat   = r.get("fat_concentration_pct")

        pairs = [
            ("Fat_concentration_wt%", fat,                       "wt%",  "table"),
            ("Viscosity_Pa_s",        v50 / 1000 if pd.notna(v50) else None, "Pa.s", "table"),
            ("d90_µm",                d43,                       "µm",   "table"),
        ]
        for norm_var, val, unit, method in pairs:
            if val is not None and pd.notna(val):
                rows.append({
                    "Paper":               paper,
                    "Normalized_Variable": norm_var,
                    "Value":               val,
                    "Unit":                unit,
                    "Relation_Method":     method,
                    "Confidence":          0.95,
                })
    return pd.DataFrame(rows)


m_long = mirhosseini_to_long(mirhosseini_df)
v_long = van_aken_to_long(van_aken_df)

print(f"  Mirhosseini long rows: {len(m_long)}")
print(f"  van Aken long rows:    {len(v_long)}")

# Load NLP extracted data
print("\n  Loading NLP extracted data...")
xl     = pd.ExcelFile(NLP_XLSX)
frames = [pd.read_excel(xl, sheet_name=s) for s in xl.sheet_names]
nlp_df = pd.concat(frames, ignore_index=True)
print(f"  NLP rows (original): {len(nlp_df):,}")

# Append table-extracted rows
table_long = pd.concat([m_long, v_long], ignore_index=True)
extended   = pd.concat([nlp_df, table_long], ignore_index=True)
print(f"  Extended rows:       {len(extended):,}  (+{len(table_long)} from tables)")

# Save extended extraction
with pd.ExcelWriter(OUT_EXTENDED, engine="openpyxl") as writer:
    # One sheet per paper (NLP) + one sheet for table-extracted
    for sheet in xl.sheet_names:
        df_sheet = pd.read_excel(xl, sheet_name=sheet)
        df_sheet.to_excel(writer, sheet_name=sheet[:31], index=False)
    # Table-extracted rows as two new sheets
    m_long.to_excel(writer, sheet_name="Mirhosseini2007_table", index=False)
    v_long.to_excel(writer, sheet_name="van_Aken2011_table",    index=False)

print(f"\n  Extended Excel saved to: {OUT_EXTENDED}")
print(f"  Sheets: {len(xl.sheet_names)} NLP + 2 table = {len(xl.sheet_names)+2} total")
print("\nDone. Run 02_ml_model.py pointing at extended_extracted_all_papers.xlsx to retrain.")
