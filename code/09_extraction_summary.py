"""
09_extraction_summary.py
========================
Generates additional figures for Chapter 5 (Results):

  Figure 4 — Heatmap: papers × variables (presence / absence)
  Figure 5 — Extraction method contribution (NLP vs Table vs OCR)
  Figure 6 — Value distributions for key variables (viscosity, particle size)

All figures saved to: D:\\Monisha\\UCC Project\\outputs\\figures\\

Run after 01, 03, 04, 05 (needs extended_with_ocr.xlsx).
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
os.makedirs(FIGURES_DIR, exist_ok=True)

EXTRACTED_XLSX = os.path.join(OUTPUT_DIR, "extended_with_ocr.xlsx")

# Variables to show in heatmap — inputs + ML targets
HEATMAP_VARS = [
    # Input features
    "Temperature_C",
    "Homogenization_rpm",
    "Pressure_MPa",
    "Fat_concentration_wt%",
    "Concentration_wt%",
    "Protein_type",
    "Oil_Fat_type",
    "Volume_mL",
    # Output targets
    "Viscosity_Pa_s",
    "d90_µm",
    "Sensory_thickness",
    "Sensory_creaminess",
    "Friction_coefficient",
]

INPUT_VARS = [
    "Temperature_C", "Homogenization_rpm", "Pressure_MPa",
    "Fat_concentration_wt%", "Concentration_wt%",
    "Protein_type", "Oil_Fat_type", "Volume_mL",
]

OUTPUT_VARS = [
    "Viscosity_Pa_s", "d90_µm",
    "Sensory_thickness", "Sensory_creaminess", "Friction_coefficient",
]

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

COL_BLUE  = "#2166ac"
COL_RED   = "#d6604d"
COL_GREEN = "#1a9850"
COL_GREY  = "#888888"
COL_LIGHT = "#f7f7f7"

# ── Load data ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading extracted data...")
print("=" * 60)

xl     = pd.ExcelFile(EXTRACTED_XLSX)
frames = []
for sheet in xl.sheet_names:
    df = pd.read_excel(xl, sheet_name=sheet)
    df["Sheet"] = sheet          # track which sheet each row came from
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
all_df["Value"] = pd.to_numeric(all_df["Value"], errors="coerce")
all_df["Normalized_Variable"] = all_df["Normalized_Variable"].astype(str)

print(f"  Total rows : {len(all_df):,}")
print(f"  Papers     : {all_df['Paper'].nunique()}")
print(f"  Sheets     : {len(xl.sheet_names)}")

# Wide format for heatmap
wide = (
    all_df[all_df["Value"].notna()]
    .groupby(["Paper", "Normalized_Variable"])["Value"]
    .median()
    .unstack()
)
wide = wide.drop(columns=["nan"], errors="ignore")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Heatmap: papers × variables
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FIGURE 4: Papers × Variables heatmap")
print("=" * 60)

# Keep only the variables we care about, in order
vars_in_data = [v for v in HEATMAP_VARS if v in wide.columns]
heat_df = wide[vars_in_data].copy()

# Binary presence matrix: 1 = has value, 0 = missing
presence = heat_df.notna().astype(int)

# Short paper labels for y-axis
def short_paper(name):
    name = str(name)
    for prefix in ["1-s2.0-", "S0", "S2"]:
        if name.startswith(prefix):
            name = name.replace("1-s2.0-", "")
            return name[:18].strip("-_ ")
    # Author-year style name
    parts = name.replace("-", " ").split()
    return " ".join(parts[:4])[:28]

paper_labels = [short_paper(p) for p in presence.index]

# Column labels
col_labels = [
    v.replace("_wt%", " (wt%)")
     .replace("_rpm", " (rpm)")
     .replace("_MPa", " (MPa)")
     .replace("_mL",  " (mL)")
     .replace("_C",   " (°C)")
     .replace("_",    " ")
    for v in vars_in_data
]

n_papers = len(presence)
n_vars   = len(vars_in_data)
n_inputs = len([v for v in vars_in_data if v in INPUT_VARS])

fig, ax = plt.subplots(figsize=(13, max(8, n_papers * 0.35)))

# Draw cells manually so inputs and outputs get different colours
for col_i, var in enumerate(vars_in_data):
    for row_i, paper in enumerate(presence.index):
        val = presence.loc[paper, var]
        if val == 1:
            colour = COL_BLUE if var in INPUT_VARS else COL_RED
            alpha  = 0.85
        else:
            colour = "#e8e8e8"
            alpha  = 1.0
        rect = mpatches.FancyBboxPatch(
            (col_i + 0.05, row_i + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=colour, edgecolor="white",
            linewidth=0.5, alpha=alpha,
        )
        ax.add_patch(rect)

# Separator line between inputs and outputs
ax.axvline(x=n_inputs, color=COL_GREY, linestyle="--", linewidth=1.2, alpha=0.6)

ax.set_xlim(0, n_vars)
ax.set_ylim(0, n_papers)
ax.set_xticks([i + 0.5 for i in range(n_vars)])
ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
ax.set_yticks([i + 0.5 for i in range(n_papers)])
ax.set_yticklabels(paper_labels, fontsize=7)
ax.set_title(
    "Meta-Dataset Coverage — Papers × Variables\n"
    "(blue = input feature present · red = output target present · grey = missing)",
    fontsize=11, fontweight="bold", pad=12
)

# Column group labels
ax.text(n_inputs / 2, n_papers + 0.3, "Input Features",
        ha="center", va="bottom", fontsize=9, color=COL_BLUE, fontweight="bold")
ax.text(n_inputs + (n_vars - n_inputs) / 2, n_papers + 0.3, "Output Targets",
        ha="center", va="bottom", fontsize=9, color=COL_RED, fontweight="bold")

ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.tick_params(left=False, bottom=False)

# Coverage % annotation above each column
for col_i, var in enumerate(vars_in_data):
    pct = presence[var].mean() * 100
    ax.text(col_i + 0.5, n_papers + 0.05, f"{pct:.0f}%",
            ha="center", va="bottom", fontsize=6.5, color=COL_GREY)

fig.tight_layout()
out4 = os.path.join(FIGURES_DIR, "fig4_heatmap.png")
fig.savefig(out4, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Papers: {n_papers}  ·  Variables: {n_vars}")
print(f"  → Saved: {out4}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Extraction method contribution
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FIGURE 5: Extraction method breakdown")
print("=" * 60)

# Classify each sheet by method
# Sheet names: NLP sheets = paper names (long),
#              Table sheets end in _table or _ext,
#              OCR sheet = "OCR_extracted"

def classify_sheet(sheet_name):
    s = str(sheet_name).lower()
    if "ocr" in s:
        return "OCR"
    if s.endswith("_table") or s.endswith("ext") or "table" in s:
        return "Table"
    return "NLP"

all_df["Method"] = all_df["Sheet"].apply(classify_sheet)

# Count valid (non-NaN) extractions per method
valid_df = all_df[all_df["Value"].notna()]
method_counts = valid_df.groupby("Method").size()

print(f"  Method breakdown:")
for method, count in method_counts.items():
    pct = count / len(valid_df) * 100
    print(f"    {method:10s}  {count:6,} rows  ({pct:.1f}%)")

# Bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

methods   = list(method_counts.index)
counts    = list(method_counts.values)
colours   = {
    "NLP":   COL_BLUE,
    "Table": COL_GREEN,
    "OCR":   COL_RED,
}
bar_colours = [colours.get(m, COL_GREY) for m in methods]

# Left: absolute counts
bars = ax1.bar(methods, counts, color=bar_colours,
               edgecolor="white", linewidth=0.5, width=0.5)
ax1.set_ylabel("Number of valid extractions")
ax1.set_title("Extractions per Method\n(valid numeric values only)", fontweight="bold")
for bar, val in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
             f"{val:,}", ha="center", fontsize=10, color=COL_GREY)
ax1.set_ylim(0, max(counts) * 1.15)

# Right: papers processed per method
sheets_per_method = all_df.drop_duplicates("Sheet").groupby("Method").size()
papers = [sheets_per_method.get(m, 0) for m in methods]

bars2 = ax2.bar(methods, papers, color=bar_colours,
                edgecolor="white", linewidth=0.5, width=0.5)
ax2.set_ylabel("Number of source sheets processed")
ax2.set_title("Sheets Processed per Method", fontweight="bold")
for bar, val in zip(bars2, papers):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
             str(val), ha="center", fontsize=10, color=COL_GREY)
ax2.set_ylim(0, max(papers) * 1.2)

fig.suptitle("Contribution of Each Extraction Method to the Meta-Dataset",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
out5 = os.path.join(FIGURES_DIR, "fig5_extraction_methods.png")
fig.savefig(out5, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  → Saved: {out5}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Distribution of key variable values across corpus
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FIGURE 6: Value distributions across corpus")
print("=" * 60)

# Variables to show distributions for
DIST_VARS = {
    "Viscosity_Pa_s":    ("Viscosity (Pa·s)",        COL_BLUE),
    "d90_µm":           ("d₉₀ Particle Size (µm)",   COL_RED),
    "Sensory_thickness": ("Sensory Thickness (/10)",  COL_GREEN),
    "Temperature_C":     ("Temperature (°C)",         "#8073ac"),
    "Pressure_MPa":      ("Homogenisation\nPressure (MPa)", "#e08214"),
    "Concentration_wt%": ("Concentration (wt%)",      "#35978f"),
}

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes_flat = axes.flatten()

for i, (var, (label, colour)) in enumerate(DIST_VARS.items()):
    ax = axes_flat[i]

    if var not in wide.columns:
        ax.set_visible(False)
        continue

    values = wide[var].dropna().values

    if len(values) < 3:
        ax.set_visible(False)
        continue

    ax.hist(values, bins=min(10, len(values)),
            color=colour, edgecolor="white", linewidth=0.5, alpha=0.85)

    med = np.median(values)
    ax.axvline(med, color="black", linestyle="--", linewidth=1.5,
               label=f"Median: {med:.2f}")

    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("Number of papers", fontsize=9)
    ax.set_title(f"{label}\n"
                 f"n={len(values)} papers  ·  range: {values.min():.2f}–{values.max():.2f}",
                 fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    print(f"  {var:30s}  n={len(values):2d}  "
          f"median={med:.3f}  min={values.min():.3f}  max={values.max():.3f}")

fig.suptitle("Distribution of Extracted Values Across the Corpus",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout(pad=2.0)
out6 = os.path.join(FIGURES_DIR, "fig6_distributions.png")
fig.savefig(out6, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  → Saved: {out6}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("All figures saved to:")
print(f"  {FIGURES_DIR}")
print("=" * 60)
print("  fig4_heatmap.png")
print("  fig5_extraction_methods.png")
print("  fig6_distributions.png")
print("\nDone.")
