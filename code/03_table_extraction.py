import fitz          # PyMuPDF
import pdfplumber
import pandas as pd
import re
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ── Configuration ─────────────────────────────────────────────────────────────

PAPER_PATH = r"D:\Monisha\UCC Project\papers\modeling-the-relationship-between-the-main-emulsion-components-and-stability-viscosity-fluid-behavior-ζ-potential-and.pdf"
PAPER_NAME = "Mirhosseini2007"
OUTPUT_DIR = r"D:\Monisha\UCC Project\outputs\tables"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Spatial word-grouping table extractor ─────────────────────────────────────
#
# Academic PDFs often draw table borders as vector graphics — not real PDF line
# objects — so tabula/pdfplumber can't detect them. Instead we:
#   1. Get every word with its (x0, y0, x1, y1) bounding box via PyMuPDF
#   2. Cluster words into "rows" by proximity on the y-axis (row_tol)
#   3. Cluster words in each row into "columns" by x-axis gaps (col_gap)
#   4. Keep only regions that look like data tables (have numeric values)

def words_to_rows(page, row_tol=4):
    """
    Extract all words from a PyMuPDF page and group them into rows.
    row_tol: max y-distance (points) between words considered the same row.
    Returns list of rows, each row is a list of (x0, text) pairs sorted by x.
    """
    words = page.get_text("words")   # (x0, y0, x1, y1, text, block, line, word)
    if not words:
        return []

    # Sort by top-y then left-x
    words = sorted(words, key=lambda w: (round(w[1] / row_tol), w[0]))

    rows = []
    current_row = []
    current_y = None

    for w in words:
        x0, y0, text = w[0], w[1], w[4]
        if current_y is None or abs(y0 - current_y) > row_tol:
            if current_row:
                rows.append(sorted(current_row, key=lambda t: t[0]))
            current_row = [(x0, text)]
            current_y = y0
        else:
            current_row.append((x0, text))

    if current_row:
        rows.append(sorted(current_row, key=lambda t: t[0]))

    return rows


def rows_to_columns(rows, col_gap=20):
    """
    Given rows of (x0, text) pairs, assign each word to a column bucket.
    col_gap: minimum x-gap (points) to start a new column.
    Returns a 2-D list: grid[row_idx][col_idx] = text.
    """
    if not rows:
        return []

    # Collect all x0 positions and cluster into column anchors
    all_x = sorted(set(x for row in rows for x, _ in row))
    col_anchors = []
    for x in all_x:
        if not col_anchors or x - col_anchors[-1] > col_gap:
            col_anchors.append(x)

    def col_idx(x):
        return min(range(len(col_anchors)), key=lambda i: abs(col_anchors[i] - x))

    n_cols = len(col_anchors)
    grid = []
    for row in rows:
        cells = [""] * n_cols
        for x, text in row:
            ci = col_idx(x)
            cells[ci] = (cells[ci] + " " + text).strip()
        grid.append(cells)

    return grid


def is_numeric(s):
    """Return True if the string looks like a number (int, float, negative)."""
    return bool(re.match(r'^-?\d+(\.\d+)?$', str(s).strip()))


def extract_data_blocks(page, row_tol=4, col_gap=20, min_numeric_ratio=0.30, min_cols=3):
    """
    From one PDF page, extract rectangular blocks that look like data tables.
    A block qualifies when >= min_numeric_ratio of its cells are numeric
    and it has >= min_cols columns.
    Returns a list of pandas DataFrames.
    """
    rows = words_to_rows(page, row_tol)
    if not rows:
        return []

    grid = rows_to_columns(rows, col_gap)
    if not grid:
        return []

    # Score each row by numeric content
    def numeric_ratio(row):
        if not row:
            return 0
        return sum(1 for c in row if is_numeric(c)) / len(row)

    # Find contiguous bands of rows with high numeric content
    # (allow 1-2 non-numeric rows for column headers)
    in_table = False
    table_rows = []
    current_block = []
    blank_streak = 0

    for row in grid:
        nr = numeric_ratio(row)
        if nr >= min_numeric_ratio and len([c for c in row if c]) >= min_cols:
            current_block.append(row)
            blank_streak = 0
            in_table = True
        elif in_table and blank_streak < 2:
            current_block.append(row)
            blank_streak += 1
        else:
            if len(current_block) >= 3:
                table_rows.append(current_block)
            current_block = []
            blank_streak = 0
            in_table = False

    if len(current_block) >= 3:
        table_rows.append(current_block)

    results = []
    for block in table_rows:
        # Trim trailing blank-streak rows
        while block and all(c == "" for c in block[-1]):
            block.pop()

        n_cols = max(len(r) for r in block)
        # Pad shorter rows
        block = [r + [""] * (n_cols - len(r)) for r in block]

        df = pd.DataFrame(block)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        df = df.reset_index(drop=True)
        if df.shape[0] >= 2 and df.shape[1] >= min_cols:
            results.append(df)

    return results


# ── Also try pdfplumber lines strategy for pages that have real borders ────────

def extract_pdfplumber_lines(pdf_path):
    """pdfplumber with lines_strict — only works when PDF has actual line objects."""
    results = []
    settings = {
        "vertical_strategy":   "lines_strict",
        "horizontal_strategy": "lines_strict",
        "snap_tolerance":      5,
    }
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables(settings)
                for t_num, raw in enumerate(tables, start=1):
                    if not raw or len(raw) < 2:
                        continue
                    df = pd.DataFrame(raw[1:], columns=raw[0])
                    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
                    if df.shape[0] >= 2 and df.shape[1] >= 2:
                        results.append((page_num, t_num, df))
            except Exception:
                continue
    return results


# ── Main extraction ────────────────────────────────────────────────────────────

print("=" * 65)
print(f"Paper: {PAPER_NAME}")
print("=" * 65)

all_tables = []

print("\n[Attempt 1] pdfplumber lines_strict (bordered tables)...")
line_tables = extract_pdfplumber_lines(PAPER_PATH)
print(f"  Found: {len(line_tables)} tables")
for page_num, t_num, df in line_tables:
    all_tables.append((f"p{page_num}-lines", df))

print("\n[Attempt 2] PyMuPDF spatial word-grouping (layout-based)...")
doc = fitz.open(PAPER_PATH)
for page_num, page in enumerate(doc, start=1):
    blocks = extract_data_blocks(page)
    if blocks:
        print(f"  Page {page_num}: {len(blocks)} data block(s) found")
    for b_idx, df in enumerate(blocks, start=1):
        all_tables.append((f"p{page_num}-spatial-b{b_idx}", df))

print(f"\n  Total data blocks across all pages: {sum(1 for s, _ in all_tables if 'spatial' in s)}")

# ── Print and save ─────────────────────────────────────────────────────────────

print()
tables_saved = []

for idx, (source, df) in enumerate(all_tables, start=1):
    print(f"--- Table {idx}  (source: {source}) ---")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Preview (first 6 rows):")
    print(df.head(6).to_string(index=False))
    print()

    out_name = f"{PAPER_NAME}_table{idx:02d}.csv"
    df.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)

    tables_saved.append({
        "table_index": idx,
        "source":      source,
        "rows":        df.shape[0],
        "cols":        df.shape[1],
        "file":        out_name,
    })

# ── Summary ───────────────────────────────────────────────────────────────────

print("=" * 65)
print(f"SUMMARY  --  {len(tables_saved)} tables extracted")
print("=" * 65)

if tables_saved:
    summary_df = pd.DataFrame(tables_saved)
    print(summary_df[["table_index", "source", "rows", "cols", "file"]].to_string(index=False))
    summary_path = os.path.join(OUTPUT_DIR, f"{PAPER_NAME}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nAll CSVs saved to: {OUTPUT_DIR}")
else:
    print("No data tables detected.")
    print("This paper likely presents results only in figures.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Parse and join tables into a clean ML-ready dataset
# ═══════════════════════════════════════════════════════════════════════════════

# ── 2a: Extract CCD design matrix (inputs) from left column of PDF page 3 ─────

def extract_design_matrix(pdf_path):
    """
    The CCD design matrix (arabic gum %, xanthan gum %, orange oil %) lives
    in the LEFT column of PDF page 3. We crop to x < 55% of page width,
    group words into rows, and pick rows that contain exactly 5 numbers:
    [run, block, arabic_gum_pct, xanthan_gum_pct, orange_oil_pct].
    """
    doc = fitz.open(pdf_path)
    page = doc[2]  # 0-indexed — page 3 of the PDF
    page_width = page.rect.width

    words = page.get_text("words")
    left_words = [w for w in words if w[0] < page_width * 0.55]
    left_words = sorted(left_words, key=lambda w: (round(w[1] / 4), w[0]))

    cur_row, cur_y, rows = [], None, []
    for w in left_words:
        x0, y0, text = w[0], w[1], w[4]
        if cur_y is None or abs(y0 - cur_y) > 4:
            if cur_row:
                rows.append([t for _, t in sorted(cur_row)])
            cur_row, cur_y = [(x0, text)], y0
        else:
            cur_row.append((x0, text))
    if cur_row:
        rows.append([t for _, t in sorted(cur_row)])

    design_rows = []
    for row_texts in rows:
        nums = []
        for t in row_texts:
            try:
                nums.append(float(t.replace(',', '.')))
            except ValueError:
                pass
        # Expect exactly 5 numbers: run (1-20), block (1-3), x1, x2, x3
        if len(nums) == 5 and 1 <= nums[0] <= 20:
            design_rows.append({
                'run':             int(nums[0]),
                'block':           int(nums[1]),
                'arabic_gum_pct':  nums[2],
                'xanthan_gum_pct': nums[3],
                'orange_oil_pct':  nums[4],
            })

    if not design_rows:
        return None
    return pd.DataFrame(design_rows).sort_values('run').reset_index(drop=True)


# ── 2b: Parse Table 7 (p7-spatial-b1) with proper column names ────────────────

def parse_table7(csv_path):
    """
    Column mapping confirmed from CSV inspection:
      col '0'    = run number (1–20)
      col '1'    = turbidity_loss_rate  Y0 (experimental, Å/day)
      col '3'    = turbidity_loss_rate  Yi (predicted)
      col '6'/'7'= viscosity_60rpm      Y0 (spatial shift: smaller numbers land in col 7)
      col '8'    = viscosity_60rpm      Yi
      col '11'   = viscosity_ratio      Y0
      col '12'   = viscosity_ratio      Yi
      col '15'   = zeta_potential       Y0 (mV)
      col '17'   = zeta_potential       Yi
      col '20'   = mobility             Y0 (µm·cm/V·s)
      col '21'   = mobility             Yi
    Last 2 rows are footnotes — skipped.
    """
    df = pd.read_csv(csv_path, dtype=str)
    data = df.iloc[:20].copy()

    def col(name):
        return pd.to_numeric(data[name], errors='coerce')

    clean = pd.DataFrame()
    clean['run']                      = col('0').astype('Int64')
    clean['turbidity_loss_rate_exp']  = col('1')
    clean['turbidity_loss_rate_pred'] = col('3')
    # viscosity Y0 shifts between col 6 and 7 depending on digit width
    clean['viscosity_60rpm_exp']      = col('6').combine_first(col('7'))
    clean['viscosity_60rpm_pred']     = col('8')
    clean['viscosity_ratio_exp']      = col('11')
    clean['viscosity_ratio_pred']     = col('12')
    clean['zeta_potential_exp']       = col('15')
    clean['zeta_potential_pred']      = col('17')
    clean['mobility_exp']             = col('20')
    clean['mobility_pred']            = col('21')

    return clean.reset_index(drop=True)


# ── Run Step 2 ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 2: Building clean ML-ready dataset")
print("=" * 65)

print("\n[2a] Extracting CCD design matrix from left column of page 3...")
design_df = extract_design_matrix(PAPER_PATH)

if design_df is not None:
    print(f"  Extracted {len(design_df)} runs")
    print(design_df.to_string(index=False))
else:
    print("  Could not extract design matrix — check page layout")

table7_csv = os.path.join(OUTPUT_DIR, f"{PAPER_NAME}_table03.csv")
print(f"\n[2b] Parsing Table 7 from {os.path.basename(table7_csv)}...")
outputs_df = parse_table7(table7_csv)
print(f"  Parsed {len(outputs_df)} runs")
print(outputs_df.to_string(index=False))

if design_df is not None and len(design_df) > 0:
    print("\n[2c] Joining inputs + outputs by run number...")
    combined = design_df.merge(outputs_df, on='run', how='inner')
    print(f"  Combined: {combined.shape[0]} rows x {combined.shape[1]} columns")
    print()
    print(combined.to_string(index=False))

    clean_path = os.path.join(OUTPUT_DIR, f"{PAPER_NAME}_clean_dataset.csv")
    combined.to_csv(clean_path, index=False)
    print(f"\nClean dataset saved to: {clean_path}")
else:
    print("\nSkipping join — design matrix not available.")
    outputs_only = os.path.join(OUTPUT_DIR, f"{PAPER_NAME}_outputs_only.csv")
    outputs_df.to_csv(outputs_only, index=False)
    print(f"Outputs-only saved to: {outputs_only}")
