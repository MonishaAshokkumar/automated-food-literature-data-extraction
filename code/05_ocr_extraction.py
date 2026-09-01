import fitz
import pandas as pd
import re
import os
import glob
import sys
import io
import numpy as np
from multiprocessing import Process, Queue
import queue as _queue_module

sys.stdout.reconfigure(encoding="utf-8")

# ── OCR backend: RapidOCR (ONNX — no PaddlePaddle, Windows CPU safe) ──────────
try:
    from rapidocr_onnxruntime import RapidOCR
    print("OCR backend: RapidOCR (ONNX)")
    OCR_BACKEND = "rapidocr"
except ImportError:
    print("ERROR: Install rapidocr-onnxruntime.  pip install rapidocr-onnxruntime")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Install Pillow first.  pip install Pillow")
    sys.exit(1)

# Lazy engine — one instance per process (avoids re-loading on every call)
_engine = None
def get_engine():
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine

# ── Configuration ─────────────────────────────────────────────────────────────

PAPERS_DIR    = r"D:\Monisha\UCC Project\papers"
EXTENDED_XLSX = r"D:\Monisha\UCC Project\outputs\extended_extracted_all_papers.xlsx"
OCR_XLSX      = r"D:\Monisha\UCC Project\outputs\ocr_extracted.xlsx"
OUT_EXTENDED  = r"D:\Monisha\UCC Project\outputs\extended_with_ocr.xlsx"

MIN_IMG_WIDTH      = 150    # pixels — skip icons, logos, small decorations
MIN_IMG_HEIGHT     = 150
MIN_TEXT_CHARS     = 20     # skip images with almost no OCR text
CONFIDENCE_OCR     = 0.80   # raised from 0.70 — stricter confidence cuts noise
MAX_DUPLICATE_VALS = 3      # if same value appears more than this → axis tick, skip
SUBPROCESS_TIMEOUT = 300    # seconds — per paper; crash-isolated

# ── Reuse exact patterns and normalizers from 01_sentence_extraction.py ───────

KNOWN_UNITS = {
    'µm', 'um', 'nm', 'mm', 'cm', 'm',
    '%', 'wt%', 'wt.%', 'w/w', 'v/v', 'w/v',
    'g/l', 'mg/l', 'mg/ml', 'g/ml', 'µg/ml', 'µg/g',
    'g', 'kg', 'mg', 'µg',
    'ml', 'µl', 'l',
    '°c', '°f', 'k',
    'kj', 'kcal',
    'mpa', 'pa', 'kpa', 'pa·s', 'pa.s', 'mpa·s',
    'cp', 'rpm', 'min', 'h', 's',
    'mol', 'mmol', 'µmol',
    'mm/s', 'n', 'bar', 'mpa',
}

FOOD_PATTERNS = [
    ("d90_µm",                 re.compile(r'd90\s*=?\s*(\d+\.?\d*)\s*(µm|nm|mm)', re.I),            1, None),
    ("d50_µm",                 re.compile(r'd[v,]?0[.,]5\s*=?\s*(\d+\.?\d*)\s*(µm|nm|mm)', re.I),   1, None),
    ("d32_µm",                 re.compile(r'd3[,.]2\s*=?\s*(\d+\.?\d*)\s*(µm|nm|mm)', re.I),        1, None),
    ("d43_µm",                 re.compile(r'd4[,.]3\s*=?\s*(\d+\.?\d*)\s*(µm|nm|mm)', re.I),        1, None),
    ("Temperature_C",          re.compile(r'(\d+\.?\d*)\s*°C', re.I),                               1, "°C"),
    ("Concentration_wt%",      re.compile(r'(\d+\.?\d*)\s*wt\.?%', re.I),                           1, "wt%"),
    ("Homogenization_rpm",     re.compile(r'(\d[\d,]*)\s*rpm', re.I),                               1, "rpm"),
    ("Processing_time_s",      re.compile(r'for\s+(\d+)\s*s\b', re.I),                              1, "s"),
    ("Participants_n",         re.compile(r'(\d+)\s*(?:panelists?|participants?|assessors?)', re.I), 1, ""),
    ("Viscosity_Pa_s",         re.compile(r'(\d+\.?\d*)\s*Pa[·.]s', re.I),                          1, "Pa·s"),
    ("Sliding_speed_mm_s",     re.compile(r'(\d+\.?\d*)\s*mm/s', re.I),                             1, "mm/s"),
    ("Normal_force_N",         re.compile(r'(\d+\.?\d*)\s*N\b', re.I),                              1, "N"),
    ("Pressure_bar",           re.compile(r'(\d+)\s*bar\b', re.I),                                  1, "bar"),
    ("Pressure_MPa",           re.compile(r'(\d+\.?\d*)\s*MPa\b', re.I),                            1, "MPa"),
    ("Volume_mL",              re.compile(r'(\d+\.?\d*)\s*mL\b', re.I),                             1, "mL"),
    ("Shear_rate_s-1",         re.compile(r'(\d+\.?\d*)\s*s[\-⁻]1', re.I),                         1, "s⁻¹"),
    ("Particle_size_µm",       re.compile(r'(\d+\.?\d*)\s*(µm|nm|mm)\b', re.I),                    1, None),
    ("Fat_concentration_wt%",  re.compile(r'(\d+\.?\d*)\s*%\s*(?:fat|oil|lipid)', re.I),           1, "wt%"),
]

VARIABLE_NORMALIZER = {
    'olive oil':              'Oil_type',
    'oil':                    'Oil_Fat_type',
    'fat':                    'Oil_Fat_type',
    'fat content':            'Fat_concentration_wt%',
    'oil content':            'Oil_concentration_wt%',
    'casein':                 'Protein_type',
    'whey protein':           'Protein_type',
    'protein':                'Protein_type',
    'protein concentration':  'Protein_concentration_wt%',
    'emulsion':               'Emulsion_system',
    'droplet size':           'd90_µm',
    'droplet':                'Particle_size_µm',
    'd90':                    'd90_µm',
    'size':                   'Particle_size_µm',
    'viscosity':              'Viscosity_Pa_s',
    'apparent viscosity':     'Viscosity_Pa_s',
    'friction':               'Friction_coefficient',
    'friction coefficient':   'Friction_coefficient',
    'lubrication':            'Lubrication_property',
    'panelists':              'Participants_n',
    'participants':           'Participants_n',
    'creaminess':             'Sensory_creaminess',
    'smoothness':             'Sensory_smoothness',
    'thickness':              'Sensory_thickness',
    'temperature':            'Temperature_C',
}

VALUE_RANGES = {
    "Sensory_creaminess":        (0,    100),
    "Sensory_smoothness":        (0,    100),
    "Sensory_thickness":         (0,    100),
    "Participants_n":            (1,    500),
    "Temperature_C":             (0,    200),
    "Particle_size_µm":          (0.001, 1000),
    "d90_µm":                    (0.001, 1000),
    "Viscosity_Pa_s":            (0,    100000),
    "Friction_coefficient":      (0,    5),
    "Homogenization_rpm":        (0,    50000),
    "Pressure_MPa":              (0,    1000),
    "Fat_concentration_wt%":     (0,    100),
    "Oil_concentration_wt%":     (0,    100),
    "Protein_concentration_wt%": (0,    100),
    "Concentration_wt%":         (0,    100),
    "Volume_mL":                 (0,    10000),
    "Shear_rate_s-1":            (0,    100000),
}

NOISE_ENTITIES = {
    'results', 'result', 'study', 'studies', 'effect', 'effects',
    'figure', 'table', 'food', 'foods', 'method', 'methods',
    'data', 'model', 'models', 'value', 'values', 'level', 'levels',
    'sample', 'samples', 'type', 'types', 'mean', 'range', 'rate',
}

def normalize_variable(entity_text):
    return VARIABLE_NORMALIZER.get(entity_text.lower().strip())

def value_in_range(norm_var, value):
    if norm_var not in VALUE_RANGES:
        return True
    try:
        v = float(str(value).replace(',', ''))
    except (ValueError, TypeError):
        return True
    lo, hi = VALUE_RANGES[norm_var]
    return lo <= v <= hi

def extract_by_patterns(sentence_text):
    rows = []
    seen_spans = []
    for var_name, pattern, val_group, fixed_unit in FOOD_PATTERNS:
        for match in pattern.finditer(sentence_text):
            span = (match.start(), match.end())
            if any(s[0] <= span[0] < s[1] or span[0] <= s[0] < span[1] for s in seen_spans):
                continue
            seen_spans.append(span)
            value = match.group(val_group)
            if fixed_unit is None and match.lastindex and match.lastindex >= 2:
                unit = match.group(2) or ""
            else:
                unit = fixed_unit if fixed_unit is not None else ""
            rows.append({
                "Sentence":            sentence_text,
                "Entity":              match.group(0),
                "Entity_Label":        "PATTERN",
                "Value":               value,
                "Unit":                unit,
                "Relation_Method":     "ocr_pattern",
                "Normalized_Variable": var_name,
                "Confidence":          1,
            })
    return rows


# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess_image(pil_img):
    """
    Improve image quality before OCR:
    - Convert to greyscale (removes colour noise)
    - Increase contrast (makes text sharper against background)
    - Sharpen (reduces blur from low-resolution images)
    """
    from PIL import ImageEnhance, ImageFilter
    img = pil_img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)
    return img.convert("RGB")


def fix_ocr_numbers(text):
    """
    Fix common OCR character substitution errors in numeric contexts.
    Only fixes characters inside tokens that already contain a digit.
    """
    tokens = text.split()
    fixed = []
    for tok in tokens:
        if re.search(r'\d', tok):
            tok = tok.replace('O', '0').replace('o', '0')
            tok = tok.replace('l', '1').replace('I', '1')
            tok = tok.replace('B', '8')
            tok = tok.replace('S', '5')
            tok = tok.replace('Z', '2')
            tok = tok.replace('G', '6')
        fixed.append(tok)
    return " ".join(fixed)


# ── Table layout detection ────────────────────────────────────────────────────

def is_table_layout(ocr_results, row_tol=15, min_rows=4, min_cols=3, min_total=16):
    """
    Decide if OCR results look like a table rather than a graph.

    Logic:
      - Cluster text boxes by y-centre into rows (within row_tol pixels)
      - A table has >= min_rows rows, each with >= min_cols elements
      - Total confident detections must be >= min_total

    Graphs have few text boxes (axis labels only) at the edges.
    Tables have many boxes arranged in a regular grid.
    """
    if len(ocr_results) < min_total:
        return False

    y_centres = []
    for (bbox, text, conf) in ocr_results:
        if conf < CONFIDENCE_OCR:
            continue
        ys = [pt[1] for pt in bbox]
        y_centres.append(sum(ys) / len(ys))

    if not y_centres:
        return False

    y_centres_sorted = sorted(y_centres)
    rows = []
    current_row = [y_centres_sorted[0]]
    for y in y_centres_sorted[1:]:
        if y - current_row[-1] <= row_tol:
            current_row.append(y)
        else:
            rows.append(current_row)
            current_row = [y]
    rows.append(current_row)

    qualifying_rows = [r for r in rows if len(r) >= min_cols]
    return len(qualifying_rows) >= min_rows


# ── OCR: one image ────────────────────────────────────────────────────────────

def ocr_image(pil_img):
    """
    Run RapidOCR on a PIL Image.
    Returns (text_string, ocr_results) where ocr_results is a list of
    (bbox, text, confidence) tuples — used by is_table_layout().
    """
    MAX_DIM = 2000
    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    if w > MAX_DIM or h > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    try:
        img_array = np.array(pil_img)
    except MemoryError:
        return "", []

    engine = get_engine()
    try:
        result, _ = engine(img_array)
    except Exception:
        return "", []

    if result is None:
        return "", []

    # Normalise to (bbox, text, conf) tuples — same shape as before
    ocr_results = [(box, text, float(score)) for box, text, score in result]
    lines = [text for (_, text, conf) in ocr_results if conf >= CONFIDENCE_OCR]
    return " ".join(lines), ocr_results


# ── Extract images from one PDF and OCR them ─────────────────────────────────

def extract_ocr_rows_from_pdf(pdf_path, paper_name):
    doc = fitz.open(pdf_path)
    all_rows = []
    image_count = 0
    useful_count = 0

    for page_num, page in enumerate(doc, start=1):
        img_list = page.get_images(full=True)

        for img_index, img_info in enumerate(img_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            width  = base_image.get("width", 0)
            height = base_image.get("height", 0)

            if width < MIN_IMG_WIDTH or height < MIN_IMG_HEIGHT:
                continue

            image_count += 1
            img_bytes = base_image["image"]

            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                pil_img = preprocess_image(pil_img)
            except Exception:
                continue

            ocr_text, ocr_results = ocr_image(pil_img)
            ocr_text = fix_ocr_numbers(ocr_text.strip())

            if len(ocr_text) < MIN_TEXT_CHARS:
                continue

            # Skip graphs — only process images whose text layout looks like a table
            if not is_table_layout(ocr_results):
                continue

            useful_count += 1

            ocr_text = re.sub(r'[\x00-\x1F\x7F]', ' ', ocr_text)
            ocr_text = re.sub(r' {2,}', ' ', ocr_text)

            rows = extract_by_patterns(ocr_text)
            for r in rows:
                r["Paper"]  = paper_name
                r["Source"] = f"p{page_num}_img{img_index}"

            # Axis-tick filter: same value repeated many times → skip
            from collections import Counter
            val_counts = Counter(r["Value"] for r in rows)
            rows = [r for r in rows if val_counts[r["Value"]] <= MAX_DUPLICATE_VALS]

            # Physical range filter
            rows = [
                r for r in rows
                if value_in_range(r["Normalized_Variable"], r["Value"])
            ]

            all_rows.extend(rows)

    return all_rows, image_count, useful_count


# ── Subprocess isolation — native crashes can't kill the pipeline ─────────────

def _ocr_worker(pdf_path, paper_name, result_queue):
    """
    Runs inside a child process. If RapidOCR causes a C-level crash on a
    specific paper, only this process dies — the parent catches the non-zero
    exit code and moves on.
    """
    try:
        rows, n_imgs, n_useful = extract_ocr_rows_from_pdf(pdf_path, paper_name)
        result_queue.put((rows, n_imgs, n_useful))
    except Exception as e:
        result_queue.put(([], 0, 0))


def ocr_paper_safe(pdf_path, paper_name):
    """
    Run OCR for one paper in an isolated subprocess.
    Returns (rows, n_imgs, n_useful) or ([], 0, 0) on crash or timeout.
    """
    q = Queue()
    p = Process(target=_ocr_worker, args=(pdf_path, paper_name, q))
    p.start()
    p.join(SUBPROCESS_TIMEOUT)

    if p.is_alive():
        p.terminate()
        p.join()
        print(f"  [TIMEOUT] OCR timed out after {SUBPROCESS_TIMEOUT}s — skipping")
        return [], 0, 0

    if p.exitcode != 0:
        print(f"  [CRASH] OCR subprocess crashed (exit code {p.exitcode}) — skipping")
        return [], 0, 0

    try:
        return q.get_nowait()
    except _queue_module.Empty:
        return [], 0, 0


# ── Main loop ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 65)
    print("OCR EXTRACTION — RapidOCR — food science PDFs")
    print("=" * 65)

    pdf_files = sorted(glob.glob(os.path.join(PAPERS_DIR, "*.pdf")))
    print(f"Found {len(pdf_files)} PDFs\n")

    all_ocr_rows = []
    paper_stats  = []

    for pdf_path in pdf_files:
        paper_name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\nProcessing: {paper_name[:60]}")

        rows, n_imgs, n_useful = ocr_paper_safe(pdf_path, paper_name)
        all_ocr_rows.extend(rows)

        print(f"  Images found: {n_imgs}  |  With useful text: {n_useful}  |  Rows extracted: {len(rows)}")
        paper_stats.append({
            "paper":         paper_name,
            "images_found":  n_imgs,
            "useful_images": n_useful,
            "rows_extracted":len(rows),
        })

    # ── Save OCR results ──────────────────────────────────────────────────────

    print("\n" + "=" * 65)
    print("Saving OCR results")
    print("=" * 65)

    if all_ocr_rows:
        ocr_df = pd.DataFrame(all_ocr_rows)
        ocr_df["Value"] = pd.to_numeric(ocr_df["Value"], errors="coerce")

        useful_ocr = ocr_df[
            ocr_df["Value"].notna() &
            ocr_df["Normalized_Variable"].notna()
        ].copy()

        print(f"\n  Total OCR rows:  {len(ocr_df)}")
        print(f"  Useful rows:     {len(useful_ocr)}  (have value + variable)")

        with pd.ExcelWriter(OCR_XLSX, engine="openpyxl") as writer:
            for pname in ocr_df["Paper"].unique():
                paper_rows = ocr_df[ocr_df["Paper"] == pname]
                if not paper_rows.empty:
                    paper_rows.to_excel(writer, sheet_name=pname[:31], index=False)
            pd.DataFrame(paper_stats).to_excel(writer, sheet_name="__stats__", index=False)

        print(f"\n  OCR results saved to: {OCR_XLSX}")

        # ── Merge into extended dataset ───────────────────────────────────────
        print("\n  Merging with extended_extracted_all_papers.xlsx...")

        xl     = pd.ExcelFile(EXTENDED_XLSX)
        frames = [pd.read_excel(xl, sheet_name=s) for s in xl.sheet_names]
        existing_df = pd.concat(frames, ignore_index=True)
        print(f"  Existing rows: {len(existing_df):,}")

        merge_cols = ["Paper", "Sentence", "Entity", "Entity_Label",
                      "Value", "Unit", "Relation_Method", "Normalized_Variable", "Confidence"]
        ocr_for_merge = useful_ocr[[c for c in merge_cols if c in useful_ocr.columns]].copy()

        combined = pd.concat([existing_df, ocr_for_merge], ignore_index=True)
        print(f"  Combined rows: {len(combined):,}  (+{len(ocr_for_merge)} from OCR)")

        with pd.ExcelWriter(OUT_EXTENDED, engine="openpyxl") as writer:
            for sheet in xl.sheet_names:
                df_sheet = pd.read_excel(xl, sheet_name=sheet)
                df_sheet.to_excel(writer, sheet_name=sheet[:31], index=False)
            ocr_for_merge.to_excel(writer, sheet_name="OCR_extracted", index=False)

        print(f"  Saved to: {OUT_EXTENDED}")

    else:
        print("\n  No OCR rows extracted from any paper.")
        print("  The existing extended_extracted_all_papers.xlsx remains unchanged.")

    # ── Summary ───────────────────────────────────────────────────────────────

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    stats_df = pd.DataFrame(paper_stats)
    print(stats_df[stats_df["images_found"] > 0].to_string(index=False))
    print(f"\nTotal images scanned: {stats_df['images_found'].sum()}")
    print(f"Images with text:     {stats_df['useful_images'].sum()}")
    print(f"Total OCR rows:       {stats_df['rows_extracted'].sum()}")
    print("\nDone.")
