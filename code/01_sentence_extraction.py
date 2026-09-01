import fitz
import spacy
import pandas as pd
import re
import os
import glob
import sys

sys.stdout.reconfigure(encoding="utf-8")

# --------------------------------------------------
# Configuration
# --------------------------------------------------
PAPERS_DIR   = r"D:\Monisha\UCC Project\papers"
OUTPUT_EXCEL = r"D:\Monisha\UCC Project\extracted_all_papers.xlsx"

nlp = spacy.load("en_core_sci_md")

# --------------------------------------------------
# Step 1: Extract raw text from PDF
# --------------------------------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

# --------------------------------------------------
# Step 2: Filter to body sections only
# Keep:  Abstract → Discussion/Conclusion
# Drop:  title block, references, acknowledgments, etc.
# --------------------------------------------------
SECTION_START = re.compile(r'\bAbstract\b', re.IGNORECASE)

SECTION_END = re.compile(
    r'\n\s*(References|Bibliography|Acknowledgments?|Funding|'
    r'Author\s+Contributions?|Conflicts?\s+of\s+Interest|'
    r'Supplementary|Appendix)\b',
    re.IGNORECASE
)

def filter_to_body(text):
    start = SECTION_START.search(text)
    if start:
        text = text[start.start():]
    end = SECTION_END.search(text)
    if end:
        text = text[:end.start()]
    return text

# --------------------------------------------------
# Step 3: Clean text
# --------------------------------------------------
def clean_text(text):
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

# --------------------------------------------------
# Step 4: Entity → Value linking
# Uses spaCy dependency tree first, falls back to closest number
# --------------------------------------------------
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

def get_unit_after(token, sent_span):
    next_i = token.i + 1
    if next_i < sent_span.end:
        next_tok = sent_span.doc[next_i]
        if next_tok.text.lower() in KNOWN_UNITS:
            return next_tok.text
    return ""

def find_value_for_entity(ent_span, sent_span):
    """Returns (value, unit, method). method: 'dependency' | 'proximity' | 'none'"""
    s_start, s_end = sent_span.start, sent_span.end

    def in_sent(tok):
        return s_start <= tok.i < s_end

    for ent_tok in ent_span:
        for child in ent_tok.children:
            if child.like_num and in_sent(child):
                return child.text, get_unit_after(child, sent_span), "dependency"
        if ent_tok.head.like_num and in_sent(ent_tok.head):
            return ent_tok.head.text, get_unit_after(ent_tok.head, sent_span), "dependency"
        for sibling in ent_tok.head.children:
            if sibling.like_num and in_sent(sibling) and sibling.i != ent_tok.i:
                return sibling.text, get_unit_after(sibling, sent_span), "dependency"

    ent_center = (ent_span.start + ent_span.end) / 2.0
    num_tokens = [tok for tok in sent_span if tok.like_num]
    if not num_tokens:
        return None, None, "none"
    closest = min(num_tokens, key=lambda t: abs(t.i - ent_center))
    return closest.text, get_unit_after(closest, sent_span), "proximity"

# --------------------------------------------------
# Step 4b: Pattern-based extraction (high confidence)
# --------------------------------------------------
FOOD_PATTERNS = [
    ("d90_µm",                 re.compile(r'd90\s*=?\s*(\d+\.?\d*)\s*(µm|nm|mm)', re.I),            1, None),
    ("d50_µm",                 re.compile(r'd[v,]?0[.,]5\s*=?\s*(\d+\.?\d*)\s*(µm|nm|mm)', re.I),   1, None),
    ("d32_µm",                 re.compile(r'd3[,.]2\s*=?\s*(\d+\.?\d*)\s*(µm|nm|mm)', re.I),        1, None),
    ("d43_µm",                 re.compile(r'd4[,.]3\s*=?\s*(\d+\.?\d*)\s*(µm|nm|mm)', re.I),        1, None),
    ("Temperature_C",          re.compile(r'(\d+\.?\d*)\s*°C', re.I),                               1, "°C"),
    ("Concentration_wt%",      re.compile(r'(\d+\.?\d*)\s*wt\.?%', re.I),                           1, "wt%"),
    ("Homogenization_rpm",     re.compile(r'(\d[\d,]*)\s*rpm', re.I),                                1, "rpm"),
    ("Processing_time_s",      re.compile(r'for\s+(\d+)\s*s\b', re.I),                              1, "s"),
    ("Participants_n",         re.compile(r'(\d+)\s*(?:pre.trained\s+)?(?:panelists?|participants?|assessors?)', re.I), 1, ""),
    ("Viscosity_Pa_s",         re.compile(r'(\d+\.?\d*)\s*Pa[·.]s', re.I),                          1, "Pa·s"),
    ("Sliding_speed_mm_s",     re.compile(r'(\d+\.?\d*)\s*mm/s', re.I),                             1, "mm/s"),
    ("Normal_force_N",         re.compile(r'(\d+\.?\d*)\s*N\b', re.I),                               1, "N"),
    ("Pressure_bar",           re.compile(r'(\d+)\s*bar\b', re.I),                                   1, "bar"),
    ("Pressure_MPa",           re.compile(r'(\d+\.?\d*)\s*MPa\b', re.I),                             1, "MPa"),
    ("Volume_mL",              re.compile(r'(\d+\.?\d*)\s*mL\b', re.I),                              1, "mL"),
    ("Shear_rate_s-1",         re.compile(r'(\d+\.?\d*)\s*s[\-⁻]1', re.I),                          1, "s⁻¹"),
    ("Particle_size_µm",       re.compile(r'(\d+\.?\d*)\s*(µm|nm|mm)\b', re.I),                     1, None),
]

def extract_by_patterns(sentence_text):
    pattern_rows = []
    seen_spans = []

    for var_name, pattern, val_group, fixed_unit in FOOD_PATTERNS:
        for match in pattern.finditer(sentence_text):
            span = (match.start(), match.end())
            if any(s[0] <= span[0] < s[1] or span[0] <= s[0] < span[1] for s in seen_spans):
                continue
            seen_spans.append(span)

            value = match.group(val_group) if val_group else match.group(0)
            if fixed_unit is None and match.lastindex and match.lastindex >= 2:
                unit = match.group(2) or ""
            else:
                unit = fixed_unit if fixed_unit is not None else ""

            pattern_rows.append({
                "Sentence":            sentence_text,
                "Entity":              match.group(0),
                "Entity_Label":        "PATTERN",
                "Value":               value,
                "Unit":                unit,
                "Relation_Method":     "pattern",
                "Category":            var_name.split("_")[0],
                "Normalized_Variable": var_name,
            })
    return pattern_rows

# --------------------------------------------------
# Step 6 helpers: noise filter, category, normalizer
# --------------------------------------------------
NOISE_ENTITIES = {
    'results', 'result', 'study', 'studies', 'effect', 'effects',
    'systems', 'system', 'increase', 'increases', 'decrease', 'decreases',
    'reduction', 'reductions', 'figure', 'table', 'foods', 'food',
    'constant', 'distribution', 'method', 'methods', 'analysis',
    'data', 'group', 'groups', 'model', 'models', 'value', 'values',
    'level', 'levels', 'condition', 'conditions', 'factor', 'factors',
    'parameter', 'parameters', 'sample', 'samples', 'surfaces', 'surface',
    'perceived', 'perception', 'significant', 'difference', 'differences',
    'term', 'terms', 'type', 'types', 'mean', 'range', 'rate',
    'composition', 'hereinafter', 'herein', 'respectively', 'whereas',
    'however', 'therefore', 'moreover', 'furthermore', 'addition',
}

EXTRA_NOISE = {
    'soft texture', 'oil phase', 'intensity scoring', 'rheological',
    'rheological properties', 'tribological', 'tribological properties',
    'emulsion film', 'lubricating film', 'boundary regime', 'mixed regime',
    'oral processing', 'sensory properties', 'sensory analysis',
    'mass fraction', 'volume fraction', 'friction curves', 'flow curves',
    'shear rate', 'sliding speed', 'lubrication behavior',
}

VARIABLE_CATEGORIES = {
    "Oil / Fat":              ['oil', 'fat', 'lipid', 'triglyceride', 'olive', 'sunflower', 'rapeseed', 'fatty'],
    "Protein":                ['protein', 'casein', 'whey', 'albumin', 'gelatin', 'gluten', 'isolate', 'concentrate'],
    "Emulsifier":             ['emulsifier', 'lecithin', 'tween', 'span', 'surfactant', 'saponin', 'polysorbate'],
    "Droplet / Particle":     ['droplet', 'd90', 'd50', 'd32', 'd43', 'd10', 'particle'],
    "Size / Diameter":        ['size', 'diameter'],
    "Emulsion":               ['emulsion'],
    "Dairy":                  ['milk', 'cream', 'reconstituted', 'skimmed', 'lactose'],
    "Viscosity":              ['viscosity', 'rheol', 'consistency', 'flow'],
    "Friction / Lubrication": ['friction', 'lubrication', 'lubricating', 'tribology', 'coefficient'],
    "Sensory":                ['panelist', 'participant', 'creaminess', 'smoothness', 'thickness',
                               'mouthfeel', 'texture', 'sensory', 'consumer', 'perception'],
    "Composition":            ['content', 'concentration', 'ratio', 'proportion'],
    "Temperature":            ['temperature', 'temp'],
    "pH":                     ['ph'],
    "Processing":             ['homogeniz', 'sonication', 'pressure', 'speed', 'rpm', 'microfluidiz'],
}

VARIABLE_NORMALIZER = {
    'olive oil':              'Oil_type',
    'oil':                    'Oil_type',
    'fat':                    'Oil_Fat_type',
    'fat content':            'Fat_concentration_wt%',
    'oil content':            'Oil_concentration_wt%',
    'fat droplets':           'Particle_type',
    'fat droplet':            'Particle_type',
    'oil droplets':           'Particle_type',
    'casein sodium salt':     'Protein_type',
    'casein':                 'Protein_type',
    'whey protein':           'Protein_type',
    'protein':                'Protein_type',
    'protein concentration':  'Protein_concentration_wt%',
    'milk':                   'Emulsion_system',
    'skimmed milk':           'Emulsion_system',
    'milk cream':             'Emulsion_system',
    'reconstituted milk':     'Emulsion_system',
    'emulsion':               'Emulsion_system',
    'emulsions':              'Emulsion_system',
    'model emulsions':        'Emulsion_system',
    'droplet size':           'd90_µm',
    'droplet':                'Particle_size_µm',
    'd90':                    'd90_µm',
    'size':                   'Particle_size_µm',
    'viscosity':              'Viscosity_Pa_s',
    'apparent viscosity':     'Viscosity_Pa_s',
    'friction':               'Friction_coefficient',
    'friction curves':        'Friction_coefficient',
    'friction coefficient':   'Friction_coefficient',
    'lubrication':            'Lubrication_property',
    'lubricating effect':     'Lubrication_property',
    'panelists':              'Participants_n',
    'participants':           'Participants_n',
    'creaminess':             'Sensory_creaminess',
    'smoothness':             'Sensory_smoothness',
    'thickness':              'Sensory_thickness',
    'temperature':            'Temperature_C',
}

def classify_entity(entity_text):
    e = entity_text.lower()
    for category, keywords in VARIABLE_CATEGORIES.items():
        if any(kw in e for kw in keywords):
            return category
    return None

def is_noise(entity_text):
    e = entity_text.strip().lower()
    if len(e) < 3:
        return True
    if re.match(r'^(fig\.?|figure|table|eq\.?)\s*\d*$', e):
        return True
    if re.match(r'^\d+\.?\d*$', e):
        return True
    if re.match(r'^[A-Z]{2,6}$', entity_text.strip()):
        return True
    if e in NOISE_ENTITIES:
        return True
    return False

def normalize_variable(entity_text):
    return VARIABLE_NORMALIZER.get(entity_text.lower().strip())

# --------------------------------------------------
# Step 6b: Value range filter
# Rejects extracted values that are physically impossible
# (e.g. year numbers captured as sensory scores)
# --------------------------------------------------
VALUE_RANGES = {
    "Sensory_creaminess":        (0,    100),
    "Sensory_smoothness":        (0,    100),
    "Sensory_thickness":         (0,    100),
    "Participants_n":            (1,    500),
    "Temperature_C":             (0,    200),
    "Particle_size_µm":          (0.001, 1000),
    "d90_µm":                    (0.001, 1000),
    "Viscosity_Pa_s":            (0,   100000),
    "Friction_coefficient":      (0,    5),
    "Lubrication_property":      (0,    5),
    "Homogenization_rpm":        (0,   50000),
    "Pressure_MPa":              (0,   1000),
    "Pressure_bar":              (0,   5000),
    "Fat_concentration_wt%":     (0,    100),
    "Oil_concentration_wt%":     (0,    100),
    "Protein_concentration_wt%": (0,    100),
    "Concentration_wt%":         (0,    100),
    "Volume_mL":                 (0,  10000),
    "Processing_time_s":         (0,  86400),
    "Shear_rate_s-1":            (0, 100000),
    "Sliding_speed_mm_s":        (0,   1000),
    "Normal_force_N":            (0,    500),
}

def value_in_range(norm_var, value):
    """Returns False if value is outside the physically realistic range for this variable."""
    if norm_var not in VALUE_RANGES:
        return True
    try:
        v = float(str(value).replace(',', ''))
    except (ValueError, TypeError):
        return True
    lo, hi = VALUE_RANGES[norm_var]
    return lo <= v <= hi

# --------------------------------------------------
# process_paper(): full pipeline for one PDF
# --------------------------------------------------
def process_paper(pdf_path):
    paper_name = os.path.splitext(os.path.basename(pdf_path))[0]

    print(f"  [1] Extracting text...")
    raw_text = extract_text_from_pdf(pdf_path)

    print(f"  [2] Filtering sections...")
    body_text = filter_to_body(raw_text)
    print(f"      Kept {len(body_text):,} of {len(raw_text):,} characters")

    print(f"  [3] Cleaning text...")
    clean = clean_text(body_text)

    print(f"  [4] Running SciSpacy NLP...")
    doc = nlp(clean)

    print(f"  [5] Extracting entity–value pairs...")
    rows = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if len(sent_text) < 20:
            continue
        rows.extend(extract_by_patterns(sent_text))
        for ent in sent.ents:
            value, unit, method = find_value_for_entity(ent, sent)
            rows.append({
                "Sentence":            sent_text,
                "Entity":              ent.text,
                "Entity_Label":        ent.label_,
                "Value":               value,
                "Unit":                unit,
                "Relation_Method":     method,
                "Category":            None,
                "Normalized_Variable": None,
            })

    if not rows:
        print(f"  No rows extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    print(f"  [6] Filtering and classifying...")
    nlp_mask = df["Relation_Method"] != "pattern"
    df.loc[nlp_mask, "Category"]            = df.loc[nlp_mask, "Entity"].apply(classify_entity)
    df.loc[nlp_mask, "Normalized_Variable"] = df.loc[nlp_mask, "Entity"].apply(normalize_variable)

    noise_mask = nlp_mask & df["Entity"].apply(is_noise)
    df = df[~noise_mask]

    word_number_mask = df["Value"].notna() & ~df["Value"].str.contains(r'\d', na=False)
    df.loc[word_number_mask, ["Value", "Unit", "Relation_Method"]] = [None, None, "none"]

    # Apply value range filter — nullify values that fall outside realistic bounds
    range_mask = df.apply(
        lambda row: not value_in_range(row["Normalized_Variable"], row["Value"])
        if pd.notna(row["Value"]) and pd.notna(row["Normalized_Variable"])
        else False,
        axis=1,
    )
    df.loc[range_mask, ["Value", "Unit", "Relation_Method"]] = [None, None, "none"]

    df = df[~df["Entity"].str.lower().isin(EXTRA_NOISE)]

    keep_mask = (df["Relation_Method"] == "pattern") | df["Value"].notna()
    df = df[keep_mask].reset_index(drop=True)

    confidence_order = {"pattern": 1, "dependency": 2, "proximity": 3, "none": 4}
    df["Confidence"]   = df["Relation_Method"].map(confidence_order)
    df["Paper"]        = paper_name

    return df

# --------------------------------------------------
# Run on all PDFs in PAPERS_DIR
# --------------------------------------------------
pdf_files = sorted(glob.glob(os.path.join(PAPERS_DIR, "*.pdf")))

if not pdf_files:
    print(f"No PDF files found in: {PAPERS_DIR}")
else:
    print(f"Found {len(pdf_files)} PDF(s) to process.\n")

    results = {}

    for pdf_path in pdf_files:
        paper_name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\n{'='*60}")
        print(f"Processing: {paper_name}")
        print(f"{'='*60}")

        df = process_paper(pdf_path)
        results[paper_name] = df

        if not df.empty:
            total        = len(df)
            pat_rows     = len(df[df["Relation_Method"] == "pattern"])
            nlp_rows     = total - pat_rows
            with_value   = len(df[df["Value"].notna()])
            print(f"\n  >> {total} rows  |  {pat_rows} pattern  |  {nlp_rows} NLP  |  {with_value} with value")

    # --------------------------------------------------
    # Save: one sheet per paper in a single Excel file
    # --------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Saving to Excel...")

    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        for paper_name, df in results.items():
            if df.empty:
                continue
            sheet_name = paper_name[:31]   # Excel sheet name limit is 31 chars
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  Sheet: {sheet_name}  ({len(df)} rows)")

    print(f"\nDone. All papers saved to:\n{OUTPUT_EXCEL}")
