# Automated Food Literature Data Extraction

Code and datasets for the MSc dissertation:
**"Automated Extraction of Quantitative Data from Food Science Literature for Data-Driven Modelling"**

Monisha Ashokkumar — MSc Data Science & Analytics, University College Cork, 2025

---

## Repository Structure

├── code/
│   ├── 01_pdf_extraction.py        # PDF text extraction using PyMuPDF
│   ├── 01_sentence_extraction.py   # Sentence segmentation and candidate filtering
│   ├── 02_nlp_analysis.py          # NER and dependency parsing (SciSpaCy)
│   ├── 02_ml_model.py              # ML model training and evaluation
│   ├── 03_table_extraction.py      # Table extraction from PDFs
│   ├── 04_combine_tables.py        # Combine table extraction outputs
│   ├── 04_variable_extract.py      # Variable normalisation and schema mapping
│   ├── 05_ocr_extraction.py        # OCR extraction from figure images
│   ├── 06_wide_format_output.py    # Pivot to wide-format meta-dataset
│   ├── 07_predict_new_paper.py     # Predict properties for a new paper
│   ├── 08_visualisations.py        # Generate figures
│   ├── 09_extraction_summary.py    # Extraction statistics summary
│   └── 10_ground_truth_evaluation.py  # Precision, recall and F1 evaluation
│
└── data/
├── wide_format_dataset.xlsx    # Meta-dataset: one row per paper, one column per variable
├── extended_with_ocr.xlsx      # Full raw extraction output (all methods combined)
└── ExampleData.xlsx            # Manually curated reference database (ground truth)



---

## Requirements

- Python 3.9+
- pymupdf
- spacy
- scispacy
- en_core_sci_md (SciSpaCy model)
- pandas
- scikit-learn
- openpyxl
- tabula-py
- camelot-py
- easyocr

---

## Usage

Run scripts in the following order:

1. `01_pdf_extraction.py` — extract text from PDFs
2. `01_sentence_extraction.py` — segment and filter sentences
3. `02_nlp_analysis.py` — run NER and dependency parsing
4. `04_variable_extract.py` — normalise extracted variables
5. `03_table_extraction.py` + `04_combine_tables.py` — extract table data
6. `05_ocr_extraction.py` — extract from figures
7. `06_wide_format_output.py` — build the meta-dataset
8. `02_ml_model.py` — train and evaluate ML models
9. `10_ground_truth_evaluation.py` — evaluate against reference database

---

## Data Availability

The 27 source PDF papers are not included due to copyright restrictions.
The extracted meta-dataset and raw extraction outputs are provided in the `data/` folder.
