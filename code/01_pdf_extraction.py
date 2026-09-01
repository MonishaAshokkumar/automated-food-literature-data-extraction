import fitz
import re

# ==========================
# STEP 1: LOAD PDF
# ==========================

pdf_path = r"D:/Monisha/UCC Project/papers/foods-10-03024.pdf"

doc = fitz.open(pdf_path)

text = ""

for page in doc:
    text += page.get_text()

text = text.lower()

# ==========================
# STEP 2: EXTRACT OIL TYPES
# ==========================

oil_types = [
    "olive oil",
    "sunflower oil",
    "palm oil",
    "coconut oil",
    "soybean oil",
    "corn oil",
    "canola oil",
    "rapeseed oil"
]

found_oils = []

for oil in oil_types:
    if oil in text:
        found_oils.append(oil)

# ==========================
# STEP 3: EXTRACT TEMPERATURES
# ==========================

temperatures = re.findall(
    r'\d+(?:\.\d+)?\s*°c',
    text
)

# remove duplicates
temperatures = list(set(temperatures))

# ==========================
# STEP 4: EXTRACT WT.% VALUES
# ==========================

concentrations = re.findall(
    r'\d+(?:\.\d+)?\s*wt\.?\s*%',
    text
)

concentrations = list(set(concentrations))

# ==========================
# STEP 5: EXTRACT PARTICIPANTS
# ==========================

participants = re.findall(
    r'(\d+)\s+participants',
    text
)

participants = list(set(participants))

# ==========================
# STEP 6: EXTRACT DROPLET SIZES
# ==========================

droplet_sizes = re.findall(
    r'\d+(?:\.\d+)?\s*[µμu]m',
    text
)

droplet_sizes = list(set(droplet_sizes))

# ==========================
# STEP 7: STORE RESULTS
# ==========================

paper_data = {
    "oil_types": found_oils,
    "temperatures": temperatures,
    "concentrations": concentrations,
    "participants": participants,
    "droplet_sizes": droplet_sizes
}

# ==========================
# STEP 8: DISPLAY RESULTS
# ==========================

print("\n========== EXTRACTED DATA ==========\n")

for key, value in paper_data.items():
    print(f"{key}:")
    print(value)
    print()