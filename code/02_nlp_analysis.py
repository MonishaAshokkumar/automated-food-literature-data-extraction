import spacy

# Load model
nlp = spacy.load("en_core_web_sm")

# Load paper text
with open(
    r"D:\Monisha\UCC Project\notebooks\paper1.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

# Run NLP
doc = nlp(text)

# Print entities
print("\nEntities Found:\n")

for ent in doc.ents:
    print(ent.text, " --> ", ent.label_)