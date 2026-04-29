import xml.etree.ElementTree as ET
try:
    import pandas as pd
except ImportError:
    print("Error: pandas is not installed. Install it with: pip install pandas")
    exit(1)
import json
from pathlib import Path


# ----------------------------
# SETTINGS
# ----------------------------
XML_FILE = "coded_dreams.xml"   # change this to your actual filename
OUTPUT_CSV = "coded_dreams.csv"

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def clean_text(text):
    """Strip whitespace and return None if empty."""
    if text is None:
        return None
    text = text.strip()
    return text if text else None


def extract_codings(codings_elem):
    """
    Turn the <codings> section into a compact dictionary.
    Repeated tags are stored as lists.
    Nested coding tags (e.g., <fri>, <agg>, <sex>, <emot>) are stored as dicts.
    """
    if codings_elem is None:
        return {}

    codings = {}

    for child in codings_elem:
        tag = child.tag

        # If the element has no children, just store its text
        if len(child) == 0:
            value = clean_text(child.text)
        else:
            # Store nested structure as a dictionary
            value = {sub.tag: clean_text(sub.text) for sub in child}

        # Handle repeated tags by storing as a list
        if tag in codings:
            if not isinstance(codings[tag], list):
                codings[tag] = [codings[tag]]
            codings[tag].append(value)
        else:
            codings[tag] = value

    return codings


# ----------------------------
# MAIN PARSING LOGIC
# ----------------------------
xml_path = Path(XML_FILE)

if not xml_path.exists():
    raise FileNotFoundError(
        f"Could not find '{XML_FILE}' in your project folder. "
        "Make sure the XML file is in the same folder as this script."
    )

tree = ET.parse(xml_path)
root = tree.getroot()

rows = []

# Loop through each collection
for collection in root.findall("collection"):
    collection_id = clean_text(collection.findtext("id"))
    collection_name = clean_text(collection.findtext("name"))
    collection_type = clean_text(collection.findtext("type"))
    collection_sex = clean_text(collection.findtext("sex"))
    collection_age = clean_text(collection.findtext("age"))
    collection_time = clean_text(collection.findtext("time"))

    # Loop through each dream inside the collection
    for dream in collection.findall("dream"):
        dream_number = clean_text(dream.findtext("number"))
        dream_date = clean_text(dream.findtext("date"))
        dream_report = clean_text(dream.findtext("report"))

        codings_elem = dream.find("codings")
        codings_dict = extract_codings(codings_elem)

        rows.append({
            "collection_id": collection_id,
            "collection_name": collection_name,
            "collection_type": collection_type,
            "collection_sex": collection_sex,
            "collection_age": collection_age,
            "collection_time": collection_time,
            "dream_number": dream_number,
            "dream_date": dream_date,
            "dream_report": dream_report,
            "codings_json": json.dumps(codings_dict, ensure_ascii=False)
        })

# Convert to DataFrame
df = pd.DataFrame(rows)

# Optional: create a unique dream ID
df["dream_id"] = df["collection_id"].astype(str) + "_" + df["dream_number"].astype(str)

# Reorder columns
column_order = [
    "dream_id",
    "collection_id",
    "collection_name",
    "collection_type",
    "collection_sex",
    "collection_age",
    "collection_time",
    "dream_number",
    "dream_date",
    "dream_report",
    "codings_json"
]
df = df[column_order]

# Save CSV
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"Done. Saved {len(df)} dreams to '{OUTPUT_CSV}'.")
print(df.head())

# ----------------------------
# CREATE CODINGS-LEVEL TABLE
# ----------------------------

import json

def flatten_codings(dream_id, codings_json_str):
    rows = []
    
    if not codings_json_str:
        return rows

    codings_dict = json.loads(codings_json_str)

    for coding_type, value in codings_dict.items():

        # Case 1: list of items
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    rows.append({
                        "dream_id": dream_id,
                        "coding_type": coding_type,
                        "init": item.get("init"),
                        "rec": item.get("rec"),
                        "code": item.get("code"),
                        "char": item.get("char"),
                        "raw": str(item)
                    })
                else:
                    rows.append({
                        "dream_id": dream_id,
                        "coding_type": coding_type,
                        "init": None,
                        "rec": None,
                        "code": item,
                        "char": None,
                        "raw": item
                    })

        # Case 2: single dict
        elif isinstance(value, dict):
            rows.append({
                "dream_id": dream_id,
                "coding_type": coding_type,
                "init": value.get("init"),
                "rec": value.get("rec"),
                "code": value.get("code"),
                "char": value.get("char"),
                "raw": str(value)
            })

        # Case 3: single value
        else:
            rows.append({
                "dream_id": dream_id,
                "coding_type": coding_type,
                "init": None,
                "rec": None,
                "code": value,
                "char": None,
                "raw": value
            })

    return rows


# Build codings dataframe
coding_rows = []

for _, row in df.iterrows():
    dream_id = row["dream_id"]
    codings_json = row["codings_json"]
    
    coding_rows.extend(flatten_codings(dream_id, codings_json))


df_codings = pd.DataFrame(coding_rows)

# Save codings CSV
df_codings.to_csv("dreambank_codings.csv", index=False, encoding="utf-8-sig")

print(f"Saved codings table with {len(df_codings)} rows.")