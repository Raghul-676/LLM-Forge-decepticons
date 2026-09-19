
from pathlib import Path
from collections import Counter
import json
import re

import pandas as pd


ROOT = Path("data/raw")
OUTPUT = Path("outputs")
OUTPUT.mkdir(parents=True, exist_ok=True)


def normalize_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def inspect_value(value):
    """Return a compact description of a nested JSON value."""
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": list(value.keys())[:20]
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "first_item_type": (
                type(value[0]).__name__ if value else None
            )
        }
    return {
        "type": type(value).__name__,
        "preview": str(value)[:300]
    }


# =========================================================
# 1. CONSTITUTION AUDIT
# =========================================================

print("\n" + "=" * 70)
print("CONSTITUTION AUDIT")
print("=" * 70)

constitution_path = (
    ROOT / "constitution" / "Constitution Of India.csv"
)

df = pd.read_csv(constitution_path, dtype=str)

texts = df["Articles"].fillna("").map(normalize_text)

constitution_summary = {
    "rows": len(df),
    "columns": df.columns.tolist(),
    "empty_rows": int((texts == "").sum()),
    "exact_duplicate_rows": int(texts.duplicated().sum()),
    "unique_texts": int(texts.nunique()),
    "total_characters": int(texts.str.len().sum())
}

print(json.dumps(constitution_summary, indent=2))

print("\nFirst 5 rows:")
for i, text in enumerate(texts.head(5), start=1):
    print(f"\nROW {i}:")
    print(text[:500])

print("\nLast 3 rows:")
for i, text in texts.tail(3).items():
    print(f"\nROW {i + 1}:")
    print(text[:500])


# =========================================================
# 2. CENTRAL ACTS AUDIT
# =========================================================

print("\n" + "=" * 70)
print("CENTRAL ACTS AUDIT")
print("=" * 70)

acts_folder = (
    ROOT / "central_acts" / "annotatedCentralActs"
)

act_files = sorted(acts_folder.glob("*.json"))

records = []
top_level_keys = Counter()
section_keys = Counter()
errors = []

sample_section = None

for file in act_files:

    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        top_level_keys.update(data.keys())

        sections = data.get("Sections", {})

        if isinstance(sections, dict):
            section_count = len(sections)
            section_items = list(sections.items())
        else:
            section_count = 0
            section_items = []

        for section_name, section_data in section_items:
            if isinstance(section_data, dict):
                section_keys.update(section_data.keys())

                if sample_section is None:
                    sample_section = {
                        "source_file": file.name,
                        "section_name": section_name,
                        "section_data": section_data
                    }

        records.append({
            "source_file": file.name,
            "act_title": data.get("Act Title"),
            "act_id": data.get("Act ID"),
            "enactment_date": data.get("Enactment Date"),
            "section_count": section_count,
            "definition_type": type(
                data.get("Act Definition")
            ).__name__,
            "sections_type": type(sections).__name__,
            "schedule_type": type(
                data.get("Schedule")
            ).__name__,
            "footnotes_type": type(
                data.get("Footnotes")
            ).__name__
        })

    except Exception as exc:
        errors.append({
            "source_file": file.name,
            "error": str(exc)
        })


acts_df = pd.DataFrame(records)

acts_summary = {
    "json_files_found": len(act_files),
    "acts_loaded": len(records),
    "load_errors": len(errors),
    "total_sections": int(
        acts_df["section_count"].sum()
    ) if not acts_df.empty else 0,
    "acts_without_sections": int(
        (acts_df["section_count"] == 0).sum()
    ) if not acts_df.empty else 0,
    "duplicate_act_ids": int(
        acts_df["act_id"].dropna().duplicated().sum()
    ) if not acts_df.empty else 0,
    "top_level_keys": dict(top_level_keys),
    "section_keys": dict(section_keys)
}

print(json.dumps(acts_summary, indent=2))

print("\nFirst 5 Acts:")
if not acts_df.empty:
    print(
        acts_df[
            ["act_title", "act_id", "enactment_date", "section_count"]
        ].head(5).to_string(index=False)
    )

print("\nSample section structure:")
if sample_section:
    sample = {
        "source_file": sample_section["source_file"],
        "section_name": sample_section["section_name"],
        "fields": {
            key: inspect_value(value)
            for key, value in sample_section["section_data"].items()
        }
    }
    print(json.dumps(sample, indent=2, ensure_ascii=False))

print("\nLoad errors:")
print(errors[:10])


# =========================================================
# 3. SAVE REPORTS
# =========================================================

acts_df.to_csv(
    OUTPUT / "central_acts_inventory.csv",
    index=False,
    encoding="utf-8-sig"
)

summary = {
    "constitution": constitution_summary,
    "central_acts": acts_summary,
    "central_acts_errors": errors
}

(OUTPUT / "statutory_audit_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
print("Saved: outputs/statutory_audit_summary.json")
print("Saved: outputs/central_acts_inventory.csv")