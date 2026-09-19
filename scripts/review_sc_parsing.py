from pathlib import Path
import json
import re
import pandas as pd


INPUT_FILE = Path(
    "data/processed/supreme_court/sc_structured_test.jsonl"
)

REVIEW_FILE = Path(
    "outputs/sc_manual_review.txt"
)

CHECKLIST_FILE = Path(
    "outputs/sc_manual_review_checklist.csv"
)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def suggested_title(petitioner, respondent):
    """
    Remove only a trailing Vs./V. separator from
    the petitioner before constructing the title.
    """
    petitioner = re.sub(
        r"\s+\b(?:VS|V)\.?\s*$",
        "",
        petitioner or "",
        flags=re.IGNORECASE
    ).strip()

    respondent = (respondent or "").strip()

    if petitioner and respondent:
        return f"{petitioner} v. {respondent}"

    return petitioner or respondent or ""


records = load_jsonl(INPUT_FILE)

review_rows = []
report_parts = []


for record in records:

    meta = record.get("metadata", {})
    raw_text = record.get("raw_text", "")
    body = record.get("judgment_text", "")

    # Show the original header before the first JUDGMENT: marker.
    marker = re.search(
        r"(?im)^[ \t]*JUDGMENT[ \t]*:",
        raw_text
    )

    if marker:
        raw_header = raw_text[:marker.start()]
    else:
        raw_header = raw_text[:5000]

    # Inspect whether HEADNOTE is present in the original.
    headnote_marker = bool(
        re.search(
            r"(?im)^[ \t]*HEADNOTE[ \t]*:",
            raw_header
        )
    )

    source_file = meta.get("source_file", "")

    report_parts.append(
        "=" * 80 + "\n"
        f"FILE: {source_file}\n"
        "=" * 80 + "\n\n"
        f"PARSED TITLE:\n{record.get('title', '')}\n\n"
        f"SUGGESTED TITLE:\n"
        f"{suggested_title(record.get('petitioner'), record.get('respondent'))}\n\n"
        f"DATE: {record.get('judgment_date')}\n"
        f"BENCH: {record.get('bench')}\n"
        f"CITATION BLOCK: {record.get('citation_block')}\n"
        f"HEADNOTE MARKER IN SOURCE: {headnote_marker}\n"
        f"HEADNOTE LENGTH: {len(record.get('headnote', ''))}\n"
        f"JUDGMENT LENGTH: {len(body)}\n"
        f"QUALITY FLAGS: {meta.get('quality_flags', [])}\n\n"
        "----- ORIGINAL PDF HEADER -----\n"
        f"{raw_header[:6000]}\n\n"
        "----- JUDGMENT BODY: FIRST 1200 CHARACTERS -----\n"
        f"{body[:1200]}\n\n"
        "----- JUDGMENT BODY: LAST 600 CHARACTERS -----\n"
        f"{body[-600:]}\n\n"
    )

    review_rows.append({
        "source_file": source_file,
        "parsed_title": record.get("title", ""),
        "suggested_title": suggested_title(
            record.get("petitioner"),
            record.get("respondent")
        ),
        "date": record.get("judgment_date"),
        "respondent_present": bool(record.get("respondent")),
        "citation_present": bool(record.get("citation_block")),
        "headnote_marker_in_source": headnote_marker,
        "headnote_characters": len(record.get("headnote", "")),
        "judgment_characters": len(body),
        "quality_flags": "; ".join(
            meta.get("quality_flags", [])
        ),
        "manual_review": "PENDING",
        "review_notes": ""
    })


REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)

REVIEW_FILE.write_text(
    "\n".join(report_parts),
    encoding="utf-8"
)

pd.DataFrame(review_rows).to_csv(
    CHECKLIST_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("=" * 70)
print("MANUAL REVIEW PREPARED")
print("=" * 70)

print("Documents:", len(records))

print(
    "Headnote markers in original headers:",
    sum(row["headnote_marker_in_source"] for row in review_rows)
)

print(
    "Missing respondents:",
    sum(not row["respondent_present"] for row in review_rows)
)

print("\nReview text:", REVIEW_FILE)
print("Checklist:", CHECKLIST_FILE)