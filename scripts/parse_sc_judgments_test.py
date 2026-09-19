from pathlib import Path
from datetime import datetime
from collections import Counter
import hashlib
import json
import re

import fitz
import pandas as pd


PDF_FOLDER = Path("data/raw/supreme_court/pdfs")

OUTPUT_FILE = Path(
    "data/processed/supreme_court/sc_structured_test.jsonl"
)

REPORT_FILE = Path(
    "outputs/sc_parse_quality_test.csv"
)


# ---------------------------------------------------------
# 1. Conservative text cleaning
# ---------------------------------------------------------

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", "")

    # Remove only clearly identifiable page boilerplate.
    text = re.sub(
        r"(?im)^[ \t]*https?://JUDIS\.NIC\.IN/?[ \t]*$",
        "",
        text
    )

    text = re.sub(
        r"(?im)^[ \t]*Page[ \t]+\d+[ \t]+of[ \t]+\d+[ \t]*$",
        "",
        text
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------
# 2. PDF extraction
# ---------------------------------------------------------

def extract_pdf(pdf_path):
    with fitz.open(pdf_path) as document:
        pages = [
            page.get_text("text")
            for page in document
        ]

    raw_text = "\n".join(pages)

    return {
        "raw_text": raw_text,
        "text": clean_text(raw_text),
        "page_count": len(pages)
    }


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# ---------------------------------------------------------
# 3. Identify legacy JUDIS header labels
# ---------------------------------------------------------

# Labels are anchored to the beginning of a line.
# This avoids matching ordinary mentions of "judgment"
# or "citation" inside the body of a case.

MARKER_RE = re.compile(
    r"(?im)^[ \t]*"
    r"(?P<label>"
    r"PETITIONER|RESPONDENT|BENCH|CITATION|ACT|HEADNOTE|JUDGMENT"
    r")[ \t]*:[ \t]*"
    r"|"
    r"^[ \t]*(?P<date_label>DATE[ \t]+OF[ \t]+JUDGMENT)"
    r"[ \t]*:?[ \t]*"
)


def marker_name(match):
    if match.group("date_label"):
        return "DATE OF JUDGMENT"

    return match.group("label").upper()


def extract_sections(text):
    """
    Parse the legacy JUDIS header.

    If the expected structure is absent, return empty fields
    and flag the document for review rather than guessing.
    """

    fields = {
        "PETITIONER": "",
        "RESPONDENT": "",
        "DATE OF JUDGMENT": "",
        "BENCH": "",
        "CITATION": "",
        "ACT": "",
        "HEADNOTE": "",
        "JUDGMENT": ""
    }

    matches = list(MARKER_RE.finditer(text))

    # A legacy header should contain PETITIONER near the beginning.
    petitioner_match = next(
        (
            m for m in matches
            if marker_name(m) == "PETITIONER"
            and m.start() < 12000
        ),
        None
    )

    if petitioner_match is None:
        return fields, False

    # Find the first actual JUDGMENT: marker after the header begins.
    body_match = next(
        (
            m for m in matches
            if marker_name(m) == "JUDGMENT"
            and m.start() > petitioner_match.start()
        ),
        None
    )

    header_end = (
        body_match.start()
        if body_match
        else len(text)
    )

    header_matches = [
        m for m in matches
        if petitioner_match.start() <= m.start() < header_end
        and marker_name(m) != "JUDGMENT"
    ]

    for index, match in enumerate(header_matches):
        name = marker_name(match)

        end = (
            header_matches[index + 1].start()
            if index + 1 < len(header_matches)
            else header_end
        )

        value = clean_text(text[match.end():end])

        if value:
            if fields[name]:
                fields[name] += "\n" + value
            else:
                fields[name] = value

    if body_match:
        fields["JUDGMENT"] = clean_text(
            text[body_match.end():]
        )

    return fields, True


# ---------------------------------------------------------
# 4. Normalize extracted metadata
# ---------------------------------------------------------

def normalize_date(raw_date):
    match = re.search(
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b",
        raw_date or ""
    )

    if not match:
        return None

    value = match.group(1)

    for fmt in ("%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(
                value, fmt
            ).strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None


def normalize_bench(bench_text):
    lines = []
    seen = set()

    for line in bench_text.splitlines():
        line = clean_text(line)

        line = re.sub(
            r"(?i)^BENCH\s*:\s*",
            "",
            line
        ).strip()

        key = re.sub(r"\s+", " ", line).upper()

        if line and key not in seen:
            lines.append(line)
            seen.add(key)

    return lines


# ---------------------------------------------------------
# 5. Parse one judgment
# ---------------------------------------------------------

def parse_judgment(pdf_path):
    extracted = extract_pdf(pdf_path)

    full_text = extracted["text"]
    fields, legacy_header_found = extract_sections(full_text)

    petitioner = re.sub(
        r"\s+\b(?:VS|V)\.?\s*$",
        "",
        fields["PETITIONER"],
        flags=re.IGNORECASE
    ).strip()
    respondent = fields["RESPONDENT"]
    judgment_date = normalize_date(
        fields["DATE OF JUDGMENT"]
    )

    bench = normalize_bench(fields["BENCH"])
    judgment_text = fields["JUDGMENT"]

    quality_flags = []

    if not full_text:
        quality_flags.append("no_extractable_text")

    if not legacy_header_found:
        quality_flags.append("unsupported_header_format")

    if not petitioner:
        quality_flags.append("missing_petitioner")

    if not respondent:
        quality_flags.append("missing_respondent")

    if not judgment_date:
        quality_flags.append("missing_or_invalid_date")

    if not bench:
        quality_flags.append("missing_bench")

    if not judgment_text:
        quality_flags.append("missing_judgment_body")

    elif len(judgment_text) < 300:
        quality_flags.append("short_judgment_body")

    # Missing citations/headnotes are not automatically errors.
    # Some genuine judgments and orders do not contain them.

    if petitioner and respondent:
        title = f"{petitioner} v. {respondent}"
    elif petitioner:
        title = petitioner
    else:
        title = pdf_path.stem

    record = {
        "document_id": f"sc_{pdf_path.stem}",
        "document_type": "supreme_court_judgment",
        "title": title,
        "jurisdiction": "India",
        "court": "Supreme Court of India",
        "judgment_date": judgment_date,
        "petitioner": petitioner,
        "respondent": respondent,
        "bench": bench,
        "citation_block": fields["CITATION"],
        "acts_referred_block": fields["ACT"],
        "headnote": fields["HEADNOTE"],
        "judgment_text": judgment_text,
        "text": full_text,
        "raw_text": extracted["raw_text"],
        "language": "English",
        "metadata": {
            "source_file": pdf_path.name,
            "source_sha256": file_sha256(pdf_path),
            "page_count": extracted["page_count"],
            "character_count": len(full_text),
            "judgment_character_count": len(judgment_text),
            "legacy_header_found": legacy_header_found,
            "quality_flags": quality_flags,
            "parse_status": (
                "review_required"
                if quality_flags
                else "parsed"
            )
        }
    }

    return record


# ---------------------------------------------------------
# 6. Run on only 20 PDFs
# ---------------------------------------------------------

def main():
    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))[:20]

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {PDF_FOLDER}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    records = []
    report = []

    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"\n[{index}/{len(pdf_files)}] {pdf_path.name}")

        try:
            record = parse_judgment(pdf_path)
            records.append(record)

            meta = record["metadata"]

            report.append({
                "source_file": pdf_path.name,
                "status": meta["parse_status"],
                "date": record["judgment_date"],
                "has_petitioner": bool(record["petitioner"]),
                "has_respondent": bool(record["respondent"]),
                "has_bench": bool(record["bench"]),
                "has_citation": bool(record["citation_block"]),
                "has_headnote": bool(record["headnote"]),
                "has_judgment": bool(record["judgment_text"]),
                "page_count": meta["page_count"],
                "character_count": meta["character_count"],
                "judgment_characters": meta["judgment_character_count"],
                "quality_flags": "; ".join(meta["quality_flags"])
            })

            print("Title:", record["title"][:120])
            print("Date:", record["judgment_date"])
            print("Pages:", meta["page_count"])
            print("Judgment chars:", meta["judgment_character_count"])
            print("Status:", meta["parse_status"])
            print("Flags:", meta["quality_flags"])

        except Exception as exc:
            print("ERROR:", exc)

            report.append({
                "source_file": pdf_path.name,
                "status": "extraction_error",
                "quality_flags": str(exc)
            })

    # Save development sample.
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save quality report.
    report_df = pd.DataFrame(report)
    report_df.to_csv(REPORT_FILE, index=False)

    print("\n" + "=" * 70)
    print("PARSING SUMMARY")
    print("=" * 70)

    print("PDFs attempted:", len(pdf_files))
    print("Documents saved:", len(records))

    for field in [
        "petitioner",
        "respondent",
        "judgment_date",
        "bench",
        "citation_block",
        "headnote",
        "judgment_text"
    ]:
        count = sum(bool(r[field]) for r in records)
        print(f"{field}: {count}/{len(records)}")

    flag_counts = Counter(
        flag
        for record in records
        for flag in record["metadata"]["quality_flags"]
    )

    print("\nQuality flags:")
    for flag, count in flag_counts.items():
        print(f"  {flag}: {count}")

    print("\nSaved:", OUTPUT_FILE)
    print("Quality report:", REPORT_FILE)


if __name__ == "__main__":
    main()