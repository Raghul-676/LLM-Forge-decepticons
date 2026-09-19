
from pathlib import Path
from datetime import datetime
import re

import fitz

from parse_sc_judgments_test import (
    parse_judgment as parse_legacy,
    extract_pdf,
    clean_text,
    file_sha256,
)


# Matches modern headings such as:
# JUDGMENT, J U D G M E N T, ORDER, O R D E R
BODY_RE = re.compile(
    r"(?im)^[ \t]*"
    r"(?:J\s*U\s*D\s*G\s*M\s*E\s*N\s*T"
    r"|O\s*R\s*D\s*E\s*R)"
    r"[ \t]*[:.]?[ \t]*$"
)

VERSUS_RE = re.compile(
    r"(?im)^[ \t]*(?:VERSUS|VS?\.?)[ \t]*$"
)

CASE_RE = re.compile(
    r"(?im)^[ \t]*"
    r"(?:CIVIL|CRIMINAL|WRIT|TRANSFER|SPECIAL LEAVE|"
    r"REVIEW|CURATIVE|CONTEMPT|MISCELLANEOUS)"
    r"[^\n]{0,140}\b(?:NO|NOS)\.?\s*[-:]*\s*"
    r"[A-Z0-9][^\n]*"
)

ROLE = (
    r"(?:APPELLANT|RESPONDENT|PETITIONER|APPLICANT)"
    r"(?:S|\s*\(\s*S\s*\))?"
)


def normalize_date(value):
    """Parse only an explicitly supplied date."""
    if not value:
        return None

    value = re.sub(
        r"(\d{1,2})(st|nd|rd|th)\b",
        r"\1",
        value,
        flags=re.IGNORECASE
    )

    value = re.sub(r"\s+", " ", value).strip()

    for fmt in (
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%d %B %Y",
        "%d %b %Y",
        "%d-%B-%Y",
        "%d-%b-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ):
        try:
            return datetime.strptime(
                value, fmt
            ).strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None


def extract_explicit_date(header):
    """
    Only accept a date associated with an explicit judgment
    date label. Do not use the case-number year or filename.
    """
    pattern = re.compile(
        r"(?im)^[ \t]*"
        r"(?:DATE OF JUDGMENT|DATE OF DECISION|"
        r"PRONOUNCED ON|DATE OF PRONOUNCEMENT|DECIDED ON)"
        r"[ \t]*:?[ \t]*(?P<value>[^\n]{0,100})"
    )

    date_pattern = re.compile(
        r"\b(?:"
        r"\d{1,2}[./-]\d{1,2}[./-]\d{4}"
        r"|\d{1,2}(?:st|nd|rd|th)?[ -]+"
        r"[A-Za-z]+[ ,.-]+\d{4}"
        r"|[A-Za-z]+[ -]+\d{1,2},?[ -]+\d{4}"
        r")\b",
        flags=re.IGNORECASE
    )

    for match in pattern.finditer(header):
        candidate = date_pattern.search(match.group("value"))

        if candidate:
            parsed = normalize_date(candidate.group(0))

            if parsed:
                return parsed

    return None


def clean_party(value):
    """Remove layout labels, not actual party names."""
    lines = []

    for line in value.splitlines():
        line = line.strip()

        if not line or re.fullmatch(r"\d+", line):
            continue

        if re.match(
            r"(?i)^\(?ARISING\s+OUT\s+OF\b",
            line
        ):
            continue

        if re.fullmatch(r"[.…\s–—-]+", line):
            continue

        # Remove a trailing role label:
        # "... Appellant(s)" or "... Respondents"
        line = re.sub(
            rf"(?i)(?:\s*[.…–—-]+\s*|\s+){ROLE}\s*$",
            "",
            line
        ).strip()

        if line:
            lines.append(line)

    return clean_text("\n".join(lines))


def extract_parties(header):
    """
    Extract parties only when a recognizable case-number
    and standalone Versus separator are present.
    """
    versus = VERSUS_RE.search(header)
    case_matches = list(CASE_RE.finditer(header))

    if not versus or not case_matches:
        return "", "", None

    # Use the last case-number line before Versus.
    preceding_cases = [
        m for m in case_matches
        if m.end() < versus.start()
    ]

    if not preceding_cases:
        return "", "", None

    case_match = preceding_cases[-1]
    case_number = clean_text(case_match.group(0))

    petitioner_block = header[
        case_match.end():versus.start()
    ]

    respondent_block = header[versus.end():]

    # Stop at the first explicit respondent role label,
    # so later Coram or administrative text is not included.
    respondent_role = re.search(
        rf"(?im)^[ \t]*[.…–—-]*[ \t]*{ROLE}[ \t]*$",
        respondent_block
    )

    if respondent_role:
        respondent_block = respondent_block[:respondent_role.start()]

    petitioner = clean_party(petitioner_block)
    respondent = clean_party(respondent_block)

    petitioner = re.sub(
        r"(?i)\s+\b(?:VERSUS|VS|V)\.?\s*$",
        "",
        petitioner
    ).strip()

    return petitioner, respondent, case_number


def extract_author(body):
    """
    Identify an explicitly printed judgment author.
    This is not necessarily the complete judicial bench.
    """
    match = re.search(
        r"(?im)^[ \t]*"
        r"([^\n]{3,120}?,[ \t]*(?:J\.?|C\.?J\.?I\.?|JUDGE))"
        r"[ \t]*:?[ \t]*$",
        body[:700]
    )

    return clean_text(match.group(1)) if match else None


def extract_coram(header):
    """Extract a bench only when an explicit CORAM block exists."""
    match = re.search(
        r"(?im)^[ \t]*CORAM[ \t]*:[ \t]*",
        header
    )

    if not match:
        return []

    block = header[match.end():]

    stop = re.search(
        r"(?im)^[ \t]*(?:FOR THE|APPEARANCES|COUNSEL|"
        r"JUDGMENT|ORDER|DATE OF JUDGMENT)[ \t]*:",
        block
    )

    if stop:
        block = block[:stop.start()]

    lines = [
        clean_text(line)
        for line in block.splitlines()
        if clean_text(line)
    ]

    return lines[:10]


def parse_modern(pdf_path):
    extracted = extract_pdf(pdf_path)
    full_text = extracted["text"]

    # Look for a document heading near the beginning.
    body_match = BODY_RE.search(full_text[:15000])

    if body_match:
        header = full_text[:body_match.start()]
        judgment_text = clean_text(
            full_text[body_match.end():]
        )
        body_method = "modern_heading"
    else:
        header = full_text[:12000]
        judgment_text = ""
        body_method = "not_found"

    petitioner, respondent, case_number = extract_parties(
        header
    )

    judgment_date = extract_explicit_date(header)
    bench = extract_coram(header)
    author = extract_author(judgment_text)

    flags = []

    if not full_text:
        flags.append("no_extractable_text")

    if not body_match:
        flags.append("missing_judgment_body")

    if not case_number:
        flags.append("missing_case_number")

    if not petitioner:
        flags.append("missing_petitioner")

    if not respondent:
        flags.append("missing_respondent")

    if not judgment_date:
        flags.append("missing_or_invalid_date")

    if not bench:
        flags.append("missing_bench")

    if judgment_text and len(judgment_text) < 300:
        flags.append("short_judgment_body")

    if petitioner and respondent:
        title = f"{petitioner} v. {respondent}"
    else:
        title = petitioner or respondent or pdf_path.stem

    return {
        "document_id": f"sc_{pdf_path.stem}",
        "document_type": "supreme_court_judgment",
        "title": title,
        "jurisdiction": "India",
        "court": "Supreme Court of India",
        "case_number": case_number,
        "judgment_date": judgment_date,
        "petitioner": petitioner,
        "respondent": respondent,
        "bench": bench,
        "judgment_author": author,
        "citation_block": "",
        "acts_referred_block": "",
        "headnote": "",
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
            "document_format": "modern",
            "body_extraction_method": body_method,
            "text_status": (
                "usable"
                if judgment_text and len(judgment_text) >= 300
                else "review_required"
            ),
            "quality_flags": flags,
            "parse_status": (
                "review_required" if flags else "parsed"
            )
        }
    }


def parse_judgment(pdf_path):
    """
    Public entry point. Select the correct parser based on
    the PDF's layout.
    """
    pdf_path = Path(pdf_path)

    if pdf_path.stat().st_size == 0:
        raise ValueError("empty_file")

    with fitz.open(pdf_path) as document:
        first_page = document[0].get_text("text")

    is_legacy = bool(
        re.search(
            r"(?im)^[ \t]*PETITIONER[ \t]*:",
            first_page
        )
        and re.search(
            r"(?im)^[ \t]*DATE[ \t]+OF[ \t]+JUDGMENT",
            first_page
        )
    )

    if is_legacy:
        record = parse_legacy(pdf_path)

        record["metadata"]["document_format"] = "legacy_judis"
        record["metadata"]["text_status"] = (
            "usable"
            if len(record["judgment_text"]) >= 300
            else "review_required"
        )

        return record

    return parse_modern(pdf_path)


# ---------------------------------------------------------
# Smoke test: two known modern PDFs
# ---------------------------------------------------------

if __name__ == "__main__":
    folder = Path("data/raw/supreme_court/pdfs")

    filenames = [
        "10506-2007___jonew__judis__36954.pdf",
        "10138-2013___supremecourt__2013__10138__10138_2013_2_1501_43180_Judgement_28-Mar-2023.pdf",
    ]

    for filename in filenames:
        path = folder / filename

        print("\n" + "=" * 70)
        print(filename)

        if not path.exists():
            print("FILE NOT FOUND")
            continue

        try:
            record = parse_judgment(path)

            print("Format:", record["metadata"]["document_format"])
            print("Title:", record["title"])
            print("Case number:", record.get("case_number"))
            print("Date:", record["judgment_date"])
            print("Author:", record.get("judgment_author"))
            print("Bench:", record["bench"])
            print("Body chars:", len(record["judgment_text"]))
            print("Text status:", record["metadata"]["text_status"])
            print("Flags:", record["metadata"]["quality_flags"])

            print("\nBODY PREVIEW:")
            print(record["judgment_text"][:600])

        except Exception as exc:
            print("ERROR:", exc)