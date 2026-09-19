from pathlib import Path
import re

import fitz
import pandas as pd
from difflib import SequenceMatcher


SC_ROOT = Path("data/raw/supreme_court")
PDF_FOLDER = SC_ROOT / "pdfs"
CSV_FILE = SC_ROOT / "judgments.csv"

OUTPUT_FILE = Path("outputs/sc_metadata_link_test.csv")


# =========================================================
# TEXT UTILITIES
# =========================================================

def normalize_name(value):
    """
    Normalize names so formatting differences do not
    badly affect fuzzy matching.
    """

    if pd.isna(value):
        return ""

    value = str(value).upper()

    value = value.replace("&", " AND ")

    # Remove punctuation
    value = re.sub(r"[^A-Z0-9 ]+", " ", value)

    # Collapse whitespace
    value = re.sub(r"\s+", " ", value)

    return value.strip()

def similarity_score(text1, text2):
    """
    Return similarity score from 0 to 100.
    """

    if not text1 or not text2:
        return 0.0

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio() * 100

def extract_between(text, start_pattern, end_pattern):

    pattern = (
        start_pattern
        + r"\s*(.*?)\s*"
        + end_pattern
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if match:
        return match.group(1).strip()

    return ""


def parse_pdf_header(pdf_path):

    document = fitz.open(pdf_path)

    # Usually the useful header is on page 1.
    first_page = document[0].get_text("text")

    document.close()

    petitioner = extract_between(
        first_page,
        r"PETITIONER:",
        r"Vs\."
    )

    respondent = extract_between(
        first_page,
        r"RESPONDENT:",
        r"DATE\s+OF\s+JUDGMENT"
    )

    date_match = re.search(
        r"DATE\s+OF\s+JUDGMENT\s*:?\s*"
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
        first_page,
        flags=re.IGNORECASE
    )

    judgment_date = (
        date_match.group(1)
        if date_match
        else None
    )

    return {
        "petitioner": petitioner,
        "respondent": respondent,
        "judgment_date": judgment_date
    }


# =========================================================
# LOAD CSV
# =========================================================

print("=" * 80)
print("SUPREME COURT METADATA LINKING TEST")
print("=" * 80)

metadata = pd.read_csv(CSV_FILE)

print("\nCSV rows:", len(metadata))

metadata["parsed_date"] = pd.to_datetime(
    metadata["judgment_dates"],
    dayfirst=True,
    errors="coerce"
)

print(
    "Valid dates:",
    metadata["parsed_date"].notna().sum()
)

print(
    "Invalid/missing dates:",
    metadata["parsed_date"].isna().sum()
)


# =========================================================
# SHOW DATE RANGE
# =========================================================

valid_dates = metadata["parsed_date"].dropna()

if len(valid_dates):

    print("\nMetadata date range:")

    print(
        valid_dates.min().date(),
        "→",
        valid_dates.max().date()
    )


print("\nCases by year (first 20 years):")

year_counts = (
    metadata["parsed_date"]
    .dt.year
    .value_counts()
    .sort_index()
)

print(year_counts.head(20).to_string())


# =========================================================
# PREPARE NORMALIZED COLUMNS
# =========================================================

metadata["norm_pet"] = (
    metadata["pet"]
    .apply(normalize_name)
)

metadata["norm_res"] = (
    metadata["res"]
    .apply(normalize_name)
)


# =========================================================
# MATCH TEST PDFs
# =========================================================

pdf_files = sorted(
    PDF_FOLDER.glob("*.pdf")
)

# Test first 20 only
sample_pdfs = pdf_files[:20]

results = []


for index, pdf_file in enumerate(sample_pdfs, start=1):

    print("\n" + "-" * 80)

    print(
        f"[{index}/{len(sample_pdfs)}]",
        pdf_file.name
    )

    try:

        header = parse_pdf_header(
            pdf_file
        )

    except Exception as exc:

        print("PDF ERROR:", exc)

        continue


    pdf_pet = header["petitioner"]
    pdf_res = header["respondent"]
    pdf_date_raw = header["judgment_date"]


    print("PDF petitioner :", pdf_pet[:120])
    print("PDF respondent :", pdf_res[:120])
    print("PDF date       :", pdf_date_raw)


    pdf_date = pd.to_datetime(
        pdf_date_raw,
        dayfirst=True,
        errors="coerce"
    )


    # -----------------------------------------------------
    # FILTER BY EXACT DATE
    # -----------------------------------------------------

    if pd.isna(pdf_date):

        candidates = metadata.iloc[0:0]

    else:

        candidates = metadata[
            metadata["parsed_date"] == pdf_date
        ]


    print(
        "CSV candidates on same date:",
        len(candidates)
    )


    norm_pdf_pet = normalize_name(
        pdf_pet
    )

    norm_pdf_res = normalize_name(
        pdf_res
    )


    best_match = None
    best_score = -1


    # -----------------------------------------------------
    # FUZZY PARTY MATCHING
    # -----------------------------------------------------

    for row_index, row in candidates.iterrows():

        pet_score = similarity_score(
            norm_pdf_pet,
            row["norm_pet"]
        )


        if norm_pdf_res and row["norm_res"]:

            res_score = similarity_score(
                norm_pdf_res,
                row["norm_res"]
            )

            combined_score = (
                pet_score * 0.55
                +
                res_score * 0.45
            )

        else:

            combined_score = pet_score


        if combined_score > best_score:

            best_score = combined_score

            best_match = row


    # -----------------------------------------------------
    # REPORT
    # -----------------------------------------------------

    if best_match is not None:

        print(
            "Best score:",
            round(best_score, 2)
        )

        print(
            "CSV petitioner:",
            best_match["pet"]
        )

        print(
            "CSV respondent:",
            best_match["res"]
        )

        print(
            "CSV case no:",
            best_match["case_no"]
        )


        if best_score >= 85:

            status = "HIGH_CONFIDENCE"

        elif best_score >= 65:

            status = "REVIEW"

        else:

            status = "LOW_CONFIDENCE"


        results.append({

            "pdf_file":
                pdf_file.name,

            "pdf_date":
                pdf_date_raw,

            "pdf_petitioner":
                pdf_pet,

            "pdf_respondent":
                pdf_res,

            "candidate_count":
                len(candidates),

            "match_score":
                round(best_score, 2),

            "status":
                status,

            "csv_diary_no":
                best_match["diary_no"],

            "csv_case_no":
                best_match["case_no"],

            "csv_petitioner":
                best_match["pet"],

            "csv_respondent":
                best_match["res"],

            "csv_date":
                best_match["judgment_dates"],

            "csv_bench":
                best_match["bench"],

            "csv_judgement_by":
                best_match["judgement_by"],

            "csv_temp_link":
                best_match["temp_link"]
        })


    else:

        print("NO MATCH FOUND")

        results.append({

            "pdf_file":
                pdf_file.name,

            "pdf_date":
                pdf_date_raw,

            "pdf_petitioner":
                pdf_pet,

            "pdf_respondent":
                pdf_res,

            "candidate_count":
                len(candidates),

            "match_score":
                None,

            "status":
                "NO_MATCH",

            "csv_diary_no":
                None,

            "csv_case_no":
                None,

            "csv_petitioner":
                None,

            "csv_respondent":
                None,

            "csv_date":
                None,

            "csv_bench":
                None,

            "csv_judgement_by":
                None,

            "csv_temp_link":
                None
        })


# =========================================================
# SAVE REPORT
# =========================================================

result_df = pd.DataFrame(
    results
)


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 80)
print("MATCHING SUMMARY")
print("=" * 80)


print(
    result_df["status"]
    .value_counts()
    .to_string()
)


if result_df["match_score"].notna().any():

    print(
        "\nAverage match score:",
        round(
            result_df["match_score"].mean(),
            2
        )
    )


print(
    "\nReport saved to:",
    OUTPUT_FILE
)

print("=" * 80)