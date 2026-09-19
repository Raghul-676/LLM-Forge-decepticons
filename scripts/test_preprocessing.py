from pathlib import Path
import pandas as pd
import json
import re
import fitz


RAW = Path("data/raw")
PROCESSED = Path("data/processed")


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def clean_text(text):
    """
    Basic text cleaning.
    We deliberately keep this conservative for now.
    """

    if text is None:
        return ""

    text = str(text)

    # Normalize newlines
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null characters
    text = text.replace("\x00", "")

    # Collapse excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def save_jsonl(records, output_file):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


# =========================================================
# 1. CONSTITUTION
# =========================================================

print("\n" + "=" * 80)
print("PROCESSING CONSTITUTION")
print("=" * 80)


constitution_file = (
    RAW
    / "constitution"
    / "Constitution Of India.csv"
)


constitution_df = pd.read_csv(
    constitution_file
)


constitution_records = []


for index, row in constitution_df.iterrows():

    text = clean_text(
        row["Articles"]
    )

    if not text:
        continue

    record = {

        "document_id": f"constitution_{index + 1}",

        "document_type": "constitution",

        "title": "Constitution of India",

        "jurisdiction": "India",

        "court": None,

        "date": None,

        "case_number": None,

        "parties": None,

        "bench": None,

        "language": "English",

        "text": text,

        "metadata": {
            "source_row": index
        }
    }

    constitution_records.append(record)


constitution_output = (
    PROCESSED
    / "constitution"
    / "constitution_test.jsonl"
)


save_jsonl(
    constitution_records,
    constitution_output
)


print(
    "Constitution records:",
    len(constitution_records)
)

print(
    "Saved:",
    constitution_output
)


print("\nSample:")

print(
    json.dumps(
        constitution_records[0],
        indent=2,
        ensure_ascii=False
    )
)


# =========================================================
# 2. CENTRAL ACTS
# =========================================================

print("\n" + "=" * 80)
print("PROCESSING CENTRAL ACTS")
print("=" * 80)


acts_folder = (
    RAW
    / "central_acts"
    / "annotatedCentralActs"
)


act_files = list(
    acts_folder.glob("*.json")
)


# ONLY FIRST 5 FOR TESTING
act_files = act_files[:5]


act_records = []


for file in act_files:

    with open(
        file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    act_title = clean_text(
        data.get("Act Title", "")
    )

    act_id = clean_text(
        data.get("Act ID", "")
    )

    enactment_date = clean_text(
        data.get("Enactment Date", "")
    )


    sections = data.get(
        "Sections",
        {}
    )


    for section_number, section_data in sections.items():

        heading = clean_text(
            section_data.get(
                "heading",
                ""
            )
        )


        paragraphs = section_data.get(
            "paragraphs",
            {}
        )


        paragraph_text = "\n".join(
            clean_text(value)
            for value in paragraphs.values()
        )


        final_text = (
            f"{section_number}\n"
            f"{heading}\n\n"
            f"{paragraph_text}"
        )


        final_text = clean_text(
            final_text
        )


        if not final_text:
            continue


        record = {

            "document_id":
                f"act_{file.stem}_{section_number}",

            "document_type":
                "central_act_section",

            "title":
                act_title,

            "jurisdiction":
                "India",

            "court":
                None,

            "date":
                enactment_date,

            "case_number":
                None,

            "parties":
                None,

            "bench":
                None,

            "language":
                "English",

            "text":
                final_text,

            "metadata": {

                "act_id":
                    act_id,

                "section":
                    section_number,

                "section_heading":
                    heading,

                "source_file":
                    file.name
            }
        }


        act_records.append(
            record
        )


acts_output = (
    PROCESSED
    / "central_acts"
    / "acts_test.jsonl"
)


save_jsonl(
    act_records,
    acts_output
)


print(
    "Sections processed:",
    len(act_records)
)

print(
    "Saved:",
    acts_output
)


if act_records:

    print("\nSample:")

    print(
        json.dumps(
            act_records[0],
            indent=2,
            ensure_ascii=False
        )
    )


# =========================================================
# 3. SUPREME COURT
# =========================================================

print("\n" + "=" * 80)
print("TESTING SUPREME COURT PDF EXTRACTION")
print("=" * 80)


sc_folder = (
    RAW
    / "supreme_court"
)


metadata_file = (
    sc_folder
    / "judgments.csv"
)


pdf_folder = (
    sc_folder
    / "pdfs"
)


metadata_df = pd.read_csv(
    metadata_file
)


print(
    "Metadata rows:",
    len(metadata_df)
)


pdf_files = list(
    pdf_folder.glob("*.pdf")
)


print(
    "PDF files:",
    len(pdf_files)
)


# ONLY FIRST 10 PDFs
sample_pdfs = pdf_files[:10]


sc_records = []


for pdf_file in sample_pdfs:

    print(
        "\nExtracting:",
        pdf_file.name
    )


    try:

        document = fitz.open(
            pdf_file
        )


        pages = []


        for page in document:

            page_text = page.get_text(
                "text"
            )

            pages.append(
                page_text
            )


        document.close()


        full_text = "\n".join(
            pages
        )


        full_text = clean_text(
            full_text
        )


        record = {

            "document_id":
                f"sc_{pdf_file.stem}",

            "document_type":
                "supreme_court_judgment",

            "title":
                pdf_file.stem,

            "jurisdiction":
                "India",

            "court":
                "Supreme Court of India",

            "date":
                None,

            "case_number":
                None,

            "parties":
                None,

            "bench":
                None,

            "language":
                None,

            "text":
                full_text,

            "metadata": {

                "source_file":
                    pdf_file.name,

                "page_count":
                    len(pages),

                "character_count":
                    len(full_text)
            }
        }


        sc_records.append(
            record
        )


        print(
            "Pages:",
            len(pages)
        )

        print(
            "Characters extracted:",
            len(full_text)
        )


        print("\nPreview:")

        print(
            full_text[:500]
        )


    except Exception as e:

        print(
            "ERROR:",
            e
        )


sc_output = (
    PROCESSED
    / "supreme_court"
    / "sc_test.jsonl"
)


save_jsonl(
    sc_records,
    sc_output
)


print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)


print(
    "Constitution records:",
    len(constitution_records)
)

print(
    "Act section records:",
    len(act_records)
)

print(
    "Supreme Court PDFs tested:",
    len(sc_records)
)