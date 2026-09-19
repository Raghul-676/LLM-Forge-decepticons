import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    r"D:\Projects\LLM_training\data\ILSIC-dataset"
)

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".parquet",
    ".xlsx",
    ".xls",
    ".txt"
}


# ============================================================
# HELPERS
# ============================================================

def shorten(value, limit=500):
    """
    Convert a value to readable text and prevent huge output.
    """

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            default=str
        )
    except Exception:
        text = str(value)

    text = text.replace(
        "\n",
        " "
    )

    if len(text) > limit:
        return text[:limit] + "..."

    return text


def print_separator():
    print(
        "-" * 100
    )


# ============================================================
# CSV INSPECTION
# ============================================================

def inspect_csv(path):

    try:

        df = pd.read_csv(
            path,
            nrows=5
        )

    except UnicodeDecodeError:

        df = pd.read_csv(
            path,
            nrows=5,
            encoding="latin-1"
        )

    except Exception as exc:

        print(
            "Could not read CSV:",
            exc
        )

        return

    print(
        "Columns:"
    )

    for column in df.columns:

        print(
            f"  - {column}"
        )

    print(
        "\nSample rows:"
    )

    for row_number, row in df.iterrows():

        print(
            f"\nROW {row_number + 1}"
        )

        for column in df.columns:

            print(
                f"{column}: "
                f"{shorten(row[column])}"
            )


# ============================================================
# JSONL INSPECTION
# ============================================================

def inspect_jsonl(path):

    samples = []

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:

                    item = json.loads(
                        line
                    )

                    samples.append(
                        item
                    )

                except Exception:
                    continue

                if len(samples) >= 5:
                    break

    except Exception as exc:

        print(
            "Could not read JSONL:",
            exc
        )

        return

    if not samples:

        print(
            "No valid JSONL records found."
        )

        return

    first = samples[0]

    if isinstance(
        first,
        dict
    ):

        print(
            "Keys:"
        )

        for key in first.keys():

            print(
                f"  - {key}"
            )

    print(
        "\nSample records:"
    )

    for number, sample in enumerate(
        samples,
        start=1
    ):

        print(
            f"\nRECORD {number}"
        )

        if isinstance(
            sample,
            dict
        ):

            for key, value in sample.items():

                print(
                    f"{key}: "
                    f"{shorten(value)}"
                )

        else:

            print(
                shorten(sample)
            )


# ============================================================
# JSON INSPECTION
# ============================================================

def inspect_json(path):

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(
                f
            )

    except Exception as exc:

        print(
            "Could not read JSON:",
            exc
        )

        return

    print(
        "Top-level type:",
        type(data).__name__
    )

    # --------------------------------------------------------
    # LIST OF RECORDS
    # --------------------------------------------------------

    if isinstance(
        data,
        list
    ):

        print(
            "Number of items:",
            len(data)
        )

        samples = data[:5]

        for number, sample in enumerate(
            samples,
            start=1
        ):

            print(
                f"\nITEM {number}"
            )

            if isinstance(
                sample,
                dict
            ):

                print(
                    "Keys:",
                    list(
                        sample.keys()
                    )
                )

                for key, value in sample.items():

                    print(
                        f"{key}: "
                        f"{shorten(value)}"
                    )

            else:

                print(
                    shorten(sample)
                )

    # --------------------------------------------------------
    # DICTIONARY
    # --------------------------------------------------------

    elif isinstance(
        data,
        dict
    ):

        print(
            "Top-level keys:"
        )

        for key in list(
            data.keys()
        )[:30]:

            print(
                f"  - {key}"
            )

        print(
            "\nPreview:"
        )

        for key, value in list(
            data.items()
        )[:5]:

            print(
                f"\n{key}:"
            )

            print(
                shorten(
                    value,
                    limit=1000
                )
            )

    else:

        print(
            shorten(
                data,
                limit=1000
            )
        )


# ============================================================
# PARQUET INSPECTION
# ============================================================

def inspect_parquet(path):

    try:

        df = pd.read_parquet(
            path
        )

    except Exception as exc:

        print(
            "Could not read Parquet:",
            exc
        )

        return

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:"
    )

    for column in df.columns:

        print(
            f"  - {column}"
        )

    print(
        "\nFirst 5 rows:"
    )

    for row_number, row in (
        df.head(5).iterrows()
    ):

        print(
            f"\nROW {row_number}"
        )

        for column in df.columns:

            print(
                f"{column}: "
                f"{shorten(row[column])}"
            )


# ============================================================
# EXCEL INSPECTION
# ============================================================

def inspect_excel(path):

    try:

        workbook = pd.ExcelFile(
            path
        )

    except Exception as exc:

        print(
            "Could not open Excel file:",
            exc
        )

        return

    print(
        "Sheets:",
        workbook.sheet_names
    )

    for sheet in workbook.sheet_names[:3]:

        print(
            f"\nSHEET: {sheet}"
        )

        df = pd.read_excel(
            path,
            sheet_name=sheet,
            nrows=5
        )

        print(
            "Columns:",
            list(df.columns)
        )

        print(
            df.head().to_string()
        )


# ============================================================
# TXT INSPECTION
# ============================================================

def inspect_txt(path):

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as f:

            text = f.read(
                1500
            )

    except Exception as exc:

        print(
            "Could not read TXT:",
            exc
        )

        return

    print(
        "Preview:"
    )

    print(
        text
    )


# ============================================================
# FILE INSPECTOR
# ============================================================

def inspect_file(path):

    print_separator()

    print(
        "FILE:",
        path
    )

    try:

        size_mb = (
            path.stat().st_size
            / 1024
            / 1024
        )

        print(
            f"Size: {size_mb:.3f} MB"
        )

    except Exception:

        pass

    suffix = path.suffix.lower()

    print(
        "Type:",
        suffix
    )

    print_separator()

    if suffix == ".csv":

        inspect_csv(
            path
        )

    elif suffix == ".jsonl":

        inspect_jsonl(
            path
        )

    elif suffix == ".json":

        inspect_json(
            path
        )

    elif suffix == ".parquet":

        inspect_parquet(
            path
        )

    elif suffix in {
        ".xlsx",
        ".xls"
    }:

        inspect_excel(
            path
        )

    elif suffix == ".txt":

        inspect_txt(
            path
        )

    else:

        print(
            "Unsupported preview type."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "ILSIC DATASET INSPECTOR"
    )

    print(
        "=" * 100
    )

    print(
        "\nDataset root:"
    )

    print(
        DATASET_ROOT
    )

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"Dataset folder does not exist:\n"
            f"{DATASET_ROOT}"
        )

    # --------------------------------------------------------
    # DISCOVER FILES
    # --------------------------------------------------------

    all_files = sorted(
        [
            path
            for path in DATASET_ROOT.rglob("*")
            if path.is_file()
        ]
    )

    print(
        "\nTotal files found:",
        len(all_files)
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "DATASET FILE TREE"
    )

    print(
        "=" * 100
    )

    for path in all_files:

        relative = path.relative_to(
            DATASET_ROOT
        )

        size_kb = (
            path.stat().st_size
            / 1024
        )

        print(
            f"{relative} "
            f"({size_kb:,.1f} KB)"
        )

    # --------------------------------------------------------
    # SUPPORTED DATA FILES
    # --------------------------------------------------------

    data_files = [
        path
        for path in all_files
        if path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    ]

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SUPPORTED DATA FILES FOUND:"
    )

    print(
        len(data_files)
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # INSPECT EACH DATA FILE
    # --------------------------------------------------------

    for path in data_files:

        inspect_file(
            path
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "INSPECTION COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()