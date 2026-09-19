from pathlib import Path
import json


NORMALIZED_FILE = Path(
    "data/processed/central_acts/central_acts_sections.jsonl"
)

RAW_DIR = Path(
    "data/raw/central_acts/annotatedCentralActs"
)

OUTPUT_FILE = Path(
    "outputs/central_acts_sample_validation.txt"
)


def get_source_node(data, pointer):
    """Follow the JSON pointer stored in the normalized record."""
    node = data

    for part in pointer.strip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")

        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node[part]

    return node


def collect_text(value):
    if isinstance(value, str):
        return [value] if value.strip() else []

    if isinstance(value, dict):
        result = []
        for child in value.values():
            result.extend(collect_text(child))
        return result

    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(collect_text(child))
        return result

    return []


def validate_record(record):
    meta = record["metadata"]
    source_path = RAW_DIR / meta["source_file"]

    with open(source_path, "r", encoding="utf-8") as f:
        source = json.load(f)

    original = get_source_node(source, meta["source_path"])

    heading = "\n".join(
        collect_text(original.get("heading", ""))
    ).strip()

    paragraphs = "\n".join(
        collect_text(original.get("paragraphs", {}))
    ).strip()

    normalized_text = record["text"]

    checks = {
        "raw_section_matches_source":
            original == meta["raw_section"],
        "heading_preserved":
            not heading or heading in normalized_text,
        "paragraphs_preserved":
            not paragraphs or paragraphs in normalized_text,
        "source_path_resolves": True
    }

    report = [
        "=" * 80,
        f"ACT: {record['title']}",
        f"FILE: {meta['source_file']}",
        f"SECTION: {meta['section']}",
        f"SOURCE PATH: {meta['source_path']}",
        "",
        "CHECKS:",
        json.dumps(checks, indent=2),
        "",
        "HIERARCHY:",
        json.dumps(meta["hierarchy"], indent=2, ensure_ascii=False),
        "",
        "ORIGINAL HEADING:",
        heading,
        "",
        "ORIGINAL PARAGRAPHS:",
        paragraphs[:6000],
        "",
        "NORMALIZED TEXT:",
        normalized_text[:6000],
        ""
    ]

    return "\n".join(report)


def main():
    first_record = None
    trusts_record = None

    with open(NORMALIZED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if first_record is None:
                first_record = record

            if (
                trusts_record is None
                and record["metadata"]["source_file"] == "188202.json"
                and record["metadata"]["section"].strip() == "Section 1."
            ):
                trusts_record = record

            if first_record and trusts_record:
                break

    if first_record is None:
        raise ValueError("Normalized dataset is empty.")

    if trusts_record is None:
        raise ValueError("Indian Trusts Act Section 1 not found.")

    reports = [
        validate_record(first_record),
        validate_record(trusts_record)
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE.write_text(
        "\n\n".join(reports),
        encoding="utf-8"
    )

    print("Validated 2 section records.")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()