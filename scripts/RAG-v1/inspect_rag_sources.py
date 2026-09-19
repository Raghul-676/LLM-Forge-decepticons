from pathlib import Path
import json


FILES = {
    "CENTRAL ACTS": Path(
        "data/processed/central_acts/central_acts_sections.jsonl"
    ),

    "CONSTITUTION": Path(
        "data/processed/constitution/constitution_pages_structured.jsonl"
    ),

    "SUPREME COURT": Path(
        "data/processed/supreme_court/sc_pilot_200_v2.jsonl"
    ),
}


def short_value(value, limit=500):
    if isinstance(value, (dict, list)):
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2
        )
    else:
        text = str(value)

    if len(text) > limit:
        return text[:limit] + "... [TRUNCATED]"

    return text


def inspect_jsonl(name, path):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print("Path:", path)

    if not path.exists():
        print("ERROR: FILE NOT FOUND")
        return

    record_count = 0
    first_record = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record_count += 1

            if first_record is None:
                first_record = json.loads(line)

    print("Records:", record_count)

    if first_record is None:
        print("ERROR: NO RECORDS")
        return

    print("\nTOP-LEVEL KEYS:")

    for key in first_record.keys():
        print(f"  - {key}")

    print("\nFIRST RECORD FIELD PREVIEW:")

    for key, value in first_record.items():
        print("\n" + "-" * 60)
        print("FIELD:", key)
        print("TYPE:", type(value).__name__)
        print("VALUE:")
        print(short_value(value))


def main():
    print("=" * 80)
    print("RAG SOURCE SCHEMA INSPECTION")
    print("=" * 80)

    for name, path in FILES.items():
        inspect_jsonl(name, path)


if __name__ == "__main__":
    main()