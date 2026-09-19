
from pathlib import Path
from collections import Counter
import hashlib
import json


RAW_DIR = Path(
    "data/raw/central_acts/annotatedCentralActs"
)

OUT_DIR = Path("data/processed/central_acts")
REPORT_DIR = Path("outputs")

SECTIONS_FILE = OUT_DIR / "central_acts_sections.jsonl"
SUPPLEMENT_FILE = OUT_DIR / "central_acts_supplementary.jsonl"
REPORT_FILE = REPORT_DIR / "central_acts_normalization_report.json"

SOURCE_URL = "https://zenodo.org/records/5088102"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_pointer(path):
    """Preserve the exact location of a node in the source JSON."""
    def escape(value):
        return str(value).replace("~", "~0").replace("/", "~1")

    return "/" + "/".join(escape(p) for p in path)


def collect_text(value):
    """
    Collect text recursively in source order.
    Do not remove amendment notes, footnote markers, or
    other legal content.
    """
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


def section_payload(value):
    return (
        isinstance(value, dict)
        and ("heading" in value or "paragraphs" in value)
    )


def render_section(section_name, payload):
    heading = payload.get("heading", "")
    paragraphs = payload.get("paragraphs", {})

    heading_text = "\n".join(collect_text(heading)).strip()
    paragraph_text = "\n".join(
        collect_text(paragraphs)
    ).strip()

    first_line = str(section_name).strip()

    if heading_text:
        first_line += " " + heading_text

    return "\n".join(
        part for part in [first_line, paragraph_text] if part
    )


def hierarchy_metadata(node):
    """
    Keep descriptive fields from hierarchy nodes, but do
    not copy complete nested section collections into
    every section record.
    """
    if not isinstance(node, dict):
        return {}

    result = {}

    for key, value in node.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value

    return result


def normalize_act(path, section_writer, supplement_writer):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_hash = sha256_file(path)
    act_title = data.get("Act Title", "")
    act_id = data.get("Act ID", "")

    sections = []
    issues = []
    seen_paths = set()

    def walk(node, current_path=(), ancestors=()):
        """
        Recursively traverse the source tree.

        Recognized section payloads are written separately.
        Everything else is retained in the supplementary
        source tree, including unusual/unrecognized content.
        """
        if isinstance(node, dict):
            result = {}

            for key, value in node.items():
                child_path = current_path + (str(key),)

                if key.lower() == "sections" and isinstance(value, dict):
                    remaining = {}

                    for section_name, payload in value.items():
                        section_path = child_path + (str(section_name),)

                        if section_payload(payload):
                            pointer = json_pointer(section_path)

                            if pointer in seen_paths:
                                issues.append({
                                    "type": "duplicate_source_path",
                                    "path": pointer
                                })
                                remaining[section_name] = payload
                                continue

                            seen_paths.add(pointer)

                            text = render_section(
                                section_name, payload
                            )

                            if not text.strip():
                                issues.append({
                                    "type": "empty_section_text",
                                    "path": pointer
                                })

                            extra_keys = [
                                k for k in payload
                                if k not in ("heading", "paragraphs")
                            ]

                            if extra_keys:
                                issues.append({
                                    "type": "extra_section_fields",
                                    "path": pointer,
                                    "keys": extra_keys
                                })

                            unique_key = (
                                path.name + "|" + pointer
                            )

                            document_id = "act_" + hashlib.sha256(
                                unique_key.encode("utf-8")
                            ).hexdigest()[:20]

                            record = {
                                "document_id": document_id,
                                "document_type": "central_act_section",
                                "title": act_title,
                                "jurisdiction": "India",
                                "text": text,
                                "metadata": {
                                    "act_id": act_id,
                                    "enactment_date_raw":
                                        data.get("Enactment Date"),
                                    "section": str(section_name),
                                    "heading": payload.get("heading"),
                                    "hierarchy": list(ancestors),
                                    "source_file": path.name,
                                    "source_path": pointer,
                                    "source_sha256": source_hash,
                                    "source_url": SOURCE_URL,
                                    "source_version":
                                        "historical_snapshot",
                                    "legal_status": "unverified",
                                    "raw_section": payload
                                }
                            }

                            sections.append(record)

                        else:
                            issues.append({
                                "type": "unrecognized_section_payload",
                                "path": json_pointer(section_path),
                                "value_type": type(payload).__name__
                            })

                            # Preserve the original content rather than
                            # silently discarding an unknown structure.
                            remaining[section_name] = walk(
                                payload,
                                section_path,
                                ancestors
                            )

                    result[key] = remaining

                else:
                    next_ancestors = ancestors

                    if isinstance(value, dict):
                        next_ancestors = ancestors + ({
                            "path": json_pointer(child_path),
                            "metadata": hierarchy_metadata(value)
                        },)

                    result[key] = walk(
                        value,
                        child_path,
                        next_ancestors
                    )

            return result

        if isinstance(node, list):
            return [
                walk(
                    value,
                    current_path + (str(index),),
                    ancestors + ({
                        "path": json_pointer(
                            current_path + (str(index),)
                        ),
                        "metadata": hierarchy_metadata(value)
                    },) if isinstance(value, dict) else ancestors
                )
                for index, value in enumerate(node)
            ]

        return node

    supplementary_tree = walk(data)

    for record in sections:
        section_writer.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

    # Keep the non-section source tree separately. This
    # retains definitions, schedules, forms, footnotes,
    # hierarchy labels, and unrecognized structures.
    supplement_writer.write(
        json.dumps({
            "document_type": "central_act_supplementary",
            "act_id": act_id,
            "act_title": act_title,
            "source_file": path.name,
            "source_sha256": source_hash,
            "source_url": SOURCE_URL,
            "source_version": "historical_snapshot",
            "legal_status": "unverified",
            "supplementary_tree": supplementary_tree
        }, ensure_ascii=False) + "\n"
    )

    return {
        "source_file": path.name,
        "act_id": act_id,
        "act_title": act_title,
        "sections_written": len(sections),
        "issues": issues
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.json"))

    results = []
    errors = []
    issue_counts = Counter()
    total_sections = 0

    with open(
        SECTIONS_FILE, "w", encoding="utf-8"
    ) as section_writer, open(
        SUPPLEMENT_FILE, "w", encoding="utf-8"
    ) as supplement_writer:

        for index, path in enumerate(files, start=1):
            try:
                result = normalize_act(
                    path,
                    section_writer,
                    supplement_writer
                )

                results.append(result)
                total_sections += result["sections_written"]

                for issue in result["issues"]:
                    issue_counts[issue["type"]] += 1

                if index % 50 == 0 or index == len(files):
                    print(
                        f"[{index}/{len(files)}] "
                        f"Sections written: {total_sections}"
                    )

            except Exception as exc:
                errors.append({
                    "source_file": path.name,
                    "error": str(exc)
                })
                print("ERROR:", path.name, exc)

    summary = {
        "acts_found": len(files),
        "acts_processed": len(results),
        "processing_errors": len(errors),
        "sections_written": total_sections,
        "hierarchy_audit_benchmark": 35864,
        "matches_hierarchy_audit": total_sections == 35864,
        "acts_with_zero_sections": [
            r["source_file"] for r in results
            if r["sections_written"] == 0
        ],
        "issue_counts": dict(issue_counts),
        "errors": errors,
        "issue_examples": [
            {
                "source_file": r["source_file"],
                "issues": r["issues"][:3]
            }
            for r in results if r["issues"]
        ][:10]
    }

    REPORT_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("CENTRAL ACTS NORMALIZATION SUMMARY")
    print("=" * 70)

    for key, value in summary.items():
        if key not in ("errors", "issue_examples"):
            print(f"{key}: {value}")

    print("\nSaved:", SECTIONS_FILE)
    print("Saved:", SUPPLEMENT_FILE)
    print("Saved:", REPORT_FILE)


if __name__ == "__main__":
    main()