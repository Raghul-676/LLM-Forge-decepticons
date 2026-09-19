from pathlib import Path
from collections import Counter
import json


ACTS_FOLDER = Path(
    "data/raw/central_acts/annotatedCentralActs"
)

OUTPUT_FILE = Path(
    "outputs/central_acts_hierarchy_audit.json"
)


def short_preview(value, limit=300):
    """Compact preview without printing entire legal documents."""
    text = json.dumps(value, ensure_ascii=False)
    return text[:limit] + ("..." if len(text) > limit else "")


def walk(node, path=()):
    """
    Visit every dictionary/list node and preserve its JSON path.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = path + (str(key),)
            yield child_path, value
            yield from walk(value, child_path)

    elif isinstance(node, list):
        for index, value in enumerate(node):
            child_path = path + (str(index),)
            yield child_path, value
            yield from walk(value, child_path)


def is_section_payload(value):
    """
    Recognize the section structure observed in your dataset.
    This is an audit heuristic, not the final parser.
    """
    return (
        isinstance(value, dict)
        and (
            "heading" in value
            or "paragraphs" in value
        )
    )


def inspect_act(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    section_paths = set()
    container_types = Counter()
    unknown_containers = []
    nested_examples = []

    for path, value in walk(data):

        # Locate every Sections container, including nested ones.
        if path[-1].lower() == "sections":

            container_types[type(value).__name__] += 1

            if isinstance(value, dict):

                for section_name, section_data in value.items():

                    if is_section_payload(section_data):

                        section_path = path + (str(section_name),)

                        section_paths.add(section_path)

                        if len(section_path) > 2 and len(nested_examples) < 3:
                            nested_examples.append({
                                "path": " / ".join(section_path),
                                "heading": section_data.get("heading"),
                                "paragraph_keys": list(
                                    section_data.get("paragraphs", {}).keys()
                                ) if isinstance(
                                    section_data.get("paragraphs"), dict
                                ) else []
                            })

            elif isinstance(value, list):
                # Preserve unusual layouts for later inspection.
                unknown_containers.append({
                    "path": " / ".join(path),
                    "type": "list",
                    "preview": short_preview(value[:1])
                })

            else:
                unknown_containers.append({
                    "path": " / ".join(path),
                    "type": type(value).__name__,
                    "preview": short_preview(value)
                })

    return {
        "source_file": file_path.name,
        "act_title": data.get("Act Title"),
        "act_id": data.get("Act ID"),
        "top_level_sections": len(
            data.get("Sections", {})
        ) if isinstance(data.get("Sections"), dict) else 0,
        "sections_discovered": len(section_paths),
        "section_container_types": dict(container_types),
        "unknown_containers": unknown_containers[:3],
        "nested_examples": nested_examples
    }


def main():
    files = sorted(ACTS_FOLDER.glob("*.json"))

    results = []
    errors = []

    for file_path in files:
        try:
            results.append(inspect_act(file_path))
        except Exception as exc:
            errors.append({
                "source_file": file_path.name,
                "error": str(exc)
            })

    total_top_level = sum(
        r["top_level_sections"] for r in results
    )

    total_discovered = sum(
        r["sections_discovered"] for r in results
    )

    zero_section_acts = [
        r for r in results
        if r["sections_discovered"] == 0
    ]

    nested_acts = [
        r for r in results
        if r["sections_discovered"] > r["top_level_sections"]
    ]

    summary = {
        "acts_loaded": len(results),
        "load_errors": len(errors),
        "top_level_sections": total_top_level,
        "sections_discovered_recursively": total_discovered,
        "acts_with_nested_sections": len(nested_acts),
        "acts_with_no_discovered_sections": len(zero_section_acts),
        "zero_section_examples": [
            {
                "source_file": r["source_file"],
                "act_title": r["act_title"]
            }
            for r in zero_section_acts[:10]
        ],
        "nested_section_examples": [
            {
                "source_file": r["source_file"],
                "act_title": r["act_title"],
                "example": r["nested_examples"][0]
            }
            for r in nested_acts
            if r["nested_examples"]
        ][:5],
        "unknown_container_examples": [
            {
                "source_file": r["source_file"],
                "containers": r["unknown_containers"]
            }
            for r in results
            if r["unknown_containers"]
        ][:5],
        "errors": errors
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("=" * 70)
    print("CENTRAL ACTS HIERARCHY AUDIT")
    print("=" * 70)

    for key in [
        "acts_loaded",
        "load_errors",
        "top_level_sections",
        "sections_discovered_recursively",
        "acts_with_nested_sections",
        "acts_with_no_discovered_sections"
    ]:
        print(f"{key}: {summary[key]}")

    print("\nNESTED SECTION EXAMPLES:")
    for example in summary["nested_section_examples"]:
        print(json.dumps(example, indent=2, ensure_ascii=False))

    print("\nZERO-SECTION EXAMPLES:")
    for example in summary["zero_section_examples"]:
        print(json.dumps(example, ensure_ascii=False))

    print("\nUNKNOWN CONTAINER EXAMPLES:")
    for example in summary["unknown_container_examples"]:
        print(json.dumps(example, indent=2, ensure_ascii=False))

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()