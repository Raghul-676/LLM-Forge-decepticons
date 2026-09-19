from pathlib import Path
import json
import pandas as pd


ROOT = Path("data/raw")


print("=" * 80)
print("LEGAL DATASET INVENTORY")
print("=" * 80)


# ---------------------------------------------------------
# 1. Show files and sizes
# ---------------------------------------------------------

for folder in [
    "constitution",
    "central_acts",
    "supreme_court",
    "high_court"
]:
    path = ROOT / folder

    print(f"\n[{folder.upper()}]")

    if not path.exists():
        print("Folder does not exist.")
        continue

    files = list(path.rglob("*"))

    actual_files = [f for f in files if f.is_file()]

    print("Number of files:", len(actual_files))

    for file in actual_files[:10]:
        size_mb = file.stat().st_size / (1024 ** 2)

        print(
            f"  {file.relative_to(ROOT)} "
            f"({size_mb:.2f} MB)"
        )

    if len(actual_files) > 10:
        print(f"  ... and {len(actual_files) - 10} more files")


# ---------------------------------------------------------
# 2. Inspect Constitution CSV
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CONSTITUTION CSV")
print("=" * 80)

constitution_files = list(
    (ROOT / "constitution").glob("*.csv")
)

if constitution_files:

    csv_file = constitution_files[0]

    print("File:", csv_file)

    df = pd.read_csv(csv_file)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

else:
    print("No CSV found.")


# ---------------------------------------------------------
# 3. Inspect one Central Act JSON
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CENTRAL ACT JSON")
print("=" * 80)

json_files = list(
    (ROOT / "central_acts").rglob("*.json")
)

if json_files:

    sample_file = json_files[0]

    print("Sample file:")
    print(sample_file)

    with open(sample_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\nTop-level type:")
    print(type(data).__name__)

    if isinstance(data, dict):

        print("\nTop-level keys:")
        print(list(data.keys()))

        print("\nSample values:")

        for key in list(data.keys())[:10]:

            value = data[key]

            preview = str(value)

            if len(preview) > 500:
                preview = preview[:500] + "..."

            print(f"\n{key}:")
            print(preview)

else:
    print("No JSON files found.")


# ---------------------------------------------------------
# 4. Inspect Supreme Court dataset
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("SUPREME COURT DATASET")
print("=" * 80)

sc_path = ROOT / "supreme_court"

extensions = {}

if sc_path.exists():

    for file in sc_path.rglob("*"):

        if file.is_file():

            suffix = file.suffix.lower()

            extensions[suffix] = extensions.get(suffix, 0) + 1


print("File types found:")

for extension, count in extensions.items():
    print(extension or "[no extension]", ":", count)


# CSV
csv_files = list(sc_path.rglob("*.csv"))

if csv_files:

    print("\nSample CSV:")
    print(csv_files[0])

    df = pd.read_csv(csv_files[0])

    print("Shape:", df.shape)

    print("Columns:")
    print(df.columns.tolist())

    print("\nFirst 2 rows:")
    print(df.head(2).to_string())


# JSON
sc_json_files = list(sc_path.rglob("*.json"))

if sc_json_files:

    print("\nSample JSON:")
    print(sc_json_files[0])

    with open(
        sc_json_files[0],
        "r",
        encoding="utf-8"
    ) as f:

        sample_json = json.load(f)

    if isinstance(sample_json, dict):

        print("Keys:")
        print(list(sample_json.keys()))


print("\n" + "=" * 80)
print("INSPECTION FINISHED")
print("=" * 80)