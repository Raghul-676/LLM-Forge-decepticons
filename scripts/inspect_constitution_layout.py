from pathlib import Path
import fitz


PDF_FILE = Path(
    "data/raw/constitution/constitution_official_2024.pdf"
)

OUTPUT_FILE = Path(
    "outputs/constitution_layout_inspection.txt"
)

# PDF page numbers, not printed page labels.
# These include the beginning of the body and Article 21.
PAGES_TO_INSPECT = [33, 34, 42, 43]


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    report = []

    with fitz.open(PDF_FILE) as pdf:

        for page_number in PAGES_TO_INSPECT:

            page = pdf[page_number - 1]

            report.append("=" * 80)
            report.append(f"PDF PAGE {page_number}")
            report.append(
                f"Page dimensions: {page.rect.width:.1f} x "
                f"{page.rect.height:.1f}"
            )

            # Original reading-order extraction.
            report.append("\n--- EXTRACTED TEXT ---\n")
            report.append(page.get_text("text"))

            # Text blocks include coordinates. This helps us
            # distinguish columns, body text, and footnotes.
            report.append("\n--- TEXT BLOCKS AND POSITIONS ---\n")

            blocks = page.get_text("blocks")

            for index, block in enumerate(blocks, start=1):
                x0, y0, x1, y1, text, *rest = block

                report.append(
                    f"BLOCK {index} | "
                    f"x={x0:.1f}–{x1:.1f}, "
                    f"y={y0:.1f}–{y1:.1f}\n"
                    f"{text[:1200]}\n"
                )

            # Inspect font information to identify script
            # and distinguish headings from ordinary text.
            report.append("\n--- FONT SAMPLES ---\n")

            font_samples = {}

            page_dict = page.get_text("dict")

            for block in page_dict.get("blocks", []):
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    for span in line["spans"]:
                        key = (
                            span["font"],
                            round(span["size"], 1)
                        )

                        if key not in font_samples:
                            font_samples[key] = span["text"][:150]

            for (font, size), sample in font_samples.items():
                report.append(
                    f"Font: {font} | Size: {size}\n"
                    f"Sample: {sample}\n"
                )

    OUTPUT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("Inspected pages:", PAGES_TO_INSPECT)
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()