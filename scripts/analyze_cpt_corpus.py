from pathlib import Path
from collections import Counter
import json
import statistics

from transformers import AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

DATASET_FILE = Path(
    "data/training/cpt/legal_cpt_corpus.jsonl"
)

MODEL_NAME = "HuggingFaceTB/SmolLM3-3B-Base"

SUMMARY_FILE = Path(
    "data/training/cpt/legal_cpt_corpus_stats.json"
)


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                yield json.loads(line)

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {exc}"
                )


def count_words(text):
    """
    Simple whitespace-based word count.
    """

    return len(
        text.split()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CPT CORPUS ANALYSIS")
    print("=" * 70)

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{DATASET_FILE}"
        )

    print(
        "\nDataset:",
        DATASET_FILE
    )

    print(
        "Tokenizer:",
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )

    print(
        "Tokenizer loaded."
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_documents = 0
    total_words = 0
    total_tokens = 0
    total_characters = 0

    source_counts = Counter()

    words_per_document = []
    tokens_per_document = []

    largest_documents = []

    # --------------------------------------------------------
    # Process dataset
    # --------------------------------------------------------

    print(
        "\nAnalyzing corpus..."
    )

    for index, record in enumerate(
        load_jsonl(DATASET_FILE),
        start=1
    ):

        text = record.get(
            "text",
            ""
        )

        if not text:
            continue

        source_type = record.get(
            "source_type",
            "unknown"
        )

        document_id = record.get(
            "id",
            f"record_{index}"
        )

        # ----------------------------------------------------
        # Character count
        # ----------------------------------------------------

        character_count = len(text)

        # ----------------------------------------------------
        # Word count
        # ----------------------------------------------------

        word_count = count_words(
            text
        )

        # ----------------------------------------------------
        # Token count
        #
        # add_special_tokens=False because we want to measure
        # raw corpus token volume.
        # ----------------------------------------------------

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False
        )

        token_count = len(
            token_ids
        )

        # ----------------------------------------------------
        # Update totals
        # ----------------------------------------------------

        total_documents += 1

        total_characters += (
            character_count
        )

        total_words += (
            word_count
        )

        total_tokens += (
            token_count
        )

        source_counts[
            source_type
        ] += 1

        words_per_document.append(
            word_count
        )

        tokens_per_document.append(
            token_count
        )

        largest_documents.append({
            "id": document_id,
            "source_type": source_type,
            "words": word_count,
            "tokens": token_count,
            "characters": character_count
        })

        # ----------------------------------------------------
        # Progress output
        # ----------------------------------------------------

        if index % 1000 == 0:

            print(
                f"Processed "
                f"{index:,} records..."
            )

    # --------------------------------------------------------
    # Sort largest docs
    # --------------------------------------------------------

    largest_documents.sort(
        key=lambda x: x["tokens"],
        reverse=True
    )

    largest_documents = (
        largest_documents[:10]
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    average_words = (
        total_words / total_documents
        if total_documents
        else 0
    )

    average_tokens = (
        total_tokens / total_documents
        if total_documents
        else 0
    )

    average_characters = (
        total_characters / total_documents
        if total_documents
        else 0
    )

    median_words = (
        statistics.median(
            words_per_document
        )
        if words_per_document
        else 0
    )

    median_tokens = (
        statistics.median(
            tokens_per_document
        )
        if tokens_per_document
        else 0
    )

    min_words = (
        min(words_per_document)
        if words_per_document
        else 0
    )

    max_words = (
        max(words_per_document)
        if words_per_document
        else 0
    )

    min_tokens = (
        min(tokens_per_document)
        if tokens_per_document
        else 0
    )

    max_tokens = (
        max(tokens_per_document)
        if tokens_per_document
        else 0
    )

    tokens_per_word = (
        total_tokens / total_words
        if total_words
        else 0
    )

    # --------------------------------------------------------
    # Summary object
    # --------------------------------------------------------

    summary = {
        "dataset_file":
            str(DATASET_FILE),

        "tokenizer":
            MODEL_NAME,

        "total_documents":
            total_documents,

        "total_characters":
            total_characters,

        "total_words":
            total_words,

        "total_tokens":
            total_tokens,

        "tokens_per_word":
            tokens_per_word,

        "average_characters_per_document":
            average_characters,

        "average_words_per_document":
            average_words,

        "average_tokens_per_document":
            average_tokens,

        "median_words_per_document":
            median_words,

        "median_tokens_per_document":
            median_tokens,

        "minimum_words_in_document":
            min_words,

        "maximum_words_in_document":
            max_words,

        "minimum_tokens_in_document":
            min_tokens,

        "maximum_tokens_in_document":
            max_tokens,

        "source_counts":
            dict(source_counts),

        "largest_documents_by_tokens":
            largest_documents
    }

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CPT CORPUS STATISTICS"
    )

    print(
        "=" * 70
    )

    print(
        f"Documents: "
        f"{total_documents:,}"
    )

    print(
        f"Characters: "
        f"{total_characters:,}"
    )

    print(
        f"Words: "
        f"{total_words:,}"
    )

    print(
        f"Tokens: "
        f"{total_tokens:,}"
    )

    print(
        f"Tokens / word: "
        f"{tokens_per_word:.3f}"
    )

    print(
        "\nAverage per document:"
    )

    print(
        f"  Characters: "
        f"{average_characters:,.2f}"
    )

    print(
        f"  Words: "
        f"{average_words:,.2f}"
    )

    print(
        f"  Tokens: "
        f"{average_tokens:,.2f}"
    )

    print(
        "\nMedian per document:"
    )

    print(
        f"  Words: "
        f"{median_words:,.2f}"
    )

    print(
        f"  Tokens: "
        f"{median_tokens:,.2f}"
    )

    print(
        "\nDocument ranges:"
    )

    print(
        f"  Words: "
        f"{min_words:,} "
        f"to "
        f"{max_words:,}"
    )

    print(
        f"  Tokens: "
        f"{min_tokens:,} "
        f"to "
        f"{max_tokens:,}"
    )

    print(
        "\nDocuments by source:"
    )

    for source, count in (
        source_counts.items()
    ):

        print(
            f"  {source}: "
            f"{count:,}"
        )

    print(
        "\nLargest documents "
        "by token count:"
    )

    for rank, document in enumerate(
        largest_documents,
        start=1
    ):

        print(
            f"  {rank}. "
            f"{document['id']} | "
            f"{document['source_type']} | "
            f"{document['tokens']:,} tokens | "
            f"{document['words']:,} words"
        )

    print(
        "\nSaved statistics:"
    )

    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()