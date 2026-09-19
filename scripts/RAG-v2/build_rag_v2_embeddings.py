import json
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

EMBEDDING_DIM = 384

# Safe starting point for RTX 3050 4 GB.
BATCH_SIZE = 64

# Save resume position periodically.
CHECKPOINT_EVERY = 5000

# BGE query prefix is NOT used here.
# This file contains passage/document embeddings.
#
# Query prefix will only be used later when embedding user queries.


# ============================================================
# FIND PROJECT ROOT
# ============================================================

def find_project_root():

    script_path = Path(__file__).resolve()

    for parent in script_path.parents:

        if (parent / "scripts").is_dir():
            return parent

    raise RuntimeError(
        "Could not locate project root."
    )


PROJECT_ROOT = find_project_root()


# ============================================================
# FIND RAG V2 DATA ROOT
# ============================================================

def find_data_root():

    candidates = [
        PROJECT_ROOT / "scripts" / "data",
        PROJECT_ROOT / "data",
    ]

    for root in candidates:

        corpus = (
            root
            / "rag"
            / "v2"
            / "legal_documents.jsonl"
        )

        if corpus.exists():
            return root

    checked = "\n".join(
        str(
            root
            / "rag"
            / "v2"
            / "legal_documents.jsonl"
        )
        for root in candidates
    )

    raise FileNotFoundError(
        "Could not find RAG v2 corpus.\n"
        "Checked:\n"
        + checked
    )


DATA_ROOT = find_data_root()


# ============================================================
# PATHS
# ============================================================

RAG_DIR = (
    DATA_ROOT
    / "rag"
    / "v2"
)

CORPUS_FILE = (
    RAG_DIR
    / "legal_documents.jsonl"
)

CORPUS_SUMMARY_FILE = (
    RAG_DIR
    / "legal_documents_summary.json"
)

EMBEDDINGS_FILE = (
    RAG_DIR
    / "legal_embeddings.npy"
)

PROGRESS_FILE = (
    RAG_DIR
    / "embedding_progress.json"
)

EMBEDDING_SUMMARY_FILE = (
    RAG_DIR
    / "legal_embeddings_summary.json"
)


# ============================================================
# LOAD NUMBER OF CHUNKS
# ============================================================

def get_total_documents():

    if CORPUS_SUMMARY_FILE.exists():

        with open(
            CORPUS_SUMMARY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            summary = json.load(f)

        total = int(
            summary["chunks_written"]
        )

        return total

    print(
        "Corpus summary not found."
    )

    print(
        "Counting JSONL records..."
    )

    total = 0

    with open(
        CORPUS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():
                total += 1

    return total


# ============================================================
# PROGRESS HELPERS
# ============================================================

def load_progress():

    if not PROGRESS_FILE.exists():
        return 0

    with open(
        PROGRESS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    completed = int(
        data.get(
            "completed_rows",
            0
        )
    )

    return completed


def save_progress(
    completed_rows,
    total_rows,
    batch_size
):

    progress = {
        "embedding_model":
            EMBEDDING_MODEL,

        "completed_rows":
            completed_rows,

        "total_rows":
            total_rows,

        "embedding_dimension":
            EMBEDDING_DIM,

        "batch_size":
            batch_size,

        "complete":
            completed_rows >= total_rows,
    }

    temporary_file = (
        PROGRESS_FILE.with_suffix(
            ".tmp"
        )
    )

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            progress,
            f,
            indent=2
        )

    temporary_file.replace(
        PROGRESS_FILE
    )


# ============================================================
# LOAD / CREATE MEMMAP
# ============================================================

def prepare_embedding_file(
    total_rows
):

    completed_rows = (
        load_progress()
    )

    # --------------------------------------------------------
    # Resume existing run
    # --------------------------------------------------------

    if (
        EMBEDDINGS_FILE.exists()
        and completed_rows > 0
    ):

        print(
            "\nExisting embedding run detected."
        )

        print(
            f"Completed rows: "
            f"{completed_rows:,}"
        )

        embeddings = open_memmap(
            EMBEDDINGS_FILE,
            mode="r+"
        )

        expected_shape = (
            total_rows,
            EMBEDDING_DIM
        )

        if embeddings.shape != expected_shape:

            raise RuntimeError(
                "Existing embedding file has "
                "incorrect shape.\n"
                f"Expected: {expected_shape}\n"
                f"Found: {embeddings.shape}\n\n"
                "Delete legal_embeddings.npy and "
                "embedding_progress.json before restarting."
            )

        return (
            embeddings,
            completed_rows
        )

    # --------------------------------------------------------
    # Avoid accidentally overwriting unexplained file
    # --------------------------------------------------------

    if (
        EMBEDDINGS_FILE.exists()
        and completed_rows == 0
    ):

        raise RuntimeError(
            "legal_embeddings.npy already exists, "
            "but there is no valid resume progress.\n\n"
            "If this is an incomplete old file, delete:\n"
            f"{EMBEDDINGS_FILE}\n"
            f"{PROGRESS_FILE}\n"
            "and run the script again."
        )

    # --------------------------------------------------------
    # New embedding matrix
    # --------------------------------------------------------

    print(
        "\nCreating new embedding matrix..."
    )

    embeddings = open_memmap(
        EMBEDDINGS_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(
            total_rows,
            EMBEDDING_DIM
        )
    )

    save_progress(
        completed_rows=0,
        total_rows=total_rows,
        batch_size=BATCH_SIZE
    )

    return (
        embeddings,
        0
    )


# ============================================================
# EMBED BATCH WITH OOM FALLBACK
# ============================================================

def encode_batch(
    model,
    texts,
    current_batch_size
):

    while True:

        try:

            embeddings = model.encode(
                texts,
                batch_size=current_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            embeddings = (
                np.asarray(
                    embeddings,
                    dtype=np.float32
                )
            )

            return (
                embeddings,
                current_batch_size
            )

        except RuntimeError as exc:

            message = str(exc).lower()

            if (
                "out of memory" not in message
                and "cuda" not in message
            ):
                raise

            if current_batch_size <= 4:
                raise

            new_batch_size = max(
                4,
                current_batch_size // 2
            )

            print(
                "\nCUDA memory issue."
            )

            print(
                f"Reducing batch size: "
                f"{current_batch_size} "
                f"-> {new_batch_size}"
            )

            current_batch_size = (
                new_batch_size
            )

            if torch.cuda.is_available():

                torch.cuda.empty_cache()


# ============================================================
# FORMAT TIME
# ============================================================

def format_duration(seconds):

    seconds = int(seconds)

    hours = (
        seconds // 3600
    )

    minutes = (
        (seconds % 3600)
        // 60
    )

    secs = (
        seconds % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "RAG V2 BGE EMBEDDING BUILDER"
    )

    print(
        "=" * 78
    )

    print(
        "\nProject root:",
        PROJECT_ROOT
    )

    print(
        "Data root:",
        DATA_ROOT
    )

    print(
        "\nCorpus:",
        CORPUS_FILE
    )

    print(
        "Output:",
        EMBEDDINGS_FILE
    )

    # --------------------------------------------------------
    # Total rows
    # --------------------------------------------------------

    total_rows = (
        get_total_documents()
    )

    print(
        f"\nTotal RAG chunks: "
        f"{total_rows:,}"
    )

    print(
        "Embedding dimension:",
        EMBEDDING_DIM
    )

    estimated_gib = (
        total_rows
        * EMBEDDING_DIM
        * 4
        / (
            1024 ** 3
        )
    )

    print(
        f"Expected embedding file: "
        f"{estimated_gib:.3f} GiB"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nLoading BGE model..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device
    )

    model.max_seq_length = 512

    # FP16 significantly reduces VRAM usage.
    if device == "cuda":

        try:

            model.half()

            print(
                "Model precision: FP16"
            )

        except Exception:

            print(
                "FP16 conversion unavailable; "
                "continuing with default precision."
            )

    # --------------------------------------------------------
    # Verify dimension before creating 1 GB file
    # --------------------------------------------------------

    print(
        "\nValidating embedding dimension..."
    )

    test_embedding = model.encode(
        ["Indian legal document"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    actual_dimension = (
        test_embedding.shape[1]
    )

    print(
        "Detected dimension:",
        actual_dimension
    )

    if actual_dimension != EMBEDDING_DIM:

        raise RuntimeError(
            f"Expected {EMBEDDING_DIM}-dimensional "
            f"embeddings but model returned "
            f"{actual_dimension}."
        )

    # --------------------------------------------------------
    # Create / resume matrix
    # --------------------------------------------------------

    (
        embedding_matrix,
        completed_rows
    ) = prepare_embedding_file(
        total_rows
    )

    if completed_rows >= total_rows:

        print(
            "\nEmbedding file is already complete."
        )

        return

    # --------------------------------------------------------
    # Embedding loop
    # --------------------------------------------------------

    current_batch_size = (
        BATCH_SIZE
    )

    row_index = (
        completed_rows
    )

    texts = []

    start_time = time.time()

    last_checkpoint = (
        completed_rows
    )

    print(
        "\n"
        + "-" * 78
    )

    if completed_rows:

        print(
            f"RESUMING FROM ROW "
            f"{completed_rows:,}"
        )

    else:

        print(
            "STARTING EMBEDDING GENERATION"
        )

    print(
        "-" * 78
    )

    with open(
        CORPUS_FILE,
        "r",
        encoding="utf-8"
    ) as corpus_handle:

        for line_number, line in enumerate(
            corpus_handle
        ):

            # Skip rows already safely completed
            # during an earlier run.
            if line_number < completed_rows:
                continue

            line = line.strip()

            if not line:
                continue

            record = json.loads(
                line
            )

            text = str(
                record.get(
                    "text",
                    ""
                )
            ).strip()

            if not text:

                raise RuntimeError(
                    f"Empty text encountered "
                    f"at RAG row "
                    f"{line_number:,}"
                )

            texts.append(
                text
            )

            # ------------------------------------------------
            # Encode one outer batch
            # ------------------------------------------------

            if len(texts) >= BATCH_SIZE:

                (
                    batch_embeddings,
                    current_batch_size
                ) = encode_batch(
                    model,
                    texts,
                    current_batch_size
                )

                batch_length = len(
                    batch_embeddings
                )

                end_index = (
                    row_index
                    + batch_length
                )

                embedding_matrix[
                    row_index:end_index
                ] = batch_embeddings

                row_index = (
                    end_index
                )

                texts = []

                # ------------------------------------------------
                # Checkpoint
                # ------------------------------------------------

                if (
                    row_index
                    - last_checkpoint
                    >= CHECKPOINT_EVERY
                ):

                    embedding_matrix.flush()

                    save_progress(
                        completed_rows=
                            row_index,

                        total_rows=
                            total_rows,

                        batch_size=
                            current_batch_size
                    )

                    last_checkpoint = (
                        row_index
                    )

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    session_rows = (
                        row_index
                        - completed_rows
                    )

                    speed = (
                        session_rows
                        / elapsed
                        if elapsed > 0
                        else 0
                    )

                    remaining = (
                        total_rows
                        - row_index
                    )

                    eta_seconds = (
                        remaining
                        / speed
                        if speed > 0
                        else 0
                    )

                    percentage = (
                        row_index
                        / total_rows
                        * 100
                    )

                    print(
                        f"Embedded "
                        f"{row_index:,}"
                        f"/"
                        f"{total_rows:,}"
                        f" | "
                        f"{percentage:6.2f}%"
                        f" | "
                        f"{speed:7.1f} chunks/s"
                        f" | ETA "
                        f"{format_duration(eta_seconds)}"
                        f" | batch="
                        f"{current_batch_size}"
                    )

        # ----------------------------------------------------
        # Final partial batch
        # ----------------------------------------------------

        if texts:

            (
                batch_embeddings,
                current_batch_size
            ) = encode_batch(
                model,
                texts,
                current_batch_size
            )

            batch_length = len(
                batch_embeddings
            )

            end_index = (
                row_index
                + batch_length
            )

            embedding_matrix[
                row_index:end_index
            ] = batch_embeddings

            row_index = (
                end_index
            )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    embedding_matrix.flush()

    if row_index != total_rows:

        raise RuntimeError(
            "Embedding row count mismatch.\n"
            f"Expected: {total_rows:,}\n"
            f"Written:  {row_index:,}"
        )

    save_progress(
        completed_rows=row_index,
        total_rows=total_rows,
        batch_size=current_batch_size
    )

    elapsed = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Norm sanity test
    # --------------------------------------------------------

    sample_size = min(
        5000,
        total_rows
    )

    sample_indexes = np.linspace(
        0,
        total_rows - 1,
        sample_size,
        dtype=int
    )

    sample = np.asarray(
        embedding_matrix[
            sample_indexes
        ],
        dtype=np.float32
    )

    norms = np.linalg.norm(
        sample,
        axis=1
    )

    mean_norm = float(
        norms.mean()
    )

    min_norm = float(
        norms.min()
    )

    max_norm = float(
        norms.max()
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "embedding_model":
            EMBEDDING_MODEL,

        "corpus_file":
            str(
                CORPUS_FILE
            ),

        "embedding_file":
            str(
                EMBEDDINGS_FILE
            ),

        "documents":
            total_rows,

        "embedding_dimension":
            EMBEDDING_DIM,

        "dtype":
            "float32",

        "normalized":
            True,

        "device":
            device,

        "final_batch_size":
            current_batch_size,

        "elapsed_seconds_this_run":
            round(
                elapsed,
                2
            ),

        "sample_norm_validation": {
            "sample_size":
                sample_size,

            "mean_norm":
                mean_norm,

            "min_norm":
                min_norm,

            "max_norm":
                max_norm,
        },

        "file_size_gib":
            round(
                EMBEDDINGS_FILE.stat().st_size
                / (
                    1024 ** 3
                ),
                3
            ),
    }

    with open(
        EMBEDDING_SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "RAG V2 EMBEDDINGS COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"\nEmbeddings: "
        f"{total_rows:,}"
    )

    print(
        f"Dimension: "
        f"{EMBEDDING_DIM}"
    )

    print(
        "dtype: float32"
    )

    print(
        "Normalized: yes"
    )

    print(
        f"\nMean sampled norm: "
        f"{mean_norm:.6f}"
    )

    print(
        f"Minimum sampled norm: "
        f"{min_norm:.6f}"
    )

    print(
        f"Maximum sampled norm: "
        f"{max_norm:.6f}"
    )

    print(
        f"\nElapsed this run: "
        f"{format_duration(elapsed)}"
    )

    print(
        f"Embedding file size: "
        f"{summary['file_size_gib']:.3f} GiB"
    )

    print(
        "\nOUTPUT FILES"
    )

    print(
        "-" * 78
    )

    print(
        "Embeddings:",
        EMBEDDINGS_FILE
    )

    print(
        "Summary:",
        EMBEDDING_SUMMARY_FILE
    )

    print(
        "Progress:",
        PROGRESS_FILE
    )


if __name__ == "__main__":
    main()