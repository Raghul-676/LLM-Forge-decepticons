from pathlib import Path
import json
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"

CORPUS_FILE = Path(
    "data/rag/v1/legal_documents.jsonl"
)

OUTPUT_FILE = Path(
    "data/rag/v1/legal_embeddings.npy"
)

SUMMARY_FILE = Path(
    "data/rag/v1/legal_embeddings_summary.json"
)

BATCH_SIZE = 64


def main():

    print("=" * 70)
    print("RAG PROTOTYPE V1 - BUILD EMBEDDINGS")
    print("=" * 70)

    if not CORPUS_FILE.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_FILE}"
        )

    # --------------------------------------------------------
    # Load document texts
    # --------------------------------------------------------

    print("\nLoading RAG corpus...")

    texts = []
    ids = []

    with open(
        CORPUS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            record = json.loads(line)

            text = record.get("text", "").strip()

            if not text:
                raise ValueError(
                    f"Empty text at corpus line {line_number}"
                )

            texts.append(text)
            ids.append(record["id"])

    print("Documents loaded:", len(texts))

    if not texts:
        raise ValueError("Corpus contains no documents.")

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate document IDs found."
        )

    # --------------------------------------------------------
    # Choose device
    # --------------------------------------------------------

    if torch.cuda.is_available():
        device = "cuda"
        print("Embedding device: CUDA")
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )
    else:
        device = "cpu"
        print("Embedding device: CPU")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading embedding model:")
    print(MODEL_NAME)

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    # BGE-small-en-v1.5 supports a 512-token input window.
    model.max_seq_length = 512

    print(
        "Model max sequence length:",
        model.max_seq_length
    )

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    print("\nCreating embeddings...")
    print("Batch size:", BATCH_SIZE)
    print(
        "This may take several minutes depending "
        "on your GPU/CPU."
    )

    start_time = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    elapsed = time.time() - start_time

    # Force a predictable storage type.
    embeddings = embeddings.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if embeddings.shape[0] != len(texts):
        raise ValueError(
            "Embedding count does not match document count."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )

    # Normalized embeddings should have norm ~1.
    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    mean_norm = float(
        np.mean(norms)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_FILE,
        embeddings
    )

    summary = {
        "embedding_model": MODEL_NAME,
        "document_count": len(texts),
        "embedding_count": int(
            embeddings.shape[0]
        ),
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "dtype": str(
            embeddings.dtype
        ),
        "normalized": True,
        "mean_vector_norm": mean_norm,
        "batch_size": BATCH_SIZE,
        "device": device,
        "elapsed_seconds": round(
            elapsed,
            2
        ),
        "corpus_file": str(
            CORPUS_FILE
        ),
        "embedding_file": str(
            OUTPUT_FILE
        )
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("EMBEDDING BUILD COMPLETE")
    print("=" * 70)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()