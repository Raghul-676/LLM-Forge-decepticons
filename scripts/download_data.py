"""
Download RAG v2 datasets and embeddings from Hugging Face Hub.
Places files directly into scripts/data/rag/ so the web assistant can run immediately.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

REPO_ID = "raghul12345678/decepticons-legal-rag-data"
TARGET_DIR = PROJECT_ROOT / "scripts" / "data" / "rag"

def main():
    print("========================================================")
    print("Downloading Decepticons Legal AI RAG Datasets & Embeddings")
    print(f"Source Repository: https://huggingface.co/datasets/{REPO_ID}")
    print(f"Destination: {TARGET_DIR}")
    print("========================================================\n")

    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    token_arg = hf_token if (hf_token and not hf_token.startswith("your_")) else None

    try:
        downloaded_path = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=["rag/*"],
            local_dir=str(TARGET_DIR.parent),
            token=token_arg
        )
        print("\nSUCCESS: All RAG dataset files and embeddings downloaded successfully!")
        print(f"Files available at: {TARGET_DIR}")
        print("\nYou can now launch the web assistant using:")
        print("  .\\run_ui.bat")
    except Exception as exc:
        print(f"\nERROR downloading data: {exc}")
        print("\nIf the dataset is private, please ensure your HF_TOKEN is configured in .env.")

if __name__ == "__main__":
    main()
