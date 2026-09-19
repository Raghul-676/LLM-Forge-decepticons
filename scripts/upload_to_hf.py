"""
Upload RAG v2 datasets and embeddings to Hugging Face Hub.
Requires a Hugging Face Access Token with WRITE permissions.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "raghul12345678/decepticons-legal-rag-data"

def main():
    if not HF_TOKEN or HF_TOKEN.startswith("your_"):
        print("ERROR: HF_TOKEN is not set or invalid in .env")
        print("Please add a Hugging Face token with WRITE permissions to your .env file.")
        return

    api = HfApi(token=HF_TOKEN)

    print(f"Creating/verifying Hugging Face dataset repo: {REPO_ID} ...")
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
        print("Repository verified!")
    except Exception as exc:
        print(f"ERROR: Could not create/access repository: {exc}")
        print("\nPlease ensure:")
        print("1. Your token at https://huggingface.co/settings/tokens has 'Write' permissions.")
        print(f"2. Or create the dataset manually at: https://huggingface.co/new-dataset with name 'decepticons-legal-rag-data'")
        return

    rag_dir = PROJECT_ROOT / "scripts" / "data" / "rag"
    if not rag_dir.exists():
        print(f"ERROR: Local RAG data directory not found at {rag_dir}")
        return

    print("\n========================================================")
    print("Uploading RAG datasets and embeddings to Hugging Face...")
    print(f"Source: {rag_dir}")
    print(f"Target: {REPO_ID}")
    print("========================================================\n")

    try:
        api.upload_folder(
            folder_path=str(rag_dir),
            path_in_repo="rag",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Upload RAG v2 legal corpus, ILSIC scenarios, and BGE embeddings"
        )
        print("\nSUCCESS: All RAG dataset files and embeddings uploaded successfully!")
        print(f"View dataset: https://huggingface.co/datasets/{REPO_ID}")
    except Exception as exc:
        print(f"\nERROR during upload: {exc}")

if __name__ == "__main__":
    main()
