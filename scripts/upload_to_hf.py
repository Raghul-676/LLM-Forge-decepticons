"""
Robust file-by-file uploader for RAG v2 datasets and embeddings to Hugging Face Hub.
Includes automatic retries, skip-if-already-uploaded, and progress tracking.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

HF_TOKEN = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
REPO_ID = "raghul12345678/decepticons-legal-rag-data"

MAX_RETRIES = 5
RETRY_DELAY = 10

def main():
    if not HF_TOKEN or HF_TOKEN.startswith("your_"):
        print("ERROR: HF_WRITE_TOKEN or HF_TOKEN is not set or invalid in .env", flush=True)
        return

    api = HfApi(token=HF_TOKEN)

    print(f"Checking dataset repository: {REPO_ID} ...", flush=True)
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
        print("Repository verified!\n", flush=True)
    except Exception as exc:
        print(f"ERROR accessing repository: {exc}", flush=True)
        return

    rag_dir = PROJECT_ROOT / "scripts" / "data" / "rag"
    if not rag_dir.exists():
        print(f"ERROR: Local RAG data directory not found at {rag_dir}", flush=True)
        return

    # Collect all files to upload
    all_files = sorted([p for p in rag_dir.rglob("*") if p.is_file()])
    total_files = len(all_files)
    total_bytes = sum(p.stat().st_size for p in all_files)

    print("========================================================", flush=True)
    print(f"Target Repository : https://huggingface.co/datasets/{REPO_ID}", flush=True)
    print(f"Total Files       : {total_files}", flush=True)
    print(f"Total Size        : {total_bytes / (1024 * 1024):.2f} MB ({total_bytes / (1024 * 1024 * 1024):.2f} GB)", flush=True)
    print("========================================================\n", flush=True)

    # Get already uploaded files to skip redundant uploads
    try:
        existing_files = set(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
    except Exception:
        existing_files = set()

    for idx, file_path in enumerate(all_files, start=1):
        rel_path = file_path.relative_to(rag_dir.parent) # e.g. rag/ilsic/...
        repo_path = rel_path.as_posix()
        size_mb = file_path.stat().st_size / (1024 * 1024)

        if repo_path in existing_files:
            print(f"[{idx}/{total_files}] SKIP (already uploaded): {repo_path} ({size_mb:.2f} MB)", flush=True)
            continue

        print(f"[{idx}/{total_files}] Uploading: {repo_path} ({size_mb:.2f} MB)...", flush=True)

        uploaded = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=repo_path,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    commit_message=f"Upload {repo_path} ({size_mb:.2f} MB)"
                )
                print(f"[{idx}/{total_files}] DONE: {repo_path}", flush=True)
                uploaded = True
                break
            except Exception as exc:
                print(f"[{idx}/{total_files}] Warning: Attempt {attempt}/{MAX_RETRIES} failed: {exc}", flush=True)
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_DELAY * attempt
                    print(f"Waiting {wait_time}s before retrying...", flush=True)
                    time.sleep(wait_time)

        if not uploaded:
            print(f"\nERROR: Failed to upload {repo_path} after {MAX_RETRIES} attempts.", flush=True)
            print("Please re-run this script to resume from where it left off.", flush=True)
            sys.exit(1)

    print("\n========================================================", flush=True)
    print("SUCCESS: All RAG dataset files and embeddings uploaded successfully!", flush=True)
    print(f"View dataset: https://huggingface.co/datasets/{REPO_ID}", flush=True)
    print("========================================================", flush=True)

if __name__ == "__main__":
    main()
