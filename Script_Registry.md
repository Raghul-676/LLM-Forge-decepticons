# Legal LLM — Script Registry

**Project:** Multi-Agent Legal Reasoning and Litigation Strategy Assistant
**Team:** Decepticons
**Project root:** `D:\Projects\LLM_training`
**Base model:** `HuggingFaceTB/SmolLM3-3B-Base`

This file records the scripts created during development. Update it whenever a new script is added or an existing script changes significantly.

## Current progress

* Base model downloaded and successfully tested using 4-bit inference.
* Constitution, Central Acts, and Supreme Court datasets collected.
* Supreme Court format-aware parser tested on 200 PDFs.
* Central Acts structurally normalized into 35,864 section records.
* Constitution authoritative-source preparation is the current task.
* **Actual continued pretraining and SFT have not started yet.**

## 1. Environment and baseline

| Script                       | Purpose                                                                                                  | Main output                      |
| ---------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `scripts/check_gpu.py`       | Checks whether PyTorch detects the NVIDIA GPU and CUDA environment.                                      | Terminal GPU/CUDA information    |
| `scripts/test_base_model.py` | Loads the untouched SmolLM3-3B-Base model in 4-bit mode and runs baseline prompts before legal training. | `outputs/base_model_results.txt` |

## 2. Initial dataset inspection and preprocessing

| Script                                | Purpose                                                                                                            | Main output                                                   |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| `scripts/inspect_datasets.py`         | Inspects raw dataset folders, CSV schemas, JSON structure, file counts, and sample PDF text.                       | Terminal inspection report                                    |
| `scripts/test_preprocessing.py`       | Tests basic cleaning and JSONL conversion on the Constitution, five Central Acts, and ten Supreme Court PDFs.      | `constitution_test.jsonl`, `acts_test.jsonl`, `sc_test.jsonl` |
| `scripts/test_sc_metadata_linking.py` | Tests whether legacy Supreme Court PDFs can be matched to the judgments CSV using dates and party-name similarity. | `outputs/sc_metadata_link_test.csv`                           |

**Note:** The metadata-linking test produced mostly low-confidence matches. The CSV must not be attached to PDFs through unverified fuzzy matches.

## 3. Supreme Court parsing and quality control

| Script                               | Purpose                                                                                                                   | Main output                                              |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `scripts/parse_sc_judgments_test.py` | Parses legacy JUDIS PDFs into structured records containing parties, date, bench, citations, headnote, and judgment body. | `sc_structured_test.jsonl`, `sc_parse_quality_test.csv`  |
| `scripts/review_sc_parsing.py`       | Generates a manual review report to compare extracted fields against original PDF text.                                   | `sc_manual_review.txt`, `sc_manual_review_checklist.csv` |
| `scripts/run_sc_pilot.py`            | Runs the original parser on a fixed random sample of 200 PDFs and records extraction quality.                             | `sc_pilot_200.jsonl`, quality CSV, summary JSON          |
| `scripts/inspect_sc_failures.py`     | Displays the first two pages of unsupported PDFs and lists extraction errors to identify new document layouts.            | `outputs/sc_failure_inspection.txt`                      |
| `scripts/parse_sc_format_aware.py`   | Detects legacy or modern Supreme Court layouts and routes each PDF to the appropriate parser.                             | Structured judgment record; standalone smoke-test output |
| `scripts/run_sc_pilot_v2.py`         | Reruns the format-aware parser on the exact same 200 PDFs as the original pilot.                                          | `sc_pilot_200_v2.jsonl`, quality CSV, summary JSON       |

### Supreme Court pilot results

| Metric                     |                Original |           V2 |
| -------------------------- | ----------------------: | -----------: |
| PDFs attempted             |                     200 |          200 |
| Documents saved            |                     194 |          194 |
| Judgment bodies identified |                      99 |          191 |
| Text marked usable         | Not separately measured |          190 |
| Legacy format              |                       — |           99 |
| Modern format              |             Unsupported |           95 |
| Empty PDFs                 |                6 errors | 6 classified |

**Important:** `usable` is a preliminary automated text-quality label, not a guarantee of complete or legally verified extraction. The four text exceptions and remaining metadata gaps still require review. The six zero-byte PDFs are preserved in raw data but excluded from training.

## 4. Constitution and Central Acts

| Script                                | Purpose                                                                                                                                    | Main output                                                                             |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| `scripts/audit_statutory_datasets.py` | Audits Constitution CSV rows and the top-level structure of all 858 Central Acts JSON files.                                               | `statutory_audit_summary.json`, `central_acts_inventory.csv`                            |
| `scripts/audit_act_hierarchy.py`      | Recursively discovers sections inside Chapters, Parts, and other nested structures to avoid missing provisions.                            | `outputs/central_acts_hierarchy_audit.json`                                             |
| `scripts/normalize_central_acts.py`   | Converts every discovered section into canonical JSONL while preserving source paths, hierarchy, raw payloads, and supplementary material. | `central_acts_sections.jsonl`, `central_acts_supplementary.jsonl`, normalization report |
| `scripts/validate_act_samples.py`     | Compares normalized sections against original JSON sources and verifies headings, paragraphs, and hierarchy.                               | `outputs/central_acts_sample_validation.txt`                                            |

### Central Acts results

* 858 Acts loaded and normalized successfully.
* 35,864 sections discovered recursively.
* All 425 Acts with nested sections are included.
* Zero processing errors and zero reported normalization issues.
* Sample validation passed for the Fatal Accidents Act and Indian Trusts Act.
* The source is a historical snapshot; current legal status and amendments remain unverified.

### Constitution status

The existing CSV contains 456 rows, but rows do not consistently correspond to complete Articles. The final Constitution corpus will be based on a verified consolidated source, preserving Articles, Parts, Schedules, and amendment information.

## 5. Important dataset locations

| Path                                          | Contents                                                                      |
| --------------------------------------------- | ----------------------------------------------------------------------------- |
| `data/raw/constitution/`                      | Original Constitution CSV, Index CSV, and authoritative source PDF when added |
| `data/raw/central_acts/annotatedCentralActs/` | Original 858 Acts JSON files                                                  |
| `data/raw/supreme_court/pdfs/`                | Original Supreme Court PDF collection                                         |
| `data/processed/constitution/`                | Constitution test and future normalized records                               |
| `data/processed/central_acts/`                | Normalized sections and supplementary data                                    |
| `data/processed/supreme_court/`               | Parsed judgment tests and pilots                                              |
| `outputs/`                                    | Baseline results, audit reports, quality CSVs, and summaries                  |
| `scripts/`                                    | All project scripts                                                           |

## 6. Training formats we are working toward

**Canonical processed documents:** Rich JSONL records preserving text, source identifiers, hierarchy, metadata, and quality flags.

**Continued pretraining (CPT):** Clean legal text in a `text` field, tokenized using the existing SmolLM3 tokenizer.

**Supervised fine-tuning (SFT):** Verified conversational examples using `messages` with user and assistant roles.

**RAG:** Searchable legal chunks with accurate citations, source provenance, and legal-version metadata.

**Evaluation:** Held-out examples kept separate from training to measure genuine improvement and avoid leakage.

## 7. Development rules

* Never modify or delete the original raw datasets.
* Preserve source filenames, hashes, and source paths wherever possible.
* Do not invent missing legal metadata.
* Keep historical law separate from verified current law.
* Test new parsers on a small sample before processing the full corpus.
* Preserve failed and review-required records for inspection rather than silently discarding them.
* Do not confuse development files named `*_test.jsonl` with a true held-out machine-learning test set.
* Update this registry whenever a script is created, renamed, or substantially changed.

## 8. New script entry template

Copy this template whenever we add a script:

### Script name: `scripts/<name>.py`

**Created:** YYYY-MM-DD
**Stage:** Dataset preparation / CPT / SFT / RAG / Evaluation
**Purpose:** What problem does this script solve?
**Inputs:** Files or datasets it reads.
**Outputs:** Files or reports it creates.
**Dependencies:** Required Python packages or other scripts.
**Run command:** `python scripts/<name>.py`
**Status:** Planned / Testing / Validated / Needs revision
**Notes:** Important assumptions, known limitations, or results.

---

*Last updated: 8 September 2026. The next task is preparing the authoritative Constitution source. Future scripts will be appended to this registry.*
