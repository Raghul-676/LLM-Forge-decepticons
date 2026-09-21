# Decepticons Legal AI (LLM-Forge)
## Comprehensive Project Context, System Architecture & Implementation Manual

**Project Name:** Decepticons Legal AI — Indian Legal Reasoning & Litigation Strategy Assistant  
**Team:** Decepticons  
**Project Root:** `D:\Projects\LLM_training`  
**GitHub Repository:** [https://github.com/Raghul-676/LLM-Forge-decepticons](https://github.com/Raghul-676/LLM-Forge-decepticons)  
**Hugging Face Dataset:** [https://huggingface.co/datasets/raghul12345678/decepticons-legal-rag-data](https://huggingface.co/datasets/raghul12345678/decepticons-legal-rag-data)  
**Primary Base Architecture:** Retrieval-Augmented Generation (RAG v2) + Multi-Turn Conversational Clarification + Statutory Grounding Audit  
**Target Domain:** Indian Law (Constitution of India, Central Acts, and Supreme Court Jurisprudence)

---

## 📑 Table of Contents

1. [Executive Summary & Project Objectives](#1-executive-summary--project-objectives)
2. [The Core Problem in Legal AI](#2-the-core-problem-in-legal-ai)
3. [Datasets & Legal Knowledge Repositories](#3-datasets--legal-knowledge-repositories)
4. [End-to-End System Architecture (RAG v2)](#4-end-to-end-system-architecture-rag-v2)
   - [The 8-Stage Pipeline Breakdown](#the-8-stage-pipeline-breakdown)
5. [Technical Implementation & Core Innovations](#5-technical-implementation--core-innovations)
   - [O(1) Byte-Offset Indexing](#o1-byte-offset-indexing)
   - [Multi-Model Fallback Chain for High Availability](#multi-model-fallback-chain-for-high-availability)
   - [Two-Tier Fail-Closed Grounding Audit](#two-tier-fail-closed-grounding-audit)
   - [Strict Client-Facing Interface Policy](#strict-client-facing-interface-policy)
6. [Web Application Architecture (UI & Server)](#6-web-application-architecture-ui--server)
7. [Installation, Setup & Operation Guide](#7-installation-setup--operation-guide)
8. [Script & Directory Registry](#8-script--directory-registry)
9. [Future Roadmap & Continued Research](#9-future-roadmap--continued-research)

---

## 1. Executive Summary & Project Objectives

**Decepticons Legal AI** is a specialized, authority-grounded artificial intelligence platform designed to deliver precise, citation-verified legal opinions and strategic litigation insights for complex disputes under **Indian Law**.

The project addresses the fundamental limitation of contemporary Large Language Models: **legal hallucinations, fabricated statutory provisions, and outdated legal citations**. Rather than relying on ungrounded generative memory, Decepticons Legal AI couples high-speed neural retrieval (`BAAI/bge-small-en-v1.5`) with multi-stage reasoning models (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`) via an 8-stage verification pipeline.

### Core Objectives:
1. **Dynamic Fact Clarification**: Proactively detect missing material facts (such as limitation periods, written agreements, legal notices, or jurisdictional nuances) and interrogate the user across iterative turns before rendering an opinion.
2. **Precedent Analogy via ILSIC**: Compare the user's scenario against thousands of factual legal precedents from the Indian Legal Scenario (ILSIC) dataset.
3. **Statutory Authority Verification**: Validate potential legal remedies against 858 Central Acts and Constitutional provisions.
4. **Zero-Hallucination Grounding Audit**: Enforce a strict mathematical and logical audit where every factual assertion in the final consultation must trace directly to an authentic retrieved excerpt (`[S1]`, `[S2]`).
5. **Client-Facing Presentation**: Produce clean, authoritative, non-technical legal advice formatted specifically for non-lawyers, completely stripping away internal pipeline traces, intermediate scores, or debug logs.

---

## 2. The Core Problem in Legal AI

Standard generic LLMs (e.g., standard ChatGPT, LLaMA, or Claude) fail critically when applied to Indian law due to several domain-specific hurdles:

1. **Hallucinated Statutes & Non-Existent Sections**:
   Generic models frequently invent sections (e.g., citing Section 138 of the Indian Contract Act instead of the Negotiable Instruments Act, or inventing non-existent subsections).
2. **Conflation of Civil and Criminal Remedies**:
   In disputes like loan defaults or breach of contract, models often advise criminal complaints (Section 420 IPC / Section 318 BNS) without analyzing whether dishonest intention existed at the inception of the contract.
3. **Absence of Limitation & Procedural Awareness**:
   Remedies in Indian courts depend strictly on statutory limitation periods (Limitation Act, 1963) and mandatory procedural pre-conditions (e.g., 30-day statutory notices under Section 138 of the NI Act). Generic models give advice without verifying these prerequisite facts.
4. **Static Training Cutoffs & Complex Hierarchy**:
   Indian statutory law includes central legislation, state amendments, rules, regulations, and binding Supreme Court precedents that modify the interpretation of statutory text.

**Our Solution:** A **Retrieve-Verify-Audit** paradigm where the LLM is never permitted to generate legal claims from internal weights alone; every claim must be validated against retrieved text passages, audited by an independent adversarial verification prompt, and cited directly to source records.

---

## 3. Datasets & Legal Knowledge Repositories

The project indexes and utilizes three primary knowledge layers totaling over **14 GB of raw and processed legal data**:

### A. ILSIC (Indian Legal Scenario Index Corpus)
- **Files:** `ilsic_train_scenarios.jsonl`, `ilsic_train_chunks.jsonl`, `ilsic_train_embeddings.npy`
- **Scale:** 6,461 curated real-world Indian legal dispute scenarios, split into 10,165 semantic passages.
- **Vectors:** 10,165 dense embeddings (384 dimensions) generated with `BAAI/bge-small-en-v1.5`.
- **Function:** Serves as the first-stage case-analogy engine. When a user presents a factual dispute, the system retrieves scenarios with matching factual patterns to identify applicable statutes and common legal defenses.

### B. The Central Acts Statutory Corpus
- **Files:** `legal_documents.jsonl` (1.76 GB), `legal_authority_catalog.json` (3 MB), `legal_documents_offsets.npy` (5.4 MB), `legal_embeddings.npy` (1.04 GB).
- **Scale:** 858 Central Acts normalized into 35,864 section records and over 710,000 indexed retrieval chunks.
- **Hierarchy:** Preserves the structural hierarchy: `Act -> Part -> Chapter -> Section -> Subsection -> Proviso/Explanation`.
- **Authority Catalog:** Structured JSON mapping Act titles, short titles, enacted years, and normalized section numbers for exact string and fuzzy authority targeting.

### C. The Constitution of India
- Normalized text of the Constitution of India, preserving Articles, Parts, Schedules, and constitutional amendments.
- Fully integrated into the RAG v2 index to resolve fundamental rights violations (Articles 14, 19, 21, 32, 226).

### D. Supreme Court Jurisprudence Corpus
- **Files:** `sc_all.jsonl` (3.13 GB), parsed from historical and modern Supreme Court PDF judgments.
- **Format-Aware Parser:** Distinguishes between legacy JUDIS formatting (headnotes, bench compositions, party headers) and modern digital Supreme Court formats.
- **Use Case:** Used for Continual Pre-Training (CPT) and binding precedent citation retrieval.

### E. Cloud Hosting on Hugging Face Hub
Due to GitHub's strict 100 MB per-file limit, the entire runtime dataset (~2.77 GB) is hosted on Hugging Face Datasets:
- **Repository:** [`raghul12345678/decepticons-legal-rag-data`](https://huggingface.co/datasets/raghul12345678/decepticons-legal-rag-data)
- Automatically synchronized and downloaded via [`scripts/download_data.py`](file:///d:/Projects/LLM_training/scripts/download_data.py).

---

## 4. End-to-End System Architecture (RAG v2)

The core reasoning engine is implemented in [`scripts/RAG-v2/conversational_legal_rag.py`](file:///d:/Projects/LLM_training/scripts/RAG-v2/conversational_legal_rag.py) and [`scripts/RAG-v2/conversational_ilsic_mapper.py`](file:///d:/Projects/LLM_training/scripts/RAG-v2/conversational_ilsic_mapper.py). It executes an 8-stage verification workflow:

```
                  ┌─────────────────────────────────────┐
                  │          USER QUERY INPUT           │
                  │   ("I lent money to my friend...")  │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
         ┌────────────────────────────────────────────────────────┐
         │     STAGE 1: CONVERSATIONAL INTAKE & NORMALIZATION     │
         │  - Identifies legal essence & material fact gaps       │
         │  - State: ASK (Clarification) vs READY                 │
         └───────────────────────────┬────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │ Need Clarification?                   │
                 │ [YES] ──► Prompt User in Web UI       │
                 │           (Wait for Answer, Loop)     │
                 │ [NO]  ──► Proceed to Research         │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼
         ┌────────────────────────────────────────────────────────┐
         │     STAGE 2: SCENARIO RETRIEVAL (ILSIC CORPUS)         │
         │  - Generate multi-angle search representations         │
         │  - Dense retrieval over 10,165 scenario embeddings     │
         └───────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
         ┌────────────────────────────────────────────────────────┐
         │     STAGE 3: FACTUAL ANALOGY RERANKING                 │
         │  - Cross-encoder comparison of user dispute vs cases   │
         │  - LLM evaluates factual alignment & legal issues     │
         │  - Filters accepted analogous scenarios                │
         └───────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
         ┌────────────────────────────────────────────────────────┐
         │     STAGE 4: CANDIDATE STATUTORY VERIFICATION          │
         │  - Aggregate candidate Acts/Sections from precedents   │
         │  - Verify applicability against user's specific facts  │
         │  - Filter kept authorities                             │
         └───────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
         ┌────────────────────────────────────────────────────────┐
         │     STAGE 5: LEGAL-HYPOTHESIS GENERATION               │
         │  - Formulate structured research hypotheses            │
         │  - Target specific Acts & Sections in Authority Catalog│
         └───────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
         ┌────────────────────────────────────────────────────────┐
         │     STAGE 6: DENSE RETRIEVAL OVER LEGAL CORPUS         │
         │  - Dense vector search over 710,029 document offsets   │
         │  - Reciprocal Rank Fusion (RRF) across query variants  │
         │  - Exact Section & Heading Boost Scoring               │
         └───────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
         ┌────────────────────────────────────────────────────────┐
         │     STAGE 7: PASSAGE APPLICABILITY VERIFICATION        │
         │  - LLM checks each candidate excerpt for direct truth  │
         │  - Rejects passages that mention laws only in passing  │
         │  - Produces verified, passage-scoped propositions      │
         └───────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
         ┌────────────────────────────────────────────────────────┐
         │     STAGE 8: TWO-TIER GROUNDING AUDIT & SYNTHESIS      │
         │  - Tier 1: Draft answer grounded in verified passages  │
         │  - Adversarial Claim Auditor tests each sentence       │
         │  - Tier 2: Second grounding audit on legal claims      │
         │  - Fail-closed fallback if any claim ungrounded        │
         └───────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │    FINISHED CLIENT CONSULTATION     │
                  │  - Summary of Situation             │
                  │  - Verified Legal Indications       │
                  │  - Actionable Next Steps            │
                  │  - Crucial Uncertainties            │
                  │  - Authentic Citations ([S1], [S2]) │
                  └─────────────────────────────────────┘
```

---

### The 8-Stage Pipeline Breakdown

#### Stage 1: Conversational Fact Intake & Normalization
- **Purpose:** Analyze whether the dispute description contains sufficient information to render a sound legal assessment.
- **Logic:** Calls `mapper.analyze_intake()`. The model evaluates:
  - Parties involved (creditor/debtor, landlord/tenant, employer/employee).
  - Transaction documentation (written agreement, promissory note, oral, WhatsApp chats).
  - Dates and timeline (to determine whether the 3-year limitation period under the Limitation Act, 1963 has lapsed).
  - Demands and refusals (legal notice issued vs informal demand).
- **Decision:** If crucial facts are missing, the status returns `ASK` with a targeted, polite question (e.g., *"Did you execute any written agreement or promissory note when lending the money?"*). The system pauses, displays the question in the web UI, and waits for the user's reply before continuing.

#### Stage 2: Scenario Retrieval (ILSIC Corpus)
- **Purpose:** Map the normalized factual scenario to real-world dispute precedents.
- **Logic:** Multi-query expansion converts the scenario into legal issue statements and query representations. It computes cosine similarities against 10,165 pre-computed scenario chunk embeddings using `BAAI/bge-small-en-v1.5`.

#### Stage 3: Factual Analogy Reranking
- **Purpose:** Prevent false analogies.
- **Logic:** An LLM cross-encoder evaluates the top retrieved scenarios against the user's specific facts. It assesses:
  - *Are the core factual disputes analogous?*
  - *Is the legal remedy in this precedent applicable to the user's position?*
  - Rejects scenarios with differing factual predicates (e.g., commercial contracts vs friendly loans).

#### Stage 4: Candidate Statutory Provision Verification
- **Purpose:** Extract the statutory provisions associated with the accepted precedents and verify their preliminary relevance.
- **Logic:** Aggregates sections (e.g., Negotiable Instruments Act Section 138, Indian Contract Act Sections 124/126, Specific Relief Act Section 10) and checks them against the user's fact pattern.

#### Stage 5: Legal-Hypothesis Generation & Authority Targeting
- **Purpose:** Formulate formal legal hypotheses and target concrete statutory instruments in the Authority Catalog.
- **Logic:** Cross-references candidate provisions with `legal_authority_catalog.json` (which contains 858 Central Acts). It identifies exact Act IDs, section numbers, and schedule references.

#### Stage 6: Dense Retrieval over Legal Corpus (710k Offsets)
- **Purpose:** Retrieve the authoritative statutory and judicial text excerpts from `legal_documents.jsonl`.
- **Scoring Algorithm:**
  $$\text{Final Score} = \text{RRF}(\text{Rank}) + \text{Bonus}_{\text{exact\_section}} + \text{Bonus}_{\text{heading}} + \text{Bonus}_{\text{act\_match}}$$
  - **Reciprocal Rank Fusion (RRF)**: Combines dense vector rankings across multiple query formulations ($k = 60$).
  - **Exact Section Bonus (+0.070)**: Granted when an excerpt directly matches the target section number.
  - **Heading Bonus (+0.055)**: Granted when section titles match the dispute's legal topic.

#### Stage 7: Passage Applicability Verification & Source Reranking
- **Purpose:** Passage-scoped verification to prevent statutory misinterpretation.
- **Logic:** The LLM inspects each candidate excerpt and answers:
  - *Does this passage directly establish the legal proposition, or does it merely mention the Act in passing?*
  - Only passages scoring above threshold with direct legal reasons are retained as **Selected Sources (`[S1]`, `[S2]`, ...)**.

#### Stage 8: Two-Tier Grounding Audit & Synthesis
- **Purpose:** Guarantee 100% factual and statutory truthfulness in the client-facing response.
- **Tier 1 (Draft & Claim Extraction):** A draft answer is written citing selected sources. The Adversarial Claim Auditor extracts every individual proposition and cross-checks it against the source text.
- **Tier 2 (Fail-Closed Enforcement):** If the final text contains assertions that cannot be verified from the retrieved sources, the system drops the unverified text and falls back to safe, verified claim bullets. If no sources passed verification, it issues a clearly stated uncertainty notice rather than fabricating an answer.

---

## 5. Technical Implementation & Core Innovations

### O(1) Byte-Offset Indexing
One of the primary engineering breakthroughs in this repository is the ability to search and read a **1.76 GB JSONL file (`legal_documents.jsonl`)** on consumer hardware with minimal RAM usage:
- **`legal_documents_offsets.npy`**: A pre-computed NumPy array containing 710,029 64-bit integer byte offsets.
- **Mechanism:** During retrieval, when candidate chunk indices are selected (e.g., chunk #451,200), Python executes:
  ```python
  with open(LEGAL_DOCS_FILE, "rb") as f:
      f.seek(offsets[chunk_idx])
      record = json.loads(f.readline().decode("utf-8"))
  ```
- **Performance:** Instantaneous $O(1)$ random seek lookups without ever loading the 1.76 GB file into system memory.

### Multi-Model Fallback Chain for High Availability
To prevent pipeline failures caused by API rate limits (such as Groq's 200,000 Tokens-Per-Day limit on the 120B model), [`conversational_ilsic_mapper.py`](file:///d:/Projects/LLM_training/scripts/RAG-v2/conversational_ilsic_mapper.py) implements an automated multi-model failover chain:

```python
FALLBACK_MODELS = [
    "openai/gpt-oss-120b",   # Primary: 120B parameter deep reasoning model
    "openai/gpt-oss-20b",    # Secondary: 20B fast fallback model
    "qwen/qwen3.8-27b"       # Tertiary: High-throughput alternative
]
```

- If `openai/gpt-oss-120b` hits a `429 RateLimitError` (TPD or TPM), `groq_text()` instantly switches to `openai/gpt-oss-20b` without terminating the user's session.
- Exponential backoff and retry mechanisms handle transient socket drops.

### Two-Tier Fail-Closed Grounding Audit
Most commercial RAG systems fail *open*—meaning that if retrieval is ambiguous, the model guesses an answer. Decepticons Legal AI is engineered to **fail closed**:
- If an assertion in the draft response cannot be strictly mapped to a verified statutory passage in `selected_sources`, the claim is purged.
- If zero passages pass the strict relevance threshold, the system displays an honest **"What the verified legal sources indicate: The available verified legal sources are not sufficient to give a reliable answer for this scenario."** with guidance to consult an advocate, rather than guessing.

### Strict Client-Facing Interface Policy
The web UI is designed for non-lawyers and clients. Internal pipeline traces are strictly suppressed:
- ❌ No raw similarity scores, vector distances, or RRF calculations.
- ❌ No candidate scenarios or rejected scenario logs.
- ❌ No JSON dumps or internal prompt tokens.
- ✅ Clear, structured sections:
  1. **Your Situation** (concise factual summary)
  2. **What the Verified Legal Sources Indicate** (actionable legal rights and duties)
  3. **What You Can Do Next** (practical steps: legal notice, evidence preservation, limitation periods)
  4. **Important Uncertainty** (statutory caveats, jurisdiction, missing documents)
  5. **Verified Sources** (clean badges linking to Central Acts and Supreme Court precedents)

---

## 6. Web Application Architecture (UI & Server)

The application features a lightweight, high-performance web architecture:

### Backend: [`ui/server.py`](file:///d:/Projects/LLM_training/ui/server.py)
- **Framework:** Python `http.server.HTTPServer` with `ThreadingMixIn` for asynchronous concurrent requests.
- **Engine Singleton:** Loads models and indexes once into memory:
  - BGE Embedding Model (`BAAI/bge-small-en-v1.5`)
  - ILSIC Scenarios (6,461 items) and Chunks (10,165 items)
  - ILSIC Embeddings (10,165 × 384 matrix)
  - Legal Document Offsets (710,029 entries)
  - Authority Catalog (858 Acts)
- **Session Management (`LegalSession`):**
  - Runs the 8-stage pipeline in an isolated background thread.
  - Implements `threading.Event` to pause execution when a clarification question is needed, and resumes immediately upon receiving the user's answer.
  - Exposes REST API endpoints:
    - `POST /api/analyze` — Initiates consultation.
    - `GET /api/session/poll?id=<session_id>` — Streams session state, events, and results.
    - `POST /api/session/answer` — Submits user clarification answers.

### Frontend: [`ui/index.html`](file:///d:/Projects/LLM_training/ui/index.html), [`ui/index.css`](file:///d:/Projects/LLM_training/ui/index.css), [`ui/app.js`](file:///d:/Projects/LLM_training/ui/app.js)
- **Aesthetic:** Dark-mode design with glassmorphism, tailored typography (Outfit & Inter), responsive layouts, and micro-animations.
- **Clarification Cards:** Follow-up questions render as interactive inquiry cards directly in the consultation stream. Submitted answers transition smoothly into permanent *"Factual Detail Provided"* history badges.
- **Citation Badges:** Statutory authorities render as interactive `[S1]`, `[S2]` badges with expandable cards showing the exact statutory text excerpt.

---

## 7. Installation, Setup & Operation Guide

### Prerequisites
- **OS:** Windows 10/11, Linux, or macOS
- **Python:** Version 3.10 or higher
- **Hardware:** 8 GB+ RAM recommended. (GPU optional; BGE embeddings run efficiently on CPU via PyTorch).

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/Raghul-676/LLM-Forge-decepticons.git
cd LLM-Forge-decepticons
```

---

### Step 2: Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv legal-llm-venv
.\legal-llm-venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv legal-llm-venv
.\legal-llm-venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv legal-llm-venv
source legal-llm-venv/bin/activate
```

---

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and fill in your keys:
```env
# Hugging Face Access Token (optional for downloading public dataset)
HF_TOKEN=your_huggingface_token_here

# Groq API Key (required for LLM reasoning)
GROQ_API_KEY=gsk_your_groq_api_key_here
```

---

### Step 5: Download Required Datasets & Embeddings
Download the complete RAG v2 dataset (~2.77 GB) from Hugging Face:
```bash
python scripts/download_data.py
```
*This downloads the ILSIC scenario index, Central Acts corpus, 710k offsets, and BGE vector embeddings into `scripts/data/rag/`.*

---

### Step 6: Launch the Web Chatbot

**Windows (One-Click):**
Double-click `run_ui.bat` or run:
```cmd
.\run_ui.bat
```

**Manual Command:**
```bash
python ui/server.py 8000
```

Wait ~15–20 seconds for the legal index and embeddings to load. Once ready, the terminal will display:
```text
======================================================================
DECEPTICONS LEGAL AI WEB SERVER
URL: http://127.0.0.1:8000
======================================================================
```

Open your browser and navigate to:  
👉 **`http://localhost:8000`**

---

## 8. Script & Directory Registry

| Directory / File | Description & Purpose |
| :--- | :--- |
| **`README.md`** | Public GitHub repository documentation and quick-start guide. |
| **`PROJECT_CONTEXT.md`** | This document: Comprehensive architecture, data specifications, and implementation details. |
| **`requirements.txt`** | Python dependencies (PyTorch, Transformers, Groq, PyMuPDF, Scikit-learn, etc.). |
| **`.env.example`** | Template environment file for API keys. |
| **`run_ui.bat`** | Windows startup script to launch the server on port 8000. |
| **`ui/server.py`** | Multi-threaded HTTP backend server and legal session manager. |
| **`ui/index.html`** | Single-page legal consultation web application. |
| **`ui/index.css`** | Premium dark-mode styling, tokens, and responsive layout. |
| **`ui/app.js`** | Frontend event loop, multi-turn clarification handling, and citation badge rendering. |
| **`scripts/download_data.py`** | One-click downloader for datasets and embeddings from Hugging Face Hub. |
| **`scripts/upload_to_hf.py`** | Resilient file-by-file dataset uploader with automatic retry and resume capability. |
| **`scripts/RAG-v2/conversational_legal_rag.py`** | Primary 8-stage conversational RAG pipeline engine. |
| **`scripts/RAG-v2/conversational_ilsic_mapper.py`** | ILSIC scenario mapper, Groq API client with multi-model fallback, and JSON repair. |
| **`scripts/RAG-v2/build_rag_v2_corpus.py`** | Script to compile and normalize the 858 Central Acts into section-level JSONL. |
| **`scripts/RAG-v2/build_rag_v2_embeddings.py`** | Script to generate 384-dimensional BGE embeddings for the legal corpus. |
| **`scripts/RAG-v2/build_ilsic_scenario_index.py`** | Script to build the ILSIC scenario and chunk embeddings. |
| **`scripts/synthetic/generate_ilsic_synthetic_pilot.py`** | Synthetic legal dispute generation pipeline for supervised fine-tuning (SFT). |
| **`scripts/parse_sc_format_aware.py`** | Format-aware parser for legacy JUDIS and modern Supreme Court PDF judgments. |
| **`scripts/audit_act_hierarchy.py`** | Recursive parser discovering nested sections inside Chapters and Parts of Central Acts. |

---

## 9. Future Roadmap & Continued Research

1. **Domain-Adapted Continued Pre-Training (CPT)**:
   - Perform continued pre-training on `HuggingFaceTB/SmolLM3-3B-Base` using `scripts/data/training/cpt/clean/legal_cpt_clean.jsonl` (~1.09 GB of high-quality statutory and judicial text) to instill native Indian legal vocabulary.
2. **Supervised Fine-Tuning (SFT) for Multi-Turn Intake**:
   - Fine-tune local 3B–8B models on verified conversational legal pairs (`scripts/data/training/sft/synthetic_pilot/`) to handle the clarification interview natively without relying on external API calls.
3. **Multi-Agent Judicial Simulation**:
   - Implement adversary agents (Petitioner Advocate, Respondent Advocate, Judicial Bench) that debate ambiguous statutory interpretations before finalizing litigation strategy.
4. **State-Specific Amendment Modules**:
   - Expand the Authority Catalog to index state amendments (e.g., Maharashtra or Tamil Nadu amendments to the Code of Civil Procedure or Stamp Act).

---

*Document compiled and verified for Decepticons Legal AI (LLM-Forge).*
