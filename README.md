# Decepticons Legal AI (LLM-Forge)
### Indian Legal Reasoning & Litigation Strategy Assistant

An advanced, authority-grounded Legal AI assistant designed to analyze factual legal disputes under Indian Law. The system integrates the **Constitution of India**, **Central Acts**, and **Supreme Court jurisprudence** using a multi-stage conversational Retrieval-Augmented Generation (RAG v2) pipeline.

---

## 🌟 Key Features

- **Conversational Fact Intake**: Dynamically identifies missing material facts in complex user scenarios and asks targeted clarification questions before rendering an assessment.
- **Factual Precedent Analogy**: Retrieves and evaluates analogous case scenarios using the Indian Legal Scenario (ILSIC) dataset.
- **Statutory Authority Verification**: Matches legal issues against an indexed authority catalog of Central Acts and constitutional provisions.
- **Two-Tier Grounding & Claim Audit**: Strictly audits generated draft claims against retrieved statutory excerpts to eliminate legal hallucinations before presenting the final opinion.
- **Premium Client-Facing UI**: Clean, modern dark-themed web interface with interactive follow-up question cards and verified statutory citation badges (`[S1]`, `[S2]`).

---

## 🏛️ System Architecture

The consultation engine operates through an 8-stage verification pipeline:

```
[User Query] 
     │
     ▼
[Stage 1: Conversational Intake] ──► (Clarification Questions if needed)
     │
     ▼
[Stage 2 & 3: ILSIC Scenario Retrieval & Analogy Reranking]
     │
     ▼
[Stage 4: Candidate Statutory Provision Verification]
     │
     ▼
[Stage 5: Legal-Hypothesis & Authority Target Formulation]
     │
     ▼
[Stage 6: Dense Retrieval from Central Acts & Supreme Court Corpus (BGE)]
     │
     ▼
[Stage 7: Passage Applicability Verification & Source Reranking]
     │
     ▼
[Stage 8: Claim-Citation Grounding Audit & Finished Consultation Output]
     │
     ▼
[Interactive Web UI on http://localhost:8000]
```

---

## 🚀 Quick Start & Setup Guide

### 1. Prerequisites
- **Python 3.10** or higher
- **Git**
- A **Groq API Key** (free tier available at [console.groq.com](https://console.groq.com/))
- (Optional) CUDA-compatible GPU for accelerated local BGE embeddings

---

### 2. Clone the Repository
```bash
git clone https://github.com/Raghul-676/LLM-Forge-decepticons.git
cd LLM-Forge-decepticons
```

---

### 3. Create & Activate a Virtual Environment

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

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 5. Configure Environment Variables
Copy the example environment configuration file to `.env`:

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and configure your API keys:
```env
HF_TOKEN=your_huggingface_token_here
GROQ_API_KEY=your_groq_api_key_here
```

---

### 6. Download Required Datasets & Embeddings
Download the indexed legal corpus, ILSIC scenarios, and pre-computed BGE embeddings from the Hugging Face dataset:  
🔗 **Dataset Repository**: [https://huggingface.co/datasets/raghul12345678/decepticons-legal-rag-data](https://huggingface.co/datasets/raghul12345678/decepticons-legal-rag-data)

Run the included automated download script:
```bash
python scripts/download_data.py
```

This downloads and populates `scripts/data/rag/` automatically with all required runtime files (~2.77 GB).

---

## 🖥️ Launching the Web Chatbot

### Method 1: Windows Batch File (Recommended)
Double-click `run_ui.bat` or run:
```cmd
.\run_ui.bat
```

### Method 2: Python Command
From the activated virtual environment, start the server:
```bash
python ui/server.py 8000
```

> **Note**: On startup, the server requires ~15–20 seconds to load the legal index catalog, scenario maps, and embedding models. Once initialized, the console will display:
> ```text
> ======================================================================
> DECEPTICONS LEGAL AI WEB SERVER
> URL: http://127.0.0.1:8000
> ======================================================================
> ```

Open your browser and navigate to:
👉 **`http://localhost:8000`**

---

## 📁 Repository Structure

```
├── .env.example                     # Sample environment variable template
├── .gitignore                       # Git ignore rules for venv, secrets, and datasets
├── README.md                        # Project documentation and setup guide
├── requirements.txt                 # Python package dependencies
├── run_ui.bat                       # One-click Windows startup script
│
├── ui/                              # Web Interface & Backend Server
│   ├── index.html                   # Modern legal assistant interface
│   ├── index.css                    # Curated styling, design tokens & animations
│   ├── app.js                       # Frontend state, polling & multi-turn interaction
│   └── server.py                    # Multi-threaded HTTP API & session pipeline runner
│
├── scripts/                         # Core Machine Learning & RAG Pipeline
│   ├── RAG-v2/                      # Active RAG v2 production modules
│   │   ├── conversational_legal_rag.py   # Full 8-stage legal RAG pipeline
│   │   ├── conversational_ilsic_mapper.py # Scenario analogy mapper & LLM caller
│   │   ├── build_rag_v2_corpus.py        # Legal document corpus builder
│   │   └── build_rag_v2_embeddings.py    # Dense vector embedding generator
│   └── synthetic/                   # Synthetic dataset generation & testing
│
└── evaluation/                      # Model benchmarking and evaluation scripts
```

---

## ⚖️ Legal Disclaimer

This project is an experimental artificial intelligence prototype developed for legal research, educational, and workflow demonstration purposes. It references Indian statutes and judicial precedents, but **does not constitute formal legal advice**. For binding legal counsel or court representation, always consult a qualified advocate or legal practitioner.

---
