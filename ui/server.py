import os
import sys
import json
import time
import uuid
import threading
import urllib.parse
from pathlib import Path
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# ============================================================
# RESOLVE PATHS & ADD RAG-v2 TO PYTHON PATH
# ============================================================

UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent
RAG_V2_DIR = PROJECT_ROOT / "scripts" / "RAG-v2"

if str(RAG_V2_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_V2_DIR))

# Ensure working directory resolves files properly
os.chdir(PROJECT_ROOT)

# Pre-import project modules
import conversational_ilsic_mapper as mapper
import conversational_legal_rag as rag

# ============================================================
# GLOBAL ENGINE STATE
# ============================================================

class Engine:
    def __init__(self):
        self.is_ready = False
        self.load_error = None
        self.loading_status = "Initializing..."
        self.ilsic_scenario_map = None
        self.ilsic_chunks = None
        self.ilsic_embeddings = None
        self.legal_embeddings = None
        self.legal_offsets = None
        self.authority_catalog = None
        self.embedding_model = None
        self.groq_client = None
        self.device = "cpu"

    def initialize(self):
        try:
            self.loading_status = "Loading ILSIC scenario index..."
            (
                self.ilsic_scenario_map,
                self.ilsic_chunks,
                self.ilsic_embeddings
            ) = mapper.load_ilsic_index()

            self.loading_status = "Loading 710K Legal embeddings (mmap)..."
            self.legal_embeddings = rag.np.load(
                rag.LEGAL_EMBEDDINGS_FILE,
                mmap_mode="r"
            )

            self.loading_status = "Loading document offsets & authority catalog..."
            self.legal_offsets = rag.build_or_load_offsets(
                expected_rows=self.legal_embeddings.shape[0]
            )
            self.authority_catalog = rag.build_or_load_authority_catalog(
                expected_rows=self.legal_embeddings.shape[0]
            )

            self.device = "cuda" if rag.torch.cuda.is_available() else "cpu"
            self.loading_status = f"Loading BGE-small embedding model on {self.device}..."
            self.embedding_model = rag.SentenceTransformer(
                rag.EMBEDDING_MODEL,
                device=self.device
            )
            self.embedding_model.max_seq_length = 512

            self.loading_status = "Connecting to Groq API..."
            self.groq_client = rag.Groq(api_key=mapper.GROQ_API_KEY)

            self.is_ready = True
            self.loading_status = "Ready - 710,029 Legal Chunks Active"
            print(f"[Engine] Ready on device: {self.device}")
        except Exception as exc:
            self.load_error = str(exc)
            self.loading_status = f"Error during initialization: {exc}"
            print(f"[Engine ERROR] {exc}", file=sys.stderr)

engine = Engine()

# ============================================================
# USER-FACING RESPONSE CONTROLLER
# ============================================================

def generate_user_facing_response(original_problem, followups, selected_sources, audit, client):
    conversation = mapper.format_conversation(original_problem, followups)
    allowed_claims = rag.get_allowed_audit_claims(audit)

    if not allowed_claims:
        return (
            "### Your situation\n"
            f"{rag.shorten(original_problem, 300)}\n\n"
            "### What the verified legal sources indicate\n"
            "The available verified legal sources are not sufficient to give a reliable legal answer for this scenario. Under our verification standards, unverified assertions are withheld to ensure all legal information is authoritatively supported.\n\n"
            "### What you can do next\n"
            "- Consult a qualified advocate with all relevant original records, transaction agreements, and written correspondence.\n"
            "- Determine whether specialized administrative or state-level forums have specific jurisdiction over the dispute.\n\n"
            "### Important uncertainty\n"
            "The facts presented could not be definitively matched with binding statutory provisions or verified precedents in the current authority index."
        )

    # Build reference mapping for sources so the model cites accurately
    sources_summary = []
    for idx, item in enumerate(selected_sources, start=1):
        doc = item.get("document", {})
        title = rag.get_source_title(doc) or "Statutory Authority"
        sec = rag.get_section(doc)
        sec_str = f", {sec}" if sec else ""
        sources_summary.append(f"[S{idx}] {title}{sec_str}")
    sources_text = "\n".join(sources_summary)

    allowed_text = json.dumps(allowed_claims, ensure_ascii=False, indent=2)

    system_prompt = """
You are the USER-FACING RESPONSE CONTROLLER for an Indian Legal AI prototype.

Your job is to decide what information should be shown in the website interface after the user has answered all required questions.
Show ONLY information that is directly useful, understandable, and relevant to the user's legal problem.

STRICT CONTENT RESTRICTIONS:
- Do NOT display internal processing details, debugging information, retrieval traces, ranking scores, similarity scores, semantic-search results, raw retrieved chunks, candidate scenarios, rejected scenarios, candidate-law voting, intermediate hypotheses, internal verification steps, model reasoning, JSON output, vector-search details, source-ranking calculations, query expansion details, internal warnings, pipeline stage names, or any implementation/debug information.
- Do NOT expose content merely because it was retrieved internally.
- Every legal statement you make must be supported by the verified final sources in the ALLOW-LIST.

REQUIRED RESPONSE STRUCTURE:
Use exactly these concise section headers:

### Your situation
A concise 1-2 sentence summary of the user's situation and core dispute.

### What the verified legal sources indicate
The final legal answer relevant to the user's facts. State clearly and conversationally for a non-lawyer the applicable legal provisions or principles that have been verified.
Cite the verified sources inline using brackets like [S1], [S2] immediately after each proposition. Use ONLY source IDs that appear in the supported_by field of that claim.

### What you can do next
Practical next steps supported by the verified legal analysis (e.g. issuing a formal legal demand notice within statutory timelines, preserving records, or filing before the competent forum).

### Important uncertainty
Important limitations, missing facts, limitation periods, or uncertainties that materially affect the answer.

### Sources
List only the sources actually cited above:
- [S1] Act Name, Section / Citation
- [S2] Act Name, Section / Citation

### Legal status caution
A short 1-2 sentence caution noting that research references statutory laws and precedents, and current state-specific amendments or local court rules should be verified before formal court reliance.

Keep the response clear, authoritative, and conversational for a non-lawyer.
Write purely as a finished legal-assistance response, not the output of a research pipeline.
"""

    user_prompt = f"""
USER FACTS:
{conversation}

VERIFIED ALLOW-LIST CLAIMS:
{allowed_text}

AVAILABLE VERIFIED SOURCES FOR CITATION:
{sources_text}

Write the finished legal consultation response using only the allowed claims and required sections.
"""

    try:
        response = mapper.groq_text(
            client=client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=rag.FINAL_ANSWER_MAX_TOKENS + 350,
            reasoning_effort="low"
        )
        return rag.normalize_source_citations(response)
    except Exception as exc:
        print(f"Warning: user-facing response writing fallback: {exc}")
        return (
            "### Your situation\n"
            f"{rag.shorten(original_problem, 300)}\n\n"
            "### What the verified legal sources indicate\n"
            "The available verified sources are not sufficient to give a reliable legal answer at this time.\n\n"
            "### What you can do next\n"
            "- Consult a practicing advocate with all original dispute records.\n\n"
            "### Important uncertainty\n"
            "Additional documentary evidence is required to establish applicable statutory rights."
        )

# ============================================================
# SESSION MANAGEMENT
# ============================================================

class LegalSession:
    def __init__(self, session_id, problem):
        self.session_id = session_id
        self.original_problem = problem
        self.followups = []
        self.asked_question_keys = set()
        self.total_followups = 0
        self.research_iteration = 0
        self.events = []
        self.status = "running" # 'running', 'waiting_clarification', 'completed', 'failed'
        self.clarification_question = None
        self.user_answer_event = threading.Event()
        self.user_answer_value = None
        self.sources = []
        self.final_answer = None
        self.audit = None
        self.thread = None
        self.current_stage = 0
        self.lock = threading.Lock()

    def add_event(self, event_type, stage=None, title="", data=None):
        with self.lock:
            evt = {
                "id": len(self.events),
                "timestamp": time.time(),
                "type": event_type,
                "stage": stage,
                "title": title,
                "data": data or {}
            }
            self.events.append(evt)

    def ask_user_clarification(self, question):
        normalized = rag.normalize_text(question)
        key = mapper.normalize_question(normalized)
        if key in self.asked_question_keys:
            return False

        self.asked_question_keys.add(key)
        self.clarification_question = normalized
        self.status = "waiting_clarification"
        self.user_answer_event.clear()
        self.user_answer_value = None

        self.add_event(
            "CLARIFICATION_REQUIRED",
            stage=self.current_stage,
            title="Clarification Needed",
            data={"question": normalized}
        )

        # Wait up to 10 minutes for user response
        answered = self.user_answer_event.wait(timeout=600)
        if not answered or not self.user_answer_value:
            return False

        ans = self.user_answer_value.strip()
        self.followups.append({
            "question": normalized,
            "answer": ans
        })
        self.total_followups += 1
        self.status = "running"
        self.clarification_question = None

        self.add_event(
            "CLARIFICATION_PROVIDED",
            stage=self.current_stage,
            title="Detail Added",
            data={"question": normalized, "answer": ans}
        )
        return True

    def submit_answer(self, answer_text):
        self.user_answer_value = answer_text
        self.user_answer_event.set()

    def run_pipeline(self):
        try:
            self.add_event("PIPELINE_STARTED", stage=0, title="Pipeline Started", data={"problem": self.original_problem})

            while self.research_iteration < rag.MAX_RESEARCH_ITERATIONS:
                self.research_iteration += 1
                self.add_event(
                    "ITERATION_START",
                    stage=1,
                    title=f"Research Iteration {self.research_iteration}",
                    data={"iteration": self.research_iteration}
                )

                # ----------------------------------------------------
                # 1. CONVERSATIONAL INTAKE
                # ----------------------------------------------------
                self.current_stage = 1
                self.add_event("STAGE_START", stage=1, title="Analyzing case facts...", data={})

                while True:
                    try:
                        analysis = mapper.analyze_intake(
                            original_problem=self.original_problem,
                            followups=self.followups,
                            client=engine.groq_client,
                            questions_already_asked=self.total_followups
                        )
                    except Exception as exc:
                        print(f"Warning: intake analysis: {exc}")
                        analysis = {"status": "READY", "normalized_scenario": self.original_problem, "retrieval_queries": []}

                    status = rag.normalize_text(analysis.get("status", "")).upper()
                    if status == "ASK" and self.total_followups < 6:
                        q = rag.normalize_text(analysis.get("next_question", ""))
                        if q:
                            asked = self.ask_user_clarification(q)
                            if asked:
                                continue
                    break

                normalized_scenario = rag.normalize_text(analysis.get("normalized_scenario", ""))
                retrieval_queries = analysis.get("retrieval_queries", [])
                if not normalized_scenario:
                    normalized_scenario = rag.build_fallback_scenario(self.original_problem, self.followups)
                    retrieval_queries = []

                self.add_event("STAGE_COMPLETE", stage=1, title="Case facts analyzed", data={
                    "normalized_scenario": normalized_scenario,
                    "retrieval_queries": retrieval_queries
                })

                # ----------------------------------------------------
                # 2. SCENARIO RETRIEVAL
                # ----------------------------------------------------
                self.current_stage = 2
                self.add_event("STAGE_START", stage=2, title="Searching legal precedents...", data={})

                search_queries = mapper.build_search_queries(
                    original_problem=self.original_problem,
                    normalized_scenario=normalized_scenario,
                    retrieval_queries=retrieval_queries
                )

                candidates = mapper.retrieve_candidate_scenarios(
                    search_queries=search_queries,
                    embedding_model=engine.embedding_model,
                    scenario_map=engine.ilsic_scenario_map,
                    chunks=engine.ilsic_chunks,
                    embeddings=engine.ilsic_embeddings
                )

                candidate_previews = [
                    {
                        "scenario_id": c.get("scenario_id"),
                        "score": round(float(c.get("score", 0)), 3),
                        "preview": rag.shorten(c.get("scenario_text", ""), 200),
                        "sections": c.get("sections", [])
                    }
                    for c in candidates[:6]
                ]

                self.add_event("STAGE_COMPLETE", stage=2, title="Analogous precedents retrieved", data={
                    "count": len(candidates),
                    "candidates": candidate_previews
                })

                # ----------------------------------------------------
                # 3. FACTUAL ANALOGY CHECK
                # ----------------------------------------------------
                self.current_stage = 3
                self.add_event("STAGE_START", stage=3, title="Evaluating case analogies...", data={})

                try:
                    reranked_scenarios, accepted_scenarios = mapper.rerank_analogous_scenarios(
                        original_problem=self.original_problem,
                        followups=self.followups,
                        normalized_scenario=normalized_scenario,
                        candidates=candidates,
                        client=engine.groq_client
                    )
                except Exception as exc:
                    print(f"Warning: scenario reranking: {exc}")
                    reranked_scenarios = []
                    accepted_scenarios = []

                self.add_event("STAGE_COMPLETE", stage=3, title="Case analogies evaluated", data={
                    "accepted_count": len(accepted_scenarios),
                    "accepted": [
                        {
                            "scenario_id": item.get("scenario_id"),
                            "verdict": item.get("verdict"),
                            "reason": item.get("reason"),
                            "score": item.get("score")
                        }
                        for item in accepted_scenarios[:5]
                    ]
                })

                # ----------------------------------------------------
                # 4. CANDIDATE STATUTORY VERIFICATION
                # ----------------------------------------------------
                self.current_stage = 4
                self.add_event("STAGE_START", stage=4, title="Identifying applicable statutory provisions...", data={})

                candidate_verification = {"candidate_laws": [], "needs_more_information": False, "next_question": ""}
                candidate_laws = []

                if accepted_scenarios:
                    candidate_laws = mapper.aggregate_candidate_laws(accepted_scenarios)
                    try:
                        candidate_verification = mapper.verify_candidate_laws(
                            original_problem=self.original_problem,
                            followups=self.followups,
                            normalized_scenario=normalized_scenario,
                            accepted=accepted_scenarios,
                            candidate_laws=candidate_laws,
                            client=engine.groq_client
                        )
                    except Exception as exc:
                        print(f"Warning: candidate law verification: {exc}")

                if candidate_verification.get("needs_more_information") and self.total_followups < rag.MAX_TOTAL_FOLLOWUPS:
                    q = rag.normalize_text(candidate_verification.get("next_question", ""))
                    if q:
                        asked = self.ask_user_clarification(q)
                        if asked:
                            continue

                kept_authorities = mapper.get_kept_authorities(candidate_verification)
                self.add_event("STAGE_COMPLETE", stage=4, title="Statutory provisions verified", data={
                    "candidate_laws": candidate_laws[:6],
                    "kept_authorities": kept_authorities
                })

                # ----------------------------------------------------
                # 5. LEGAL-HYPOTHESIS GENERATION
                # ----------------------------------------------------
                self.current_stage = 5
                self.add_event("STAGE_START", stage=5, title="Formulating legal issues & statutory targets...", data={})

                try:
                    hypotheses = rag.generate_research_hypotheses(
                        original_problem=self.original_problem,
                        followups=self.followups,
                        normalized_scenario=normalized_scenario,
                        candidate_verification=candidate_verification,
                        client=engine.groq_client
                    )
                except Exception as exc:
                    print(f"Warning: hypothesis generation: {exc}")
                    hypotheses = rag.build_deterministic_fallback_hypotheses(
                        original_problem=self.original_problem,
                        followups=self.followups,
                        normalized_scenario=normalized_scenario
                    )

                authority_targets, authority_validation = rag.prepare_authority_targets(
                    hypotheses=hypotheses,
                    authority_catalog=engine.authority_catalog
                )

                self.add_event("STAGE_COMPLETE", stage=5, title="Legal issues and provisions formulated", data={
                    "hypotheses": hypotheses,
                    "authority_targets": [
                        {
                            "act_title": t.get("act_title"),
                            "act_key": t.get("act_key"),
                            "status": t.get("status"),
                            "has_exact_section": bool(t.get("exact_section_rows"))
                        }
                        for t in authority_targets[:6]
                    ]
                })

                # ----------------------------------------------------
                # 6. SEARCH LEGAL CORPUS
                # ----------------------------------------------------
                self.current_stage = 6
                self.add_event("STAGE_START", stage=6, title="Retrieving relevant provisions and judicial precedents...", data={})

                legal_queries = rag.build_legal_queries(
                    original_problem=self.original_problem,
                    normalized_scenario=normalized_scenario,
                    kept_authorities=kept_authorities,
                    hypotheses=hypotheses,
                    authority_targets=authority_targets
                )

                legal_candidates = rag.search_legal_corpus(
                    queries=legal_queries,
                    embedding_model=engine.embedding_model,
                    legal_embeddings=engine.legal_embeddings,
                    offsets=engine.legal_offsets,
                    authority_targets=authority_targets
                )

                if not legal_candidates:
                    self.final_answer = (
                        "### Your situation\n"
                        f"{rag.shorten(self.original_problem, 300)}\n\n"
                        "### What the verified legal sources indicate\n"
                        "The available verified legal sources are not sufficient to give a reliable answer for this scenario.\n\n"
                        "### What you can do next\n"
                        "- Consult a qualified legal practitioner with all original transaction documents and relevant correspondence.\n\n"
                        "### Important uncertainty\n"
                        "The dispute facts could not be reliably matched with verified statutory provisions or binding judicial authorities in the current index."
                    )
                    self.status = "completed"
                    return

                # ----------------------------------------------------
                # 7. LEGAL SOURCE VERIFICATION & RERANKING
                # ----------------------------------------------------
                self.current_stage = 7

                try:
                    reranked_sources, selected_sources, source_verification = rag.rerank_legal_sources(
                        original_problem=self.original_problem,
                        followups=self.followups,
                        normalized_scenario=normalized_scenario,
                        hypotheses=hypotheses,
                        candidates=legal_candidates,
                        client=engine.groq_client
                    )
                except Exception as exc:
                    print(f"Warning: source reranking: {exc}")
                    reranked_sources = []
                    selected_sources = []
                    source_verification = {"results": [], "needs_more_information": False, "next_question": ""}

                if source_verification.get("needs_more_information") and self.total_followups < rag.MAX_TOTAL_FOLLOWUPS:
                    q = rag.normalize_text(source_verification.get("next_question", ""))
                    if q:
                        asked = self.ask_user_clarification(q)
                        if asked:
                            continue

                if not selected_sources:
                    self.final_answer = (
                        "### Your situation\n"
                        f"{rag.shorten(self.original_problem, 300)}\n\n"
                        "### What the verified legal sources indicate\n"
                        "The available verified legal sources are not sufficient to give a reliable answer for this scenario.\n\n"
                        "### What you can do next\n"
                        "- Consult a qualified legal practitioner with all original transaction documents and relevant correspondence.\n\n"
                        "### Important uncertainty\n"
                        "The retrieved authorities did not meet the required legal verification threshold to apply with certainty to these facts."
                    )
                    self.status = "completed"
                    return

                # ----------------------------------------------------
                # 8. CITATION AUDIT & FINAL USER-FACING ANSWER
                # ----------------------------------------------------
                self.current_stage = 8

                try:
                    draft = rag.generate_draft_answer(
                        original_problem=self.original_problem,
                        followups=self.followups,
                        selected=selected_sources,
                        client=engine.groq_client
                    )
                except Exception as exc:
                    print(f"Warning: draft answer generation: {exc}")
                    draft = rag.build_deterministic_preliminary_answer(selected_sources) if hasattr(rag, "build_deterministic_preliminary_answer") else ""

                try:
                    audit = rag.audit_draft_claims(
                        original_problem=self.original_problem,
                        followups=self.followups,
                        selected=selected_sources,
                        draft=draft,
                        client=engine.groq_client
                    )
                except Exception as exc:
                    print(f"Warning: audit claims: {exc}")
                    audit = {"claims": [], "audit_failed": True, "audit_partial": False}
                self.audit = audit

                # Generate finished structured legal advice
                answer = generate_user_facing_response(
                    original_problem=self.original_problem,
                    followups=self.followups,
                    selected_sources=selected_sources,
                    audit=audit,
                    client=engine.groq_client
                )

                # Second grounding audit on the legal propositions section
                legal_section = ""
                m = rag.re.search(r"### What the verified legal sources indicate\s*\n(.*?)(?=\n###|\Z)", answer, rag.re.DOTALL)
                legal_section = m.group(1).strip() if m else answer

                final_audit = rag.audit_draft_claims(
                    original_problem=self.original_problem,
                    followups=self.followups,
                    selected=selected_sources,
                    draft=legal_section,
                    client=engine.groq_client
                )

                final_enforced = rag.final_answer_passes_audit(final_audit)
                if not final_enforced:
                    allowed_claims = rag.get_allowed_audit_claims(audit)
                    if allowed_claims:
                        claims_bullets = "\n".join([f"- {c['safe_claim']} {''.join(f'[{s}]' for s in c.get('supported_by', []))}" for c in allowed_claims])
                        answer = (
                            "### Your situation\n"
                            f"{rag.shorten(self.original_problem, 300)}\n\n"
                            "### What the verified legal sources indicate\n"
                            f"{claims_bullets}\n\n"
                            "### What you can do next\n"
                            "- Consult an advocate to review primary transaction records and court jurisdiction.\n"
                            "- Ensure legal proceedings are commenced within the applicable statutory limitation period.\n\n"
                            "### Important uncertainty\n"
                            "Matters beyond the specific verified statutory propositions above remain unverified by the current authorities.\n\n"
                            "### Legal status caution\n"
                            "The above references statutory provisions and judicial authorities, but state amendments and local court rules should be verified before formal court reliance."
                        )
                    else:
                        answer = (
                            "### Your situation\n"
                            f"{rag.shorten(self.original_problem, 300)}\n\n"
                            "### What the verified legal sources indicate\n"
                            "The available verified legal sources are not sufficient to give a reliable answer for this scenario.\n\n"
                            "### What you can do next\n"
                            "- Consult a qualified legal practitioner with all original transaction documents and relevant correspondence.\n\n"
                            "### Important uncertainty\n"
                            "The factual dispute could not be reliably matched with binding statutory provisions or verified precedents in the current authority index."
                        )

                # Extract ONLY sources actually cited in the final answer
                formatted_sources = []
                used_ids = {f"S{match}" for match in rag.re.findall(r"\[S(\d+)\]", answer)}

                for num, item in enumerate(selected_sources, start=1):
                    sid = f"S{num}"
                    if sid not in used_ids:
                        continue # Only show sources actually cited

                    doc = item.get("document", {})
                    stype = rag.get_source_type(doc)
                    if stype == "central_act":
                        type_label = "Central Act"
                    elif stype == "constitution":
                        type_label = "Constitution of India"
                    else:
                        type_label = "Supreme Court Judgment"

                    formatted_sources.append({
                        "id": sid,
                        "title": rag.get_source_title(doc) or "Statutory Authority",
                        "section": rag.get_section(doc) or "",
                        "source_type": type_label,
                        "text": rag.shorten(rag.get_document_text(doc), 600)
                    })

                self.sources = formatted_sources
                self.final_answer = answer.strip()
                self.status = "completed"
                return

            self.status = "completed"
            if not self.final_answer:
                self.final_answer = (
                    "### Your situation\n"
                    f"{rag.shorten(self.original_problem, 300)}\n\n"
                    "### What the verified legal sources indicate\n"
                    "The available verified sources are not sufficient to give a reliable answer for this scenario.\n\n"
                    "### What you can do next\n"
                    "- Consult a qualified advocate with all relevant documentation."
                )
        except Exception as exc:
            print(f"[Session {self.session_id} Error] {exc}", file=sys.stderr)
            self.status = "completed"
            if not self.final_answer:
                self.final_answer = (
                    "### Your situation\n"
                    f"{rag.shorten(self.original_problem, 300)}\n\n"
                    "### What the verified legal sources indicate\n"
                    "The system encountered an unexpected error while finalizing the analysis. Please check your query and try again.\n\n"
                    "### What you can do next\n"
                    "- Consult a qualified legal practitioner with all original transaction documents and relevant correspondence.\n\n"
                    "### Important uncertainty\n"
                    "An unexpected service limit or processing error occurred."
                )

sessions = {}

# ============================================================
# SAMPLE CASES FOR DEMO
# ============================================================

SAMPLE_CASES = [
    {
        "id": "cheque_bounce",
        "title": "Cheque Bounce (Section 138 NI Act)",
        "summary": "A business client issued a cheque for ₹2,50,000 against invoices. It was returned unpaid due to 'Insufficient Funds'.",
        "problem": "A business client issued a cheque of ₹2,50,000 to pay for supplied goods. The bank returned the cheque with a memo stating 'insufficient funds'. Can I initiate criminal proceedings against the drawer under Indian law?"
    },
    {
        "id": "friendly_loan",
        "title": "Unpaid Friendly Loan & Recovery",
        "summary": "Lent ₹1,00,000 via bank transfer to a friend with written promise, but repayment has been refused.",
        "problem": "I lent ₹1,00,000 to a friend via bank transfer two years ago. He acknowledged the loan over WhatsApp and in a signed receipt, but is now refusing to return the money. What civil remedies and limitation periods apply in India?"
    },
    {
        "id": "consumer_defect",
        "title": "Consumer Product Defect & Warranty",
        "summary": "Purchased an expensive laptop that failed within 2 weeks. The authorized center refuses replacement.",
        "problem": "I purchased a premium laptop from an authorized dealer. Within 15 days the motherboard died. The service center refuses to replace the laptop and only offers slow repairs. What rights do I have under the Consumer Protection Act?"
    },
    {
        "id": "writ_petition",
        "title": "Arbitrary License Cancellation (Writ Article 226)",
        "summary": "Municipal corporation cancelled a trade license without issuing a show-cause notice or hearing.",
        "problem": "The municipal authority cancelled my valid trade license without issuing any show-cause notice or giving me an opportunity of being heard, violating natural justice. Can I file a writ petition in the High Court under Article 226?"
    }
]

# ============================================================
# HTTP REQUEST HANDLER
# ============================================================

class LegalHTTPHandler(BaseHTTPRequestHandler):

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # Static file routing
        if path == "/" or path == "/index.html":
            return self.serve_file(UI_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/index.css":
            return self.serve_file(UI_DIR / "index.css", "text/css; charset=utf-8")
        elif path == "/app.js":
            return self.serve_file(UI_DIR / "app.js", "application/javascript; charset=utf-8")

        # API Endpoints
        if path == "/api/status":
            return self.send_json({
                "ready": engine.is_ready,
                "status": "Ready" if engine.is_ready else "Initializing..."
            })

        if path == "/api/examples":
            return self.send_json({"examples": SAMPLE_CASES})

        if path == "/api/session/poll":
            session_id = query.get("id", [None])[0]
            if not session_id or session_id not in sessions:
                return self.send_json({"error": "Session not found"}, status=HTTPStatus.NOT_FOUND)

            sess = sessions[session_id]
            with sess.lock:
                return self.send_json({
                    "session_id": sess.session_id,
                    "status": sess.status,
                    "clarification_question": sess.clarification_question,
                    "sources": sess.sources,
                    "final_answer": sess.final_answer
                })

        if path == "/api/source":
            session_id = query.get("session_id", [None])[0]
            source_id = query.get("id", [None])[0]
            if not session_id or session_id not in sessions:
                return self.send_json({"error": "Session not found"}, status=HTTPStatus.NOT_FOUND)

            sess = sessions[session_id]
            matched = next((s for s in sess.sources if s["id"] == source_id), None)
            if not matched:
                return self.send_json({"error": "Source not found"}, status=HTTPStatus.NOT_FOUND)
            return self.send_json({"source": matched})

        self.send_error(HTTPStatus.NOT_FOUND, "Resource not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if path == "/api/session/start":
            if not engine.is_ready:
                return self.send_json({
                    "error": "Engine is still initializing. Please wait.",
                    "status": engine.loading_status
                }, status=HTTPStatus.SERVICE_UNAVAILABLE)

            problem = payload.get("problem", "").strip()
            if not problem:
                return self.send_json({"error": "Legal problem description is required."}, status=HTTPStatus.BAD_REQUEST)

            session_id = str(uuid.uuid4())[:8]
            sess = LegalSession(session_id, problem)
            sessions[session_id] = sess

            t = threading.Thread(target=sess.run_pipeline, daemon=True)
            sess.thread = t
            t.start()

            return self.send_json({
                "session_id": session_id,
                "status": "started"
            })

        if path == "/api/session/respond":
            session_id = payload.get("session_id")
            answer = payload.get("answer", "").strip()
            if not session_id or session_id not in sessions:
                return self.send_json({"error": "Session not found"}, status=HTTPStatus.NOT_FOUND)

            sess = sessions[session_id]
            if sess.status != "waiting_clarification":
                return self.send_json({"error": "Session is not awaiting clarification"}, status=HTTPStatus.BAD_REQUEST)

            sess.submit_answer(answer)
            return self.send_json({"status": "received"})

        self.send_error(HTTPStatus.NOT_FOUND, "Resource not found")

    def serve_file(self, file_path, content_type):
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, f"File {file_path.name} not found")
            return
        with open(file_path, "rb") as f:
            content = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        # Concise logging to keep terminal clean
        try:
            msg = format % args
            if "/api/session/poll" not in msg:
                print(f"[HTTP] {self.address_string()} - {msg}")
        except Exception:
            pass

# ============================================================
# SERVER STARTUP
# ============================================================

def run_server(host="127.0.0.1", port=8000):
    print("=" * 70)
    print("DECEPTICONS LEGAL AI WEB SERVER")
    print(f"URL: http://{host}:{port}")
    print("=" * 70)

    # Initialize engine in a background thread so the HTTP server responds immediately
    init_thread = threading.Thread(target=engine.initialize, daemon=True)
    init_thread.start()

    server = ThreadingHTTPServer((host, port), LegalHTTPHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server.")
        server.shutdown()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port=port)
