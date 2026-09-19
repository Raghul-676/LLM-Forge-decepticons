/* ============================================================
   INDIAN LEGAL AI ASSISTANT — USER-FACING INTERFACE CONTROLLER
   ============================================================ */

(() => {
  // DOM Elements
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const btnReset = document.getElementById("btnReset");
  const caseForm = document.getElementById("caseForm");
  const caseInput = document.getElementById("caseInput");
  const btnSubmit = document.getElementById("btnSubmit");
  const emptyState = document.getElementById("emptyState");
  const chatScroll = document.getElementById("chatScroll");
  const messageStream = document.getElementById("messageStream");
  const sourceCountBadge = document.getElementById("sourceCountBadge");
  const sourcesPlaceholder = document.getElementById("sourcesPlaceholder");
  const sourcesList = document.getElementById("sourcesList");

  // State
  let activeSessionId = null;
  let lastEventId = -1;
  let pollTimer = null;
  let currentActivityCard = null;
  let isSubmitting = false;

  // ============================================================
  // SERVER HEALTH CHECK
  // ============================================================
  async function checkServerStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();

      if (data.ready) {
        if (statusDot) statusDot.className = "status-dot active";
        if (statusText) statusText.textContent = "Online • Ready";
        btnSubmit.disabled = isSubmitting;
      } else {
        if (statusDot) statusDot.className = "status-dot loading";
        if (statusText) statusText.textContent = "Initializing...";
        setTimeout(checkServerStatus, 2000);
      }
    } catch (err) {
      if (statusDot) statusDot.className = "status-dot";
      if (statusText) statusText.textContent = "Connecting...";
      setTimeout(checkServerStatus, 3000);
    }
  }

  // ============================================================
  // USER-FACING MESSAGE RENDERING
  // ============================================================
  function appendUserMessage(text) {
    emptyState.style.display = "none";
    const msg = document.createElement("div");
    msg.className = "message user";
    msg.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
    messageStream.appendChild(msg);
    scrollToBottom();
  }

  function showActivity(text) {
    if (!currentActivityCard) {
      currentActivityCard = document.createElement("div");
      currentActivityCard.className = "activity-card";
      messageStream.appendChild(currentActivityCard);
    }
    currentActivityCard.innerHTML = `
      <div class="activity-spinner"></div>
      <div class="activity-text"><strong>Analyzing:</strong> ${escapeHtml(text)}</div>
    `;
    scrollToBottom();
  }

  function removeActivity() {
    if (currentActivityCard) {
      currentActivityCard.remove();
      currentActivityCard = null;
    }
  }

  function appendClarificationPrompt(question) {
    removeActivity();
    const card = document.createElement("div");
    card.className = "clarification-card";
    card.id = "activeClarificationCard";
    card.innerHTML = `
      <div class="clarification-title">
        <span>⚠️</span>
        <span>Additional Legal Detail Required</span>
      </div>
      <div class="clarification-question">${escapeHtml(question)}</div>
      <div class="clarification-input-group">
        <input type="text" class="clarification-input" id="clarificationAnswerInput" placeholder="Provide details (dates, written terms, notices, or circumstances)..." autocomplete="off" />
        <button class="btn-gold" id="btnSubmitClarification">Submit Detail</button>
      </div>
    `;
    messageStream.appendChild(card);
    scrollToBottom();

    const input = card.querySelector("#clarificationAnswerInput");
    const submitBtn = card.querySelector("#btnSubmitClarification");

    function doSubmit() {
      const val = input.value.trim();
      if (!val) return;
      submitClarification(val, card);
    }

    submitBtn.addEventListener("click", doSubmit);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        doSubmit();
      }
    });
    input.focus();
  }

  async function submitClarification(answer, cardEl) {
    // Remove the active ID so subsequent follow-up questions can be appended seamlessly
    cardEl.removeAttribute("id");
    cardEl.className = "clarification-card clarification-submitted";
    cardEl.innerHTML = `
      <div class="clarification-title"><span>✓</span><span>Factual Detail Provided</span></div>
      <div style="font-size: 13.5px; color: #d1fae5; margin-top: 4px;">${escapeHtml(answer)}</div>
    `;
    showActivity("Reviewing updated facts and assessing if further details are required...");

    try {
      await fetch("/api/session/respond", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: activeSessionId,
          answer: answer
        })
      });
    } catch (err) {
      console.error("Error submitting clarification:", err);
    }
  }

  // ============================================================
  // FINISHED LEGAL OPINION FORMATTING
  // ============================================================
  function renderFinalOpinion(answerText) {
    removeActivity();

    const msg = document.createElement("div");
    msg.className = "message assistant";

    const formattedHtml = formatLegalProse(answerText);

    msg.innerHTML = `
      <div class="bubble">
        <div class="assistant-header">
          <div class="assistant-tag">
            <span>⚖️</span>
            <span>Legal Analysis & Strategic Assessment</span>
          </div>
        </div>
        <div class="legal-prose">
          ${formattedHtml}
        </div>
      </div>
    `;

    messageStream.appendChild(msg);

    // Attach pill click events to highlight corresponding source in right panel
    msg.querySelectorAll(".citation-pill").forEach(pill => {
      pill.addEventListener("click", () => {
        const sid = pill.getAttribute("data-source");
        highlightSourceCard(sid);
      });
    });

    scrollToBottom();
  }

  function formatLegalProse(text) {
    let safe = escapeHtml(text);

    // Citations [S1], [S2] -> interactive clickable pills
    safe = safe.replace(/\[S(\d+)\]/g, (match, p1) => {
      return `<button type="button" class="citation-pill" data-source="S${p1}" title="View supporting legal authority [S${p1}]">[S${p1}]</button>`;
    });

    const lines = safe.split(/\n/);
    const output = [];
    let inList = false;

    for (let rawLine of lines) {
      const line = rawLine.trim();
      if (!line) {
        if (inList) { output.push("</ul>"); inList = false; }
        continue;
      }

      // Check for headings
      const h3Match = line.match(/^###\s+(.*)$/);
      const h2Match = line.match(/^##\s+(.*)$/);
      const boldHeaderMatch = line.match(/^\*\*([^*]+)\*\*:?$/);

      if (h3Match || h2Match || boldHeaderMatch) {
        if (inList) { output.push("</ul>"); inList = false; }
        const headerText = (h3Match ? h3Match[1] : (h2Match ? h2Match[1] : boldHeaderMatch[1])).trim();
        output.push(`<h3 class="legal-section-header"><span>⚖️</span> <span>${headerText}</span></h3>`);
      } else if (line.startsWith("- ") || line.startsWith("* ")) {
        if (!inList) { output.push("<ul>"); inList = true; }
        output.push(`<li>${line.replace(/^[-*]\s+/, "")}</li>`);
      } else {
        if (inList) { output.push("</ul>"); inList = false; }
        // Bold formatting inside paragraph
        const formattedLine = line.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        output.push(`<p>${formattedLine}</p>`);
      }
    }

    if (inList) { output.push("</ul>"); }
    return output.join("\n");
  }

  // ============================================================
  // VERIFIED SOURCES LIST RENDERING
  // ============================================================
  function renderSourcesList(sources) {
    if (!sources || sources.length === 0) {
      sourceCountBadge.textContent = "0";
      sourcesPlaceholder.style.display = "block";
      sourcesList.innerHTML = "";
      return;
    }

    sourceCountBadge.textContent = sources.length;
    sourcesPlaceholder.style.display = "none";
    sourcesList.innerHTML = "";

    sources.forEach(src => {
      const card = document.createElement("div");
      card.className = "source-card";
      card.id = `sourceCard_${src.id}`;

      card.innerHTML = `
        <div class="source-header">
          <span class="source-id-badge">${src.id}</span>
          <span class="source-type-tag">${src.source_type || "Legal Authority"}</span>
        </div>
        <div class="source-title">${escapeHtml(src.title || "Statutory Provision")}</div>
        ${src.section ? `<div class="source-section">${escapeHtml(src.section)}</div>` : ""}
        <div class="source-excerpt">${escapeHtml(src.text)}</div>
      `;
      sourcesList.appendChild(card);
    });
  }

  function highlightSourceCard(sourceId) {
    const card = document.getElementById(`sourceCard_${sourceId}`);
    if (card) {
      card.scrollIntoView({ behavior: "smooth", block: "center" });
      card.classList.add("highlight");
      setTimeout(() => card.classList.remove("highlight"), 3500);
    }
  }

  // ============================================================
  // SESSION EVENT POLLING
  // ============================================================
  async function pollSession() {
    if (!activeSessionId) return;

    try {
      const res = await fetch(`/api/session/poll?id=${activeSessionId}`);
      if (!res.ok) return;
      const data = await res.json();

      if (data.status === "waiting_clarification") {
        const existingPrompt = document.getElementById("activeClarificationCard");
        if (!existingPrompt && data.clarification_question) {
          appendClarificationPrompt(data.clarification_question);
        }
      } else if (data.status === "completed") {
        if (data.sources && data.sources.length > 0) {
          renderSourcesList(data.sources);
        }
        if (data.final_answer) {
          renderFinalOpinion(data.final_answer);
        }
        finishSession();
        return;
      } else if (data.status === "failed") {
        removeActivity();
        const errCard = document.createElement("div");
        errCard.className = "message assistant";
        errCard.innerHTML = `
          <div class="bubble" style="border-color: var(--rose-500);">
            <strong style="color: var(--rose-500);">Notice</strong>
            <p>The system was unable to complete the analysis. Please check your query and try again.</p>
          </div>
        `;
        messageStream.appendChild(errCard);
        finishSession();
        return;
      }
    } catch (err) {
      console.warn("Poll error:", err);
    }

    pollTimer = setTimeout(pollSession, 1200);
  }

  function finishSession() {
    clearTimeout(pollTimer);
    pollTimer = null;
    isSubmitting = false;
    btnSubmit.disabled = false;
  }

  // ============================================================
  // FORM SUBMISSION
  // ============================================================
  caseForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const problem = caseInput.value.trim();
    if (!problem || isSubmitting) return;

    isSubmitting = true;
    btnSubmit.disabled = true;
    caseInput.value = "";

    appendUserMessage(problem);
    showActivity("Reviewing legal scenario against verified statutes and case precedents...");

    try {
      const res = await fetch("/api/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ problem })
      });

      const data = await res.json();
      if (!res.ok) {
        removeActivity();
        alert(data.error || "Failed to start consultation");
        finishSession();
        return;
      }

      activeSessionId = data.session_id;
      lastEventId = -1;
      pollSession();
    } catch (err) {
      removeActivity();
      alert("Network error connecting to consultation server");
      finishSession();
    }
  });

  caseInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      caseForm.dispatchEvent(new Event("submit"));
    }
  });

  btnReset.addEventListener("click", () => {
    if (pollTimer) clearTimeout(pollTimer);
    activeSessionId = null;
    lastEventId = -1;
    isSubmitting = false;
    btnSubmit.disabled = false;
    messageStream.innerHTML = "";
    emptyState.style.display = "flex";
    renderSourcesList([]);
    caseInput.value = "";
    caseInput.focus();
  });

  function scrollToBottom() {
    chatScroll.scrollTop = chatScroll.scrollHeight;
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  checkServerStatus();
})();
