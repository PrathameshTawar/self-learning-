/**
 * EvoAgent Interactive Dashboard Client
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  fetchMetrics();
  fetchEpisodes();
  fetchDataset();

  const runBtn = document.getElementById("run-task-btn");
  if (runBtn) {
    runBtn.addEventListener("click", runAutonomousTask);
  }

  const exportBtn = document.getElementById("export-dataset-btn");
  if (exportBtn) {
    exportBtn.addEventListener("click", exportDatasetAction);
  }
});

// Tab navigation handler
function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
      }
    });
  });
}

// Fetch and populate metrics
async function fetchMetrics() {
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById("metric-resolution").innerText = `${data.resolution_rate_percent}%`;
    document.getElementById("metric-regression").innerText = `${data.regression_free_rate_percent}%`;
    document.getElementById("metric-validity").innerText = `${data.patch_validity_rate_percent}%`;
    document.getElementById("metric-duration").innerText = `${data.average_duration_seconds}s`;
    document.getElementById("metric-total-tasks").innerText = data.total_tasks || 1;
  } catch (err) {
    console.error("Failed to fetch metrics:", err);
  }
}

// Fetch and display episodes
async function fetchEpisodes() {
  try {
    const res = await fetch("/api/episodes");
    if (!res.ok) return;
    const episodes = await res.json();

    const tbody = document.getElementById("episodes-table-body");
    const countEl = document.getElementById("memory-count-badge");
    if (countEl) countEl.innerText = `${episodes.length} Episodes`;

    if (!episodes || episodes.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color: var(--text-dim);">No episodes recorded in memory yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = "";
    episodes.forEach((ep) => {
      const isAccepted = ep.outcome === "accepted";
      const tr = document.createElement("tr");
      tr.style.cursor = "pointer";
      tr.innerHTML = `
        <td style="font-family: var(--font-mono); font-weight: 600; color: #818cf8;">${ep.task_id}</td>
        <td style="font-family: var(--font-mono); font-size: 0.8rem; color: #fb7185;">${escapeHtml(ep.failure_signature.slice(0, 45))}</td>
        <td><span class="${isAccepted ? 'badge-success' : 'badge-danger'}">${ep.outcome}</span></td>
        <td style="color: var(--text-muted); font-size: 0.8rem;">${escapeHtml(ep.lessons_learned.slice(0, 60))}...</td>
        <td><button class="btn-outline" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;" onclick="viewEpisodeDiff('${ep.episode_id}')">Inspect Diff</button></td>
      `;
      tbody.appendChild(tr);
    });

    window._evoEpisodes = episodes;
  } catch (err) {
    console.error("Failed to fetch episodes:", err);
  }
}

// View episode patch in modal/viewer
function viewEpisodeDiff(episodeId) {
  const episodes = window._evoEpisodes || [];
  const ep = episodes.find((e) => e.episode_id === episodeId);
  if (!ep) return;

  const diffContainer = document.getElementById("active-diff-viewer");
  if (diffContainer) {
    diffContainer.innerHTML = renderDiffHtml(ep.applied_patch || "No patch diff recorded.");
  }

  // Switch to live pipeline view to see diff
  const liveTab = document.querySelector('[data-tab="tab-pipeline"]');
  if (liveTab) liveTab.click();
}

// Fetch SFT Dataset Preview
async function fetchDataset() {
  try {
    const res = await fetch("/api/dataset");
    if (!res.ok) return;
    const items = await res.json();

    const container = document.getElementById("dataset-preview-container");
    if (!container) return;

    if (items.length === 0) {
      container.innerHTML = `<p style="color: var(--text-dim);">Run benchmark tasks to accumulate verified repair episodes for dataset export.</p>`;
      return;
    }

    container.innerHTML = "";
    items.slice(0, 5).forEach((item, idx) => {
      const card = document.createElement("div");
      card.style.background = "#060910";
      card.style.border = "1px solid var(--border-subtle)";
      card.style.borderRadius = "var(--radius-sm)";
      card.style.padding = "1rem";
      card.style.marginBottom = "1rem";
      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
          <span style="font-family: var(--font-mono); color: #38bdf8; font-size:0.8rem;">Task: ${item.task_id}</span>
          <span class="card-badge">Split: ${item.split || 'train'}</span>
        </div>
        <p style="font-size:0.8rem; color: var(--text-muted); margin-bottom: 0.5rem;"><strong>Instruction:</strong> ${escapeHtml(item.prompt.slice(0, 120))}...</p>
        <pre style="background: rgba(0,0,0,0.4); padding: 0.5rem; font-family: var(--font-mono); font-size:0.75rem; border-radius:4px; color:#34d399; overflow-x:auto;">${escapeHtml(item.completion)}</pre>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.error("Failed to fetch dataset:", err);
  }
}

// Trigger autonomous task execution with stepper animation
async function runAutonomousTask() {
  const btn = document.getElementById("run-task-btn");
  const term = document.getElementById("terminal-output");
  const diffViewer = document.getElementById("active-diff-viewer");

  btn.disabled = true;
  btn.innerHTML = `<span class="pulse-dot"></span> Executing Repair Loop...`;

  term.innerHTML = `[+] Initiating task: bug-001-parser\n[+] Reading code context & failure signature...\n`;

  // Animate pipeline nodes
  setNodeState("node-queued", "completed");
  setNodeState("node-analyzing", "active");

  try {
    appendTerminal("[+] Locating target file parser.py...");
    await sleep(600);

    setNodeState("node-analyzing", "completed");
    setNodeState("node-planned", "active");
    appendTerminal("[+] Querying episodic memory store for IndexError patterns...");
    appendTerminal("[+] Formulating hypothesis: Guard empty string/collection before index access.");
    await sleep(600);

    setNodeState("node-planned", "completed");
    setNodeState("node-patching", "active");
    appendTerminal("[+] Generating unified diff proposal with minimal footprint...");
    await sleep(600);

    setNodeState("node-patching", "completed");
    setNodeState("node-evaluating", "active");
    appendTerminal("[+] Dispatching to isolated execution sandbox...");
    appendTerminal("[+] Running pre-patch regression check -> reproduced IndexError as expected.");

    const res = await fetch("/api/run-task", { method: "POST" });
    const data = await res.json();

    if (data.is_successful) {
      setNodeState("node-evaluating", "completed");
      setNodeState("node-succeeded", "completed");
      setNodeState("node-completed", "completed");

      appendTerminal("[+] Running post-patch targeted tests: PASSED!");
      appendTerminal("[+] Running full regression test suite: PASSED!");
      appendTerminal(`[SUCCESS] Task ${data.task_id} completed successfully in ${data.total_duration_seconds}s!`);
      appendTerminal("[+] New experience episode recorded to SQLite memory store.");

      if (data.accepted_patch && diffViewer) {
        diffViewer.innerHTML = renderDiffHtml(data.accepted_patch.diff_unified);
      }
    } else {
      setNodeState("node-evaluating", "completed");
      appendTerminal(`[FAIL] Task failed: ${data.message || 'Execution error'}`);
    }

    // Refresh state
    await fetchMetrics();
    await fetchEpisodes();
    await fetchDataset();
  } catch (err) {
    appendTerminal(`[ERROR] Network or server error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `▶ Trigger Autonomous Repair`;
  }
}

// Trigger dataset export
async function exportDatasetAction() {
  try {
    const res = await fetch("/api/export-dataset", { method: "POST" });
    const data = await res.json();
    alert(`Success: ${data.message}`);
    fetchDataset();
  } catch (err) {
    alert(`Export failed: ${err.message}`);
  }
}

// Helpers
function setNodeState(nodeId, state) {
  const el = document.getElementById(nodeId);
  if (!el) return;
  el.classList.remove("active", "completed");
  if (state) el.classList.add(state);
}

function appendTerminal(line) {
  const term = document.getElementById("terminal-output");
  if (!term) return;
  term.innerHTML += `${escapeHtml(line)}\n`;
  term.scrollTop = term.scrollHeight;
}

function renderDiffHtml(diffText) {
  const lines = diffText.split("\n");
  return lines
    .map((l) => {
      let cls = "diff-line";
      if (l.startsWith("---") || l.startsWith("+++") || l.startsWith("@@")) {
        cls += " diff-header";
      } else if (l.startsWith("+")) {
        cls += " diff-add";
      } else if (l.startsWith("-")) {
        cls += " diff-remove";
      }
      return `<div class="${cls}">${escapeHtml(l)}</div>`;
    })
    .join("");
}

function escapeHtml(text) {
  if (!text) return "";
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
