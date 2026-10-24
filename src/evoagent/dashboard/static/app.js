/**
 * EvoAgent Interactive Dashboard Client
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  fetchMetrics();
  fetchEpisodes();
  fetchDataset();
  fetchExperiments();

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

    document.getElementById("metric-resolution").innerText = formatMetric(data.resolution_rate_percent, "%");
    document.getElementById("metric-regression").innerText = formatMetric(data.regression_free_rate_percent, "%");
    document.getElementById("metric-validity").innerText = formatMetric(data.patch_validity_rate_percent, "%");
    document.getElementById("metric-duration").innerText = formatMetric(data.average_duration_seconds, "s");
  } catch (err) {
    console.error("Failed to fetch metrics:", err);
  }
}

// Metrics the server could not measure come back as null
function formatMetric(value, suffix) {
  return value === null || value === undefined ? "n/a" : `${value}${suffix}`;
}

// Fetch held-out ablation results written by benchmarks/run_benchmarks.py --eval-comparison
async function fetchExperiments() {
  try {
    const res = await fetch("/api/experiments");
    if (!res.ok) return;
    const data = await res.json();
    if (!data.conditions || data.conditions.length === 0) return;

    const meta = document.getElementById("experiments-meta");
    if (meta) meta.innerText = `${data.model} • ${data.generated_at.slice(0, 10)}`;

    const tbody = document.getElementById("experiments-table-body");
    tbody.innerHTML = "";
    data.conditions.forEach((cond) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-family: var(--font-mono); font-weight: 600; color: #818cf8;">${escapeHtml(cond.id)}</td>
        <td>${escapeHtml(cond.name)}</td>
        <td>${formatMetric(cond.resolution_rate_percent, "%")}</td>
        <td>${formatMetric(cond.regression_free_rate_percent, "%")}</td>
        <td>${formatMetric(cond.average_attempts_per_task, "")}</td>
        <td>${formatMetric(cond.average_duration_seconds, "s")}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to fetch experiments:", err);
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

// Trigger autonomous task execution and print the attempts the server actually ran
async function runAutonomousTask() {
  const btn = document.getElementById("run-task-btn");
  const term = document.getElementById("terminal-output");
  const diffViewer = document.getElementById("active-diff-viewer");

  btn.disabled = true;
  btn.innerHTML = `<span class="pulse-dot"></span> Executing Repair Loop...`;

  term.innerHTML = `[+] Running repair loop for bug-001-parser in a throwaway workspace...\n`;
  ["node-queued", "node-analyzing", "node-planned", "node-patching", "node-evaluating", "node-succeeded", "node-completed"]
    .forEach((id) => setNodeState(id, ""));
  setNodeState("node-queued", "completed");
  setNodeState("node-analyzing", "active");

  try {
    const res = await fetch("/api/run-task", { method: "POST" });
    const data = await res.json();

    (data.attempts || []).forEach((att) => {
      appendTerminal(`[attempt ${att.attempt_number}] ${att.state}: ${att.hypothesis}`);
      if (att.feedback) appendTerminal(`    ${att.feedback}`);
    });

    ["node-analyzing", "node-planned", "node-patching", "node-evaluating"].forEach((id) => setNodeState(id, "completed"));
    if (data.is_successful) {
      setNodeState("node-succeeded", "completed");
      setNodeState("node-completed", "completed");
      appendTerminal(`[SUCCESS] Task ${data.task_id} reached ${data.final_state} in ${data.total_duration_seconds}s; episode saved to memory.`);

      if (data.accepted_patch && diffViewer) {
        diffViewer.innerHTML = renderDiffHtml(data.accepted_patch.diff_unified);
      }
    } else {
      appendTerminal(`[FAIL] Task ended in ${data.final_state || "error"}: ${data.message || "no patch passed the evaluator"}`);
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
