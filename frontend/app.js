"use strict";

const API = (window.APP_CONFIG && window.APP_CONFIG.API_BASE_URL) || "";
const $ = (id) => document.getElementById(id);

const SAMPLES = {
  article: `City Council Approves Solar Microgrid Plan for Public Schools

The city council voted 7-2 on Tuesday to install solar microgrids at 40 public schools, a plan officials say will cut the district's electricity costs by roughly 30 percent. The $48.5 million project will be funded through a mix of state clean-energy grants and municipal green bonds.

Each microgrid combines rooftop solar panels with battery storage, allowing schools to keep running during power outages. Dr. Elena Ruiz, who leads the district's sustainability office, said the batteries would let the buildings serve as emergency shelters during heat waves and storms. "When the grid goes down, these schools will stay lit," she said.

Construction is expected to begin next spring and finish within three years. The first 12 schools were chosen because they have large, south-facing roofs and the highest energy bills in the district.

Not everyone supported the plan. Two council members argued that the money would be better spent on teacher salaries and building repairs. Council member James T. Walsh said the projected savings depend on optimistic assumptions about battery lifespan and future electricity prices.

Supporters countered that the solar panels carry a 25-year warranty and that the savings will be reinvested in classrooms. An independent audit commissioned by the council estimated that the system would pay for itself in about 11 years.`,
  chat: `Sam: Hey, are you coming to Mom's birthday dinner on Saturday?
Lucy: Of course! What time?
Sam: 7 pm at the Italian place near the station.
Lucy: Should I bring anything?
Sam: Could you pick up the cake? I'll get the flowers.
Lucy: Sure, I'll get a chocolate one.`,
};

const MODE_LABELS = { article: "Article summary", dialogue: "Chat summary" };

let lastResult = null;

// ---------------------------------------------------------------------------- API
async function api(path, options = {}) {
  const response = await fetch(API + path, options);
  let body = null;
  try {
    body = await response.json();
  } catch {
    /* non-JSON error page */
  }
  if (!response.ok) {
    const detail = body && body.detail;
    const message = Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail;
    throw new Error(message || `Request failed (${response.status})`);
  }
  return body;
}

const postJSON = (path, payload) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });

// --------------------------------------------------------------------------- Boot
async function checkHealth() {
  const status = $("status");
  const slow = setTimeout(() => {
    showBanner("Waking up the server… this can take up to a minute after a period of inactivity.");
  }, 2500);
  try {
    await api("/health");
    status.textContent = "online";
    status.className = "status status-ok";
    hideBanner();
  } catch {
    status.textContent = "offline";
    status.className = "status status-down";
    showBanner("The summarization server is unreachable. Please try again in a moment.");
  } finally {
    clearTimeout(slow);
  }
}

function showBanner(text) {
  $("banner").textContent = text;
  $("banner").hidden = false;
}
function hideBanner() {
  $("banner").hidden = true;
}

// ------------------------------------------------------------------------ Helpers
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function setBusy(busy, text = "Summarizing…") {
  for (const id of ["summarize", "fetch-url"]) $(id).disabled = busy;
  $("loading").hidden = !busy;
  $("loading-text").textContent = text;
  if (busy) {
    $("placeholder").hidden = true;
    $("result").hidden = true;
  }
}

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}

function updateCharCount() {
  $("char-count").textContent = `${$("text").value.length.toLocaleString()} characters`;
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.hidden = panel.dataset.panel !== name;
  });
}

const pct = (x) => `${Math.round(x * 100)}%`;

// ------------------------------------------------------------------------- Actions
async function summarize() {
  showError("");
  if (!$("text").value.trim()) return showError("Paste some text first, or load a sample.");
  setBusy(true, "Summarizing…");
  try {
    const payload = {
      text: $("text").value,
      mode: $("mode").value,
      num_sentences: parseInt($("length").value, 10),
    };
    lastResult = await postJSON("/api/v1/summarize", payload);
    renderResult(lastResult);
  } catch (err) {
    showError(err.message);
    $("placeholder").hidden = false;
  } finally {
    setBusy(false);
  }
}

async function loadExtracted(promise) {
  showError("");
  setBusy(true, "Reading…");
  try {
    const data = await promise;
    $("text").value = data.text;
    updateCharCount();
    switchTab("text");
    if (data.truncated) showError(`Long document: only the first ${data.chars.toLocaleString()} characters were kept.`);
  } catch (err) {
    showError(err.message);
  } finally {
    setBusy(false);
    $("placeholder").hidden = false;
  }
}

function fetchUrl() {
  const url = $("url").value.trim();
  if (!url) return showError("Enter a URL.");
  loadExtracted(postJSON("/api/v1/extract/url", { url }));
}

function uploadFile() {
  const file = $("file").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  loadExtracted(api("/api/v1/extract/file", { method: "POST", body: form }));
}

// ------------------------------------------------------------------------- Render
function renderResult(r) {
  $("result").hidden = false;
  $("badge-mode").textContent = MODE_LABELS[r.mode] || "Summary";
  $("summary").textContent = r.summary.summary || "(empty summary)";

  const warnings = $("warnings");
  warnings.textContent = "";
  for (const w of r.warnings || []) warnings.appendChild(el("li", null, w));

  const s = r.stats;
  const ss = r.summary_stats;
  const metrics = [
    [`${s.words.toLocaleString()} → ${ss.words.toLocaleString()}`, "words"],
    [pct(r.compression_ratio), "of original length"],
    [formatDuration(Math.max(0, s.reading_time_seconds - ss.reading_time_seconds)), "reading time saved"],
  ];
  const grid = $("metrics");
  grid.textContent = "";
  for (const [value, label] of metrics) {
    const card = el("div", "metric");
    card.append(el("div", "value", value), el("div", "label", label));
    grid.appendChild(card);
  }

  renderChips($("keywords"), r.keywords);
  renderChips($("phrases"), r.key_phrases);
}

function formatDuration(seconds) {
  if (seconds < 60) return `${seconds} s`;
  const minutes = Math.round(seconds / 60);
  return `${minutes} min`;
}

function renderChips(container, items) {
  container.textContent = "";
  if (!items || !items.length) container.appendChild(el("span", "muted small", "—"));
  for (const k of items || []) container.appendChild(el("span", "chip", k.text));
}

// ------------------------------------------------------------------------- Export
function toMarkdown(r) {
  const lines = ["# Summary", "", r.summary.summary, ""];
  lines.push(`- **Length:** ${r.stats.words} → ${r.summary_stats.words} words (${pct(r.compression_ratio)})`);
  if (r.keywords.length) lines.push(`- **Keywords:** ${r.keywords.map((k) => k.text).join(", ")}`);
  if (r.key_phrases.length) lines.push(`- **Key phrases:** ${r.key_phrases.map((k) => k.text).join(", ")}`);
  return lines.join("\n") + "\n";
}

function download(filename, content) {
  const url = URL.createObjectURL(new Blob([content], { type: "text/markdown;charset=utf-8" }));
  const link = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

// -------------------------------------------------------------------------- Wire up
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => switchTab(tab.dataset.tab)));
$("text").addEventListener("input", updateCharCount);
$("sample-article").addEventListener("click", () => {
  $("text").value = SAMPLES.article;
  updateCharCount();
});
$("sample-chat").addEventListener("click", () => {
  $("text").value = SAMPLES.chat;
  updateCharCount();
});
$("clear").addEventListener("click", () => {
  $("text").value = "";
  updateCharCount();
});
$("summarize").addEventListener("click", summarize);
$("fetch-url").addEventListener("click", fetchUrl);
$("url").addEventListener("keydown", (e) => e.key === "Enter" && fetchUrl());
$("file").addEventListener("change", uploadFile);
$("text").addEventListener("keydown", (e) => (e.ctrlKey || e.metaKey) && e.key === "Enter" && summarize());
$("copy").addEventListener("click", async () => {
  if (!lastResult) return;
  await navigator.clipboard.writeText(lastResult.summary.summary);
  $("copy").textContent = "Copied";
  setTimeout(() => ($("copy").textContent = "Copy"), 1200);
});
$("download").addEventListener("click", () => lastResult && download("summary.md", toMarkdown(lastResult)));
$("print").addEventListener("click", () => window.print());

updateCharCount();
checkHealth();
