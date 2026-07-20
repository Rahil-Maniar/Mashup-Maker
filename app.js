/* Mashup Studio web UI -> talks to the local Companion at 127.0.0.1:7777 */
const COMPANION = "http://127.0.0.1:7777";
let TOKEN = localStorage.getItem("companion_token") || "";
let SESSION = null;     // prepared session (songs/grids/lyrics)
let PLAN = null;        // last validated plan
const pair = { A: null, B: null };   // {path, title} per slot

const $ = (id) => document.getElementById(id);

// ---- auto-pair from URL fragment (companion opens us as .../#token=XYZ) ----
// The fragment never reaches any server; we store it and scrub the URL.
(() => {
  const m = location.hash.match(/token=([A-Za-z0-9_~.-]+)/);
  if (m) {
    TOKEN = m[1];
    localStorage.setItem("companion_token", TOKEN);
    history.replaceState(null, "", location.pathname + location.search);
  }
})();

// ---- low-level call helper ----
async function call(path, { method = "GET", body = null, auth = true } = {}) {
  const headers = {};
  if (auth && TOKEN) headers["X-Companion-Token"] = TOKEN;
  if (body) headers["Content-Type"] = "application/json";
  const res = await fetch(COMPANION + path, {
    method, headers, body: body ? JSON.stringify(body) : null,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// ---- live progress (polls the companion during long jobs) ----
let progTimer = null;
function startProgress(el, fallback) {
  stopProgress();
  el.textContent = fallback || "Working…";
  progTimer = setInterval(async () => {
    try {
      const p = await call("/progress");
      if (p.busy) {
        const step = p.steps ? ` (step ${p.step} of ${p.steps})` : "";
        el.textContent = `${p.detail || p.stage}${step}…`;
      }
    } catch { /* companion briefly busy; keep last message */ }
  }, 2000);
}
function stopProgress() { if (progTimer) { clearInterval(progTimer); progTimer = null; } }

// ---- GPU banner: detect always, install only on explicit click ----
function updateGpuBanner(g) {
  const b = $("gpuBanner");
  if (!b || !g) return;
  if (g.restart_needed) {
    b.classList.remove("hidden");
    b.innerHTML = `✅ GPU acceleration installed. <b>Close the helper window and start it again</b> to switch it on.`;
    return;
  }
  if (g.installing) {
    b.classList.remove("hidden");
    b.innerHTML = `⚡ Installing GPU acceleration — keep the helper window open…
      <div class="gpu-bar"><div class="gpu-bar-fill"></div></div>
      <span id="gpuDetail" class="muted">Starting download…</span>`;
    startGpuWatch();
    return;
  }
  if (g.error) {
    b.classList.remove("hidden");
    b.innerHTML = `<span class="muted">GPU setup didn't work on this machine — no problem, everything still runs on CPU.</span>`;
    return;
  }
  if (g.present && !g.accelerated) {
    b.classList.remove("hidden");
    b.innerHTML = `⚡ Your computer has a gaming graphics card — mashups can be <b>much faster</b>. ` +
      `<button id="gpuBtn">Enable (one-time ~5 GB download)</button>`;
    $("gpuBtn").onclick = async () => {
      $("gpuBtn").disabled = true;
      try {
        await call("/gpu/enable", { method: "POST", body: {} });
        updateGpuBanner({ installing: true });
      }
      catch (e) { b.innerHTML = `<span class="bad">${e.message}</span>`; }
    };
    return;
  }
  b.classList.add("hidden");
}

// While the GPU install runs, poll /progress for per-package detail and
// /health for state changes (done / failed). Stops itself when idle.
let gpuTimer = null;
function startGpuWatch() {
  if (gpuTimer) return;
  gpuTimer = setInterval(async () => {
    try {
      const p = await call("/progress");
      const el = $("gpuDetail");
      if (el && p.stage === "gpu" && p.detail) {
        const step = p.steps ? ` (step ${p.step} of ${p.steps})` : "";
        el.textContent = p.detail + step + "…";
      }
      if (p.stage !== "gpu") {           // finished or failed
        clearInterval(gpuTimer); gpuTimer = null;
        const h = await call("/health", { auth: false });
        updateGpuBanner(h.gpu);
      }
    } catch { /* companion busy; keep last message */ }
  }, 2000);
}

// ---- companion detection + pairing + consent ----
async function checkCompanion() {
  let h;
  try {
    h = await call("/health", { auth: false });
  } catch {
    setStatus(false, "Companion not running");
    show("setup");
    return;
  }
  if (!h.ok) { setStatus(false, "Companion error"); show("setup"); return; }
  if (h.terms) $("termsBox").textContent = h.terms;

  // verify our stored token still works (health's `authed` covers this too,
  // but /whoami is explicit and side-effect free)
  let authed = h.authed;
  if (!authed && TOKEN) {
    try { await call("/whoami"); authed = true; } catch { /* stale token */ }
  }
  if (!authed) {
    setStatus(true, "Companion found — pair below");
    show("setup");
    return;
  }
  if (!h.consented) {
    setStatus(true, "Paired — accept terms to continue");
    show("consent");
    return;
  }
  setStatus(true, "Companion paired");
  show("app");
  updateGpuBanner(h.gpu);
}

function setStatus(on, text) {
  $("dot").classList.toggle("on", on);
  $("statusText").textContent = text;
}

function show(which) {
  $("setupCard").classList.toggle("hidden", which !== "setup");
  $("consentCard").classList.toggle("hidden", which !== "consent");
  $("app").classList.toggle("hidden", which !== "app");
}

$("pairBtn").onclick = async () => {
  TOKEN = $("tokenInput").value.trim();
  if (!TOKEN) return;
  try {
    const w = await call("/whoami");      // token test, no side effects
    localStorage.setItem("companion_token", TOKEN);
    $("pairMsg").textContent = "Paired ✓";
    if (w.consented) { setStatus(true, "Companion paired"); show("app"); }
    else { setStatus(true, "Paired — accept terms to continue"); show("consent"); }
  } catch (e) {
    $("pairMsg").innerHTML = `<span class="bad">Pairing failed: ${e.message}</span>`;
  }
};

$("consentChk").onchange = () => { $("consentBtn").disabled = !$("consentChk").checked; };

$("consentBtn").onclick = async () => {
  $("consentBtn").disabled = true;
  try {
    await call("/consent", { method: "POST", body: { accept: true } });
    setStatus(true, "Companion paired");
    show("app");
  } catch (e) {
    $("consentMsg").innerHTML = `<span class="bad">${e.message}</span>`;
    $("consentBtn").disabled = false;
  }
};

// ---- recommended pairs ----
async function loadRecs() {
  const box = $("recsBox");
  $("recsBtn").disabled = true;
  $("recsMsg").textContent = "Finding good pairs… (first time loads the song database, give it a moment)";
  try {
    const { pairs } = await call("/recommend?n=4");
    box.innerHTML = pairs.length ? "" :
      `<div class="muted">No pairs found — is chart_tracks.csv / tracks.csv next to the helper?</div>`;
    pairs.forEach((p) => {
      const d = document.createElement("div");
      d.className = "res";
      d.innerHTML = `<b>${escapeHtml(p.name)}</b><br><span class="muted">${escapeHtml(p.description)} — click to load both songs</span>`;
      d.onclick = () => usePair(p, d);
      box.appendChild(d);
    });
    $("recsMsg").textContent = pairs.length ? "Click a pair, or roll again 🎲" : "";
  } catch (e) {
    box.innerHTML = "";
    $("recsMsg").innerHTML = `<span class="bad">${e.message}</span>`;
  } finally {
    $("recsBtn").disabled = false;
  }
}

async function usePair(p, el) {
  el.style.opacity = 0.5;
  $("recsMsg").textContent = "Fetching both songs…";
  $("prepBtn").disabled = true;
  try {
    for (const [slot, s] of [["A", p.song_a], ["B", p.song_b]]) {
      $("name" + slot).textContent = `Searching “${s.title}”…`;
      const { results } = await call("/search?q=" + encodeURIComponent(s.search_query));
      if (!results.length) throw new Error(`no YouTube result for “${s.title}”`);
      $("name" + slot).textContent = `Downloading “${s.title}”…`;
      const out = await call("/download", { method: "POST",
        body: { url: results[0].url, title: results[0].title } });
      pair[slot] = { path: out.path, title: out.title };
      $("name" + slot).textContent = out.title;
      $("slot" + slot).classList.add("filled");
    }
    $("recsMsg").textContent = "Pair loaded ✓ — hit Prepare pair";
  } catch (e) {
    $("recsMsg").innerHTML = `<span class="bad">${e.message}</span>`;
  } finally {
    el.style.opacity = 1;
    $("prepBtn").disabled = !(pair.A && pair.B);
  }
}

// ---- search + pick ----
async function doSearch(slot) {
  const q = $("q" + slot).value.trim();
  if (!q) return;
  const box = $("res" + slot);
  box.innerHTML = `<div class="muted">Searching…</div>`;
  try {
    const { results } = await call("/search?q=" + encodeURIComponent(q));
    box.innerHTML = "";
    results.slice(0, 6).forEach((r) => {
      const d = document.createElement("div");
      d.className = "res";
      const dur = r.duration ? `${Math.floor(r.duration/60)}:${String(r.duration%60).padStart(2,"0")}` : "?";
      d.innerHTML = `<b>${escapeHtml(r.title)}</b><br><span class="muted">${escapeHtml(r.uploader||"")} · ${dur}</span>`;
      d.onclick = () => download(slot, r);
      box.appendChild(d);
    });
  } catch (e) { box.innerHTML = `<span class="bad">${e.message}</span>`; }
}

async function download(slot, r) {
  const box = $("res" + slot);
  box.innerHTML = `<div class="muted">Downloading “${escapeHtml(r.title)}”…</div>`;
  try {
    const out = await call("/download", { method: "POST", body: { url: r.url, title: r.title } });
    pair[slot] = { path: out.path, title: out.title };
    $("name" + slot).textContent = out.title;
    $("slot" + slot).classList.add("filled");
    box.innerHTML = "";
    $("prepBtn").disabled = !(pair.A && pair.B);
  } catch (e) { box.innerHTML = `<span class="bad">${e.message}</span>`; }
}

// ---- prepare ----
async function doPrepare() {
  $("prepBtn").disabled = true;
  startProgress($("prepMsg"), "Starting up on your machine…");
  try {
    const out = await call("/prepare", { method: "POST", body: {
      path_a: pair.A.path, path_b: pair.B.path, name_a: pair.A.title, name_b: pair.B.title } });
    SESSION = out.session;
    renderBlend(out.blend);
    $("promptBox").textContent = out.prompt;
    $("djCard").classList.remove("hidden");
    stopProgress();
    $("prepMsg").textContent = "Ready ✓";
  } catch (e) {
    stopProgress();
    $("prepMsg").innerHTML = `<span class="bad">${e.message}</span>`;
  } finally {
    $("prepBtn").disabled = false;
  }
}

function renderBlend(b) {
  $("blendCard").classList.remove("hidden");
  $("blendScore").textContent = b.score + "/100";
  $("blendVerdict").textContent = b.verdict;
  $("blendChecks").innerHTML = b.checks.map(([name, lvl, msg]) => {
    const icon = { ok: "✅", warn: "⚠️", bad: "❌" }[lvl];
    return `<div class="chk"><span>${icon}</span><span><b>${name}:</b> ${escapeHtml(msg)}</span></div>`;
  }).join("");
}

// ---- brief ----
async function applyBrief() {
  const brief = {};
  const t = $("briefText").value.trim(); if (t) brief.text = t;
  const a = $("arcSel").value; if (a) brief.arc = a;
  try {
    const out = await call("/reprompt", { method: "POST", body: { session: SESSION, brief } });
    $("promptBox").textContent = out.prompt;
    $("copyMsg").textContent = "Brief applied — prompt updated.";
  } catch (e) { $("copyMsg").innerHTML = `<span class="bad">${e.message}</span>`; }
}

function copyPrompt() {
  navigator.clipboard.writeText($("promptBox").textContent);
  $("copyMsg").textContent = "Copied! Paste into your LLM.";
}

// ---- validate ----
async function doValidate() {
  $("djMsg").textContent = "Validating…";
  $("validOut").innerHTML = "";
  try {
    const out = await call("/validate", { method: "POST", body: {
      session: SESSION, plan_text: $("planText").value } });
    if (out.errors.length) {
      $("validOut").innerHTML = `<div class="bad">Plan rejected — paste these back to your LLM:</div>` +
        `<pre>${out.errors.map(e => "- " + escapeHtml(e)).join("\n")}</pre>`;
      $("renderBtn").disabled = true;
      $("djMsg").textContent = "";
      return;
    }
    PLAN = out.plan;
    let html = `<div class="ok">Plan valid ✓ ${escapeHtml(out.plan.comment || "")}</div>`;
    if (out.warnings.length) html += `<pre>${out.warnings.map(w => "⚠ " + escapeHtml(w)).join("\n")}</pre>`;
    if (out.phrase_issues.length)
      html += `<div class="warn">Clips lyric phrases (fixable, or render anyway):</div>` +
              `<pre>${out.phrase_issues.map(p => "- " + escapeHtml(p)).join("\n")}</pre>`;
    $("validOut").innerHTML = html;
    $("renderBtn").disabled = false;
    $("djMsg").textContent = "";
  } catch (e) {
    $("validOut").innerHTML = `<span class="bad">Couldn't read a plan: ${e.message}</span>`;
    $("djMsg").textContent = "";
  }
}

// ---- render ----
async function showResult(out) {
  const url = COMPANION + "/audio?file=" + encodeURIComponent(out.file);
  // fetch with token, turn into a blob the <audio> and download link can use
  const blob = await (await fetch(url, { headers: { "X-Companion-Token": TOKEN } })).blob();
  const objURL = URL.createObjectURL(blob);
  $("player").src = objURL;
  $("dlLink").href = objURL;
  $("outCard").classList.remove("hidden");
  $("criticBox").textContent = out.critic;
  $("outCard").scrollIntoView({ behavior: "smooth" });
}

async function doRender() {
  $("renderBtn").disabled = true;
  startProgress($("djMsg"), "Rendering on your machine…");
  try {
    const out = await call("/render", { method: "POST", body: { session: SESSION, plan: PLAN } });
    await showResult(out);
    stopProgress();
    $("djMsg").textContent = "Done ✓";
  } catch (e) {
    stopProgress();
    $("djMsg").innerHTML = `<span class="bad">${e.message}</span>`;
  } finally {
    $("renderBtn").disabled = false;
  }
}

// ---- one-click auto mashup (no LLM round-trip needed) ----
async function autoMashup() {
  if (!SESSION) return;
  $("autoBtn").disabled = true;
  startProgress($("autoMsg"), "Designing an arrangement…");
  try {
    const out = await call("/auto_mashup", { method: "POST", body: { session: SESSION } });
    await showResult(out);
    stopProgress();
    $("autoMsg").textContent = "Done ✓ — want more control? Use the AI DJ below.";
  } catch (e) {
    stopProgress();
    $("autoMsg").innerHTML = `<span class="bad">${e.message}</span>`;
  } finally {
    $("autoBtn").disabled = false;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => (
    { "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c]));
}

// expose handlers used by inline onclick
window.doSearch = doSearch; window.doPrepare = doPrepare; window.applyBrief = applyBrief;
window.copyPrompt = copyPrompt; window.doValidate = doValidate; window.doRender = doRender;
window.loadRecs = loadRecs; window.autoMashup = autoMashup;

checkCompanion();
setInterval(checkCompanion, 5000);   // keep the status dot live
