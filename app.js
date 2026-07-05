/* Mashup Studio web UI -> talks to the local Companion at 127.0.0.1:7777 */
const COMPANION = "http://127.0.0.1:7777";
let TOKEN = localStorage.getItem("companion_token") || "";
let SESSION = null;     // prepared session (songs/grids/lyrics)
let PLAN = null;        // last validated plan
const pair = { A: null, B: null };   // {path, title} per slot

const $ = (id) => document.getElementById(id);

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

// ---- companion detection + pairing ----
async function checkCompanion() {
  try {
    const h = await call("/health", { auth: false });
    if (h.ok) {
      setStatus(true, h.authed ? "Companion paired" : "Companion found — pair below");
      if (h.authed) { showApp(); return; }
    }
  } catch {
    setStatus(false, "Companion not running");
  }
  // if we have a stored token, verify it actually works
  if (TOKEN) {
    try { await call("/search?q=ping"); showApp(); setStatus(true, "Companion paired"); }
    catch { /* token stale; stay on setup */ }
  }
}

function setStatus(on, text) {
  $("dot").classList.toggle("on", on);
  $("statusText").textContent = text;
}
function showApp() { $("setupCard").classList.add("hidden"); $("app").classList.remove("hidden"); }

$("pairBtn").onclick = async () => {
  TOKEN = $("tokenInput").value.trim();
  if (!TOKEN) return;
  try {
    await call("/search?q=ping");         // token test
    localStorage.setItem("companion_token", TOKEN);
    $("pairMsg").textContent = "Paired ✓";
    setStatus(true, "Companion paired");
    showApp();
  } catch (e) {
    $("pairMsg").innerHTML = `<span class="bad">Pairing failed: ${e.message}</span>`;
  }
};

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
  $("prepMsg").textContent = "Separating + analyzing (this runs on your machine)…";
  try {
    const out = await call("/prepare", { method: "POST", body: {
      path_a: pair.A.path, path_b: pair.B.path, name_a: pair.A.title, name_b: pair.B.title } });
    SESSION = out.session;
    renderBlend(out.blend);
    $("promptBox").textContent = out.prompt;
    $("djCard").classList.remove("hidden");
    $("prepMsg").textContent = "Ready ✓";
  } catch (e) {
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
async function doRender() {
  $("renderBtn").disabled = true;
  $("djMsg").textContent = "Rendering on your machine…";
  try {
    const out = await call("/render", { method: "POST", body: { session: SESSION, plan: PLAN } });
    const url = COMPANION + "/audio?file=" + encodeURIComponent(out.file);
    // fetch with token, turn into a blob the <audio> and download link can use
    const blob = await (await fetch(url, { headers: { "X-Companion-Token": TOKEN } })).blob();
    const objURL = URL.createObjectURL(blob);
    $("player").src = objURL;
    $("dlLink").href = objURL;
    $("outCard").classList.remove("hidden");
    $("criticBox").textContent = out.critic;
    $("outCard").scrollIntoView({ behavior: "smooth" });
    $("djMsg").textContent = "Done ✓";
  } catch (e) {
    $("djMsg").innerHTML = `<span class="bad">${e.message}</span>`;
  } finally {
    $("renderBtn").disabled = false;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => (
    { "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c]));
}

// expose handlers used by inline onclick
window.doSearch = doSearch; window.doPrepare = doPrepare; window.applyBrief = applyBrief;
window.copyPrompt = copyPrompt; window.doValidate = doValidate; window.doRender = doRender;

checkCompanion();
setInterval(checkCompanion, 5000);   // keep the status dot live
