"""
companion.py -- The Mashup Studio Companion.

A tiny LOCAL server the user runs on their own machine. The hosted web UI
(static site) talks to it at http://127.0.0.1:7777. Everything heavy happens
HERE, on the user's device and IP: YouTube search/download, stem separation
(their GPU), analysis, lyrics, rendering.

Design principles:
- stdlib only (http.server) -- users need no pip installs beyond the project's
  existing requirements.
- 127.0.0.1 bind ONLY. Never reachable from the network.
- Pairing token: every request must carry X-Companion-Token. The token is
  printed at startup; the web UI asks for it once and stores it.
- Origin allowlist: only the configured web-app origins may call this server.
- First-run consent gate: refuses to serve until the user has explicitly
  accepted the terms (stored in companion_consent.json).

Run:  python companion.py
"""

import os
import json
import time
import secrets
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

PORT = 7777
WORK_DIR = "downloads"
CONSENT_FILE = "companion_consent.json"
TOKEN_FILE = "companion_token.txt"

# Origins allowed to talk to this Companion. Add your hosted URL here.
ALLOWED_ORIGINS = {
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
    # "https://your-app.pages.dev",   # <- add your hosted UI origin
}

CONSENT_TEXT = """
=== MASHUP STUDIO COMPANION - PLEASE READ ===

This program runs ON YOUR COMPUTER and, when you use the web app, will:
  1. Search YouTube and DOWNLOAD audio to this machine, over YOUR internet
     connection, at your request.
  2. Process that audio locally (stem separation, analysis, mashup rendering).

Downloading may violate YouTube's Terms of Service and, depending on your
country, copyright law. Outputs are for PERSONAL, non-commercial use.
You are responsible for your own use of this tool. Nothing is uploaded
anywhere; all audio stays on this machine.

Type 'I AGREE' to accept and start the Companion, anything else to quit.
"""

# one heavy job at a time (single GPU)
JOB_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# consent + token
# ---------------------------------------------------------------------------
def ensure_consent():
    if os.path.exists(CONSENT_FILE):
        return True
    print(CONSENT_TEXT)
    if input("> ").strip().upper() == "I AGREE":
        with open(CONSENT_FILE, "w") as f:
            json.dump({"accepted_at": time.time()}, f)
        return True
    return False


def get_token():
    if os.path.exists(TOKEN_FILE):
        return open(TOKEN_FILE).read().strip()
    tok = secrets.token_urlsafe(24)
    with open(TOKEN_FILE, "w") as f:
        f.write(tok)
    return tok


TOKEN = None  # set in main()


# ---------------------------------------------------------------------------
# pipeline operations (lazy imports so startup is instant)
# ---------------------------------------------------------------------------
def op_search(q):
    from get_song import search
    return {"results": search(q + " audio")}


def op_download(url, title):
    from get_song import download
    path = download(url, name_hint=title)
    if not path:
        raise RuntimeError("download produced no file")
    return {"path": path, "title": title}


def op_prepare(path_a, path_b, name_a, name_b):
    from llm_dj import analyze_song, compute_shifts, bar_grid, \
        transcribe_lyrics, lyrics_to_bar_lines, build_prompt
    from mashup_maker import separate_full_song
    from studio_logic import blend_report, file_hash

    if file_hash(path_a) == file_hash(path_b):
        raise ValueError("Both slots contain the same audio file.")

    songs, grids, lyrics = {}, {}, {}
    names = {"A": name_a, "B": name_b}
    for sid, path in (("A", path_a), ("B", path_b)):
        meta = analyze_song(path)
        stems = separate_full_song(path, WORK_DIR)
        if not stems:
            raise RuntimeError(f"separation failed for {names[sid]}")
        songs[sid] = {"stems": stems, "bpm": meta["bpm"],
                      "grid_start": meta["anchor"] % (4 * 60.0 / meta["bpm"]),
                      "key": meta["key"], "shift": 0}
    sa, sb = compute_shifts(songs["A"]["key"], songs["B"]["key"])
    songs["A"]["shift"], songs["B"]["shift"] = sa, sb
    for sid in ("A", "B"):
        grids[sid] = bar_grid(songs[sid])
        segs = transcribe_lyrics(songs[sid]["stems"]["vocals"])
        lyrics[sid] = lyrics_to_bar_lines(segs, songs[sid]["bpm"], songs[sid]["grid_start"])

    session = {"songs": songs, "names": names, "grids": grids, "lyrics": lyrics}
    with open("dj_session.json", "w") as f:
        json.dump(session, f, indent=2)
    rep = blend_report(songs["A"]["key"], songs["B"]["key"],
                       songs["A"]["bpm"], songs["B"]["bpm"], songs["A"]["shift"])
    return {"session": session, "blend": rep,
            "prompt": build_prompt(songs, grids, names, lyrics)}


def op_reprompt(session, brief):
    from llm_dj import build_prompt
    session["brief"] = brief or {}
    with open("dj_session.json", "w") as f:
        json.dump(session, f, indent=2)
    return {"prompt": build_prompt(session["songs"], session["grids"],
                                   session["names"], session["lyrics"],
                                   session.get("brief"))}


def op_validate(session, plan_text):
    from llm_dj import extract_plan_json
    from dsl_renderer import validate_plan
    from critic import check_phrase_integrity
    plan = extract_plan_json(plan_text)
    errors, warnings = validate_plan(plan, session["songs"])
    phrases = check_phrase_integrity(plan, session) if not errors else []
    return {"plan": plan, "errors": errors, "warnings": warnings,
            "phrase_issues": phrases}


def op_render(session, plan):
    from dsl_renderer import render_plan
    from critic import critique_render
    out, report = render_plan(plan, session["songs"], "llm_mashup.wav")
    res = critique_render(out, plan, session)
    return {"file": out, "render_report": report, "critic": res["report"],
            "flags": res["flags"]}


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ---- helpers ----
    def _origin_ok(self):
        origin = self.headers.get("Origin")
        return origin is None or origin in ALLOWED_ORIGINS  # None = curl/local tools

    def _cors(self):
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, X-Companion-Token")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        # Chrome Private Network Access preflight
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self):
        return self.headers.get("X-Companion-Token") == TOKEN

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def log_message(self, fmt, *args):  # quieter console
        pass

    # ---- routing ----
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if not self._origin_ok():
            return self._json(403, {"error": "origin not allowed"})
        url = urlparse(self.path)

        if url.path == "/health":
            return self._json(200, {"ok": True, "name": "mashup-companion",
                                    "authed": self._authed()})

        if not self._authed():
            return self._json(401, {"error": "missing/invalid X-Companion-Token"})

        if url.path == "/search":
            q = (parse_qs(url.query).get("q") or [""])[0]
            if not q:
                return self._json(400, {"error": "q required"})
            return self._run(op_search, q)

        if url.path == "/audio":
            f = (parse_qs(url.query).get("file") or [""])[0]
            safe = os.path.realpath(f)
            roots = (os.path.realpath(WORK_DIR), os.path.realpath("."))
            if not any(safe.startswith(r + os.sep) or safe == r for r in roots) \
                    or not safe.endswith(".wav") or not os.path.exists(safe):
                return self._json(404, {"error": "not found"})
            with open(safe, "rb") as fh:
                data = fh.read()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        return self._json(404, {"error": "unknown endpoint"})

    def do_POST(self):
        if not self._origin_ok():
            return self._json(403, {"error": "origin not allowed"})
        if not self._authed():
            return self._json(401, {"error": "missing/invalid X-Companion-Token"})
        try:
            body = self._read_body()
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON body"})
        p = urlparse(self.path).path

        if p == "/download":
            return self._run(op_download, body.get("url", ""), body.get("title", "song"))
        if p == "/prepare":
            return self._run_locked(op_prepare, body["path_a"], body["path_b"],
                                    body.get("name_a", "A"), body.get("name_b", "B"))
        if p == "/reprompt":
            return self._run(op_reprompt, body["session"], body.get("brief"))
        if p == "/validate":
            return self._run(op_validate, body["session"], body.get("plan_text", ""))
        if p == "/render":
            return self._run_locked(op_render, body["session"], body["plan"])
        return self._json(404, {"error": "unknown endpoint"})

    # ---- execution wrappers ----
    def _run(self, fn, *args):
        try:
            return self._json(200, fn(*args))
        except Exception as e:
            traceback.print_exc()
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def _run_locked(self, fn, *args):
        if not JOB_LOCK.acquire(blocking=False):
            return self._json(429, {"error": "a heavy job is already running; retry shortly"})
        try:
            return self._run(fn, *args)
        finally:
            JOB_LOCK.release()


def main():
    global TOKEN
    if not ensure_consent():
        print("Terms not accepted -- exiting.")
        return
    os.makedirs(WORK_DIR, exist_ok=True)
    TOKEN = get_token()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("=" * 60)
    print(f"  Mashup Companion running at http://127.0.0.1:{PORT}")
    print(f"  Pairing token: {TOKEN}")
    print("  Paste this token into the web app when it asks.")
    print("  (Token also saved to companion_token.txt)")
    print("=" * 60)
    srv.serve_forever()


if __name__ == "__main__":
    main()
    