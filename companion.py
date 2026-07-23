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
- First-run consent gate: heavy endpoints refuse to serve until the user has
  explicitly accepted the terms IN THE WEB APP (one click; stored in
  companion_consent.json). The terminal never blocks on input.

Run:  python companion.py
"""

import os
import json
import time
import secrets
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

PORT = 7777
WORK_DIR = "downloads"
CONSENT_FILE = "companion_consent.json"
TOKEN_FILE = "companion_token.txt"

# Your hosted web UI. When set, the Companion auto-opens the browser here with
# the pairing token in the #fragment (never sent to any server), so users are
# paired automatically -- no copy/paste. Leave "" to disable auto-open.
APP_URL = os.environ.get("MASHUP_APP_URL",
                         "https://rahil-maniar.github.io/Mashup-Maker")

# Origins allowed to talk to this Companion. NOTE: an Origin is scheme+host
# only -- never include the /Mashup-Maker path here.
ALLOWED_ORIGINS = {
    "https://rahil-maniar.github.io",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
}

CONSENT_TEXT = """This helper runs ON YOUR COMPUTER and, when you use the web app, will:

  1. Search YouTube and DOWNLOAD audio to this machine, over YOUR internet \
connection and IP address, at your request.
  2. Process that audio locally (stem separation, analysis, mashup rendering).

Downloading may violate YouTube's Terms of Service and, depending on your \
country, copyright law. Outputs are for PERSONAL, non-commercial use. \
You are responsible for your own use of this tool. Nothing is uploaded \
anywhere; all audio stays on this machine."""

# one heavy job at a time (single GPU)
JOB_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# progress reporting (the UI polls GET /progress during long jobs)
# ---------------------------------------------------------------------------
_PROGRESS = {"stage": "idle", "detail": "", "step": 0, "steps": 0,
             "busy": False, "ts": 0.0}
_PROGRESS_LOCK = threading.Lock()


def set_progress(stage, detail="", step=0, steps=0):
    with _PROGRESS_LOCK:
        _PROGRESS.update(stage=stage, detail=detail, step=step, steps=steps,
                         busy=stage not in ("idle",), ts=time.time())


# ---------------------------------------------------------------------------
# consent + token
# ---------------------------------------------------------------------------
def has_consent():
    return os.path.exists(CONSENT_FILE)


def record_consent(origin):
    with open(CONSENT_FILE, "w") as f:
        json.dump({"accepted_at": time.time(),
                   "accepted_via": origin or "local"}, f)


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


_RECOMMENDER = None
_SHOWN_PAIRS = set()   # pairs already suggested this session -- rolls stay fresh

def op_recommend(n):
    """Suggest compatible song pairs from the local chart/tracks database."""
    global _RECOMMENDER
    if _RECOMMENDER is None:
        from recommender import MashupRecommender
        _RECOMMENDER = MashupRecommender("tracks.csv")   # prefers chart_tracks.csv if present
    if _RECOMMENDER.df.empty:
        raise RuntimeError("no song database found (need chart_tracks.csv or tracks.csv "
                           "next to companion.py)")
    pairs = _RECOMMENDER.discover_mashups(n, exclude=_SHOWN_PAIRS)
    if len(pairs) < n and _SHOWN_PAIRS:
        # we've cycled through everything fresh -- start over
        _SHOWN_PAIRS.clear()
        pairs = _RECOMMENDER.discover_mashups(n)
    for p in pairs:
        _SHOWN_PAIRS.add(tuple(sorted([p["song_a"]["title"], p["song_b"]["title"]])))
    return {"pairs": pairs}


def op_download(url, title):
    from get_song import download
    path = download(url, name_hint=title)
    if not path:
        raise RuntimeError("download produced no file")
    return {"path": path, "title": title}


def op_prepare(path_a, path_b, name_a, name_b):
    _ensure_gpu_stack_ok("prepare")  # refuse cleanly if torch stack is broken
    from llm_dj import analyze_song, compute_shifts, bar_grid, \
        transcribe_lyrics, lyrics_to_bar_lines, build_prompt
    from mashup_maker import separate_full_song
    from studio_logic import blend_report, file_hash

    if file_hash(path_a) == file_hash(path_b):
        raise ValueError("Both slots contain the same audio file.")

    STEPS = 8
    songs, grids, lyrics = {}, {}, {}
    names = {"A": name_a, "B": name_b}
    step = 1
    for sid, path in (("A", path_a), ("B", path_b)):
        set_progress("prepare", f"Analyzing \u201c{names[sid]}\u201d (beat + key)", step, STEPS); step += 1
        meta = analyze_song(path)
        set_progress("prepare", f"Separating vocals from \u201c{names[sid]}\u201d "
                                f"\u2014 the slow part, several minutes on CPU", step, STEPS); step += 1
        stems = separate_full_song(path, WORK_DIR)
        if not stems:
            raise RuntimeError(f"separation failed for {names[sid]}")
        songs[sid] = {"stems": stems, "bpm": meta["bpm"],
                      "grid_start": meta["anchor"] % (4 * 60.0 / meta["bpm"]),
                      "key": meta["key"], "shift": 0}
    sa, sb = compute_shifts(songs["A"]["key"], songs["B"]["key"])
    songs["A"]["shift"], songs["B"]["shift"] = sa, sb
    for sid in ("A", "B"):
        set_progress("prepare", f"Listening for lyrics in \u201c{names[sid]}\u201d", step, STEPS); step += 1
        grids[sid] = bar_grid(songs[sid])
        segs = transcribe_lyrics(songs[sid]["stems"]["vocals"])
        lyrics[sid] = lyrics_to_bar_lines(segs, songs[sid]["bpm"], songs[sid]["grid_start"])

    set_progress("prepare", "Writing the DJ briefing", STEPS, STEPS)
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
    _ensure_gpu_stack_ok("render")  # refuse if torch stack is broken
    from dsl_renderer import render_plan
    from critic import critique_render
    set_progress("render", "Stretching and mixing stems", 1, 2)
    out, report = render_plan(plan, session["songs"], "llm_mashup.wav")
    set_progress("render", "Running the quality check", 2, 2)
    res = critique_render(out, plan, session)
    return {"file": out, "render_report": report, "critic": res["report"],
            "flags": res["flags"]}


def op_auto_mashup(session):
    """One-click path: build a default arrangement (no LLM) and render it."""
    _ensure_gpu_stack_ok("auto_mashup")  # refuse if torch stack is broken
    from studio_logic import auto_plan
    from dsl_renderer import validate_plan, render_plan
    from critic import critique_render
    set_progress("auto", "Designing an arrangement", 1, 3)
    plan = auto_plan(session)
    errors, warnings = validate_plan(plan, session["songs"])
    if errors:
        raise RuntimeError("auto-plan failed validation: " + "; ".join(errors[:3]))
    set_progress("auto", "Mixing your mashup", 2, 3)
    out, report = render_plan(plan, session["songs"], "llm_mashup.wav")
    set_progress("auto", "Running the quality check", 3, 3)
    res = critique_render(out, plan, session)
    return {"plan": plan, "file": out, "render_report": report,
            "critic": res["report"], "flags": res["flags"],
            "warnings": warnings}


# ---------------------------------------------------------------------------
# GPU acceleration (detect always; install only on explicit user opt-in)
# ---------------------------------------------------------------------------
_GPU = {"installing": False, "restart_needed": False, "error": None}


def _suppress_windows_error_dialogs():
    """Windows: stop the OS from showing modal 'Entry Point Not Found' /
    critical-error dialog boxes when a DLL fails to load. With this set, a
    bad DLL surfaces as a normal ImportError/OSError that our try/except
    handling turns into a readable JSON error -- instead of a popup that
    freezes a background process until someone clicks OK. Child processes
    inherit this error mode by default, so subprocesses are covered too."""
    if os.name != "nt":
        return
    try:
        import ctypes
        SEM = 0x0001 | 0x0002 | 0x8000  # FAILCRITICALERRORS|NOGPFAULT|NOOPENFILE
        ctypes.windll.kernel32.SetErrorMode(SEM)
    except Exception:
        pass


def _torch_stack_consistent():
    """True if torch/torchvision look ABI-compatible, judged purely from
    installed-package METADATA. No module is imported and no DLL is loaded,
    so this can never crash, hang, or trigger a Windows error dialog.

    Two independent signals:
      1. Build-tag agreement: torch '2.6.0+cu124' next to torchvision
         '0.21.0+cpu' is exactly the broken combo -- the version numbers
         "match" but the compiled extensions do not.
      2. torchvision's wheel metadata pins the torch it was compiled
         against (e.g. 'torch==2.6.0'); the installed torch must satisfy it.
    """
    try:
        from importlib.metadata import version, requires, PackageNotFoundError
    except ImportError:
        return True  # ancient Python; don't block on it
    try:
        t, tv = version("torch"), version("torchvision")
    except PackageNotFoundError:
        return True  # one of them absent -> nothing to mismatch
    except Exception:
        return True

    def tag(v):  # '2.6.0+cu124' -> 'cu124', '2.6.0' -> ''
        return v.split("+", 1)[1] if "+" in v else ""

    if tag(t) != tag(tv):
        return False

    # Cross-check onnxruntime-gpu vs torch CUDA generation.
    # Mirrors the same table used by _pick_cuda_config() in the installer:
    #   CUDA 11  -> ort <= 1.16
    #   CUDA 12  -> ort 1.17-1.20
    #   CUDA 13+ -> ort 1.21+
    # Catching this from metadata means the DLL (cublasLt64_13.dll etc.) is
    # never loaded by a mismatched binary, so no Windows error dialog can appear.
    import re as _re
    cuda_tag = tag(t)
    if cuda_tag and cuda_tag.startswith("cu"):
        try:
            cuda_num = int(_re.search(r"\d+", cuda_tag).group())  # cu124 -> 124
            cuda_major = cuda_num // 10                            # 124 -> 12
            ort_ver = version("onnxruntime-gpu")
            ort_minor = int(ort_ver.split(".")[1])  # 1.20.1 -> 20
            if cuda_major <= 11 and ort_minor > 16:
                return False
            if cuda_major == 12 and not (17 <= ort_minor <= 20):
                return False
            if cuda_major >= 13 and ort_minor < 21:
                return False
        except PackageNotFoundError:
            pass
        except Exception:
            pass

    import re
    try:
        reqs = requires("torchvision") or []
    except Exception:
        reqs = []
    for req in reqs:
        m = re.match(r"torch\s*\(?\s*==\s*([\d.]+)", req)
        if m and not (t == m.group(1) or t.startswith(m.group(1) + "+")):
            return False
    return True


def _ensure_gpu_stack_ok(context=""):
    """Gate called at the top of every heavy job (prepare, render, auto-mashup).

    Prevents a broken GPU stack from crashing deep inside separation by
    refusing to start the job and returning a clear, actionable message.

    Two cases handled:
    A) Self-heal already running (flag set before thread spawned, so no
       race window): tell the user to wait and watch the progress bar.
    B) Stack is inconsistent but repair not yet started (e.g. user hit
       prepare faster than _gpu_selfheal's subprocess probe finished):
       set flag + start repair NOW, then refuse the job with instructions.
    """
    # Case A: repair already in flight (flag guaranteed set before thread).
    if _GPU["installing"]:
        raise RuntimeError(
            "GPU packages are being repaired right now -- "
            "watch the progress bar at the top of the page and retry "
            "when it finishes, then restart the helper window.")
    # Case B: metadata check catches it before any DLL is loaded.
    if _gpu_accelerated() and not _torch_stack_consistent():
        print("  Detected mismatched GPU packages at job gate -- starting repair...")
        _GPU.update(installing=True, error=None)  # flag before thread
        threading.Thread(target=_gpu_install_worker, daemon=True).start()
        raise RuntimeError(
            "GPU packages were out of sync -- an automatic repair just started. "
            "It is a one-time download (watch the progress bar). "
            "When it finishes, restart the helper and try again."
            + (f" [{context}]" if context else ""))


def _gpu_present():
    from shutil import which
    return which("nvidia-smi") is not None


def _gpu_accelerated():
    """True only when BOTH GPU pieces are in place: onnxruntime-gpu AND a
    CUDA build of torch. audio-separator gates hardware acceleration on
    torch.cuda.is_available(), so a CPU-wheel torch (e.g. '2.13.0+cpu')
    silently forces CPU mode even with onnxruntime-gpu installed."""
    try:
        from importlib.metadata import version
        version("onnxruntime-gpu")
        return "+cu" in version("torch")
    except Exception:
        return False


def gpu_state():
    return {"present": _gpu_present(), "accelerated": _gpu_accelerated(),
            "installing": _GPU["installing"],
            "restart_needed": _GPU["restart_needed"], "error": _GPU["error"]}


def _gpu_install_worker():
    """Installs CUDA torch/torchvision/torchaudio + onnxruntime-gpu into this
    venv. torch, torchvision and torchaudio MUST be installed together from
    the same index -- mismatched versions cause 'operator does not exist'
    errors at separation time. Runs in a background thread; any failure leaves
    the CPU setup untouched. Streams installer output for live UI progress."""
    import re
    import subprocess
    import sys
    from shutil import which
    uv = which("uv")

    # ---------------------------------------------------------------------------
    # Detect the best CUDA version this GPU can run, then select matching
    # torch + onnxruntime-gpu versions automatically. This means the installer
    # works correctly across GTX 16xx (cu121), RTX 20/30/40 (cu124), and
    # future hardware (cu130+) without any hardcoded assumptions.
    # ---------------------------------------------------------------------------
    def _pick_cuda_config():
        """Return (torch_index_url, ort_gpu_pin) for this machine's GPU.

        IMPORTANT: torch wheel availability lags driver support by months.
        cu130 wheels do NOT exist on PyPI yet (as of 2026-07), so even a
        driver that supports CUDA 13 must use cu124 -- the newest published
        index -- to avoid a failed install. Update NEWEST_AVAILABLE when
        PyTorch publishes a newer CUDA build.

        ort pin rules (onnxruntime release notes):
          ort 1.17-1.20  CUDA 12 (cu121 / cu124)
          ort <= 1.16    CUDA 11 (cu118)
        """
        import subprocess as _sp, re as _re

        # Newest torch index with confirmed published wheels.
        # Change this (and the ort rule above) when PyTorch ships cu130.
        NEWEST_AVAILABLE = "cu124"

        cuda_major = 12  # safe fallback
        try:
            out = _sp.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, timeout=10).strip().splitlines()[0]
            drv = float(_re.search(r"[\d.]+", out).group())
            # driver >= 525 supports CUDA 12; >= 450 supports CUDA 11.
            # Drivers >= 576 support CUDA 13 but we cap at NEWEST_AVAILABLE
            # because cu130 torch wheels don't exist yet.
            if drv >= 525:
                cuda_major = 12  # covers driver >= 576 too until cu130 ships
            elif drv >= 450:
                cuda_major = 11
            else:
                cuda_major = 11
        except Exception:
            pass

        if cuda_major >= 12:
            return f"https://download.pytorch.org/whl/{NEWEST_AVAILABLE}", "onnxruntime-gpu>=1.17,<=1.20.1"
        else:  # CUDA 11
            return "https://download.pytorch.org/whl/cu118", "onnxruntime-gpu>=1.14,<=1.16.3"

    TORCH_INDEX, ORT_PKG = _pick_cuda_config()
    TORCH_PKGS = ["torch", "torchvision", "torchaudio", "--index-url", TORCH_INDEX]
    print(f"  GPU installer: using {TORCH_INDEX.split('/')[-1]} | {ORT_PKG}")

    def stream(args, step, steps, fallback_msg, check=True):
        """Run an installer subprocess, pushing per-package lines to progress."""
        set_progress("gpu", fallback_msg, step, steps)
        proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1, errors="replace")
        tail = []
        pat = re.compile(
            r"(Downloading|Downloaded|Prepared|Installed|"
            r"Collecting|Installing collected packages)"
            r"[ :]+(\\S+)?(?:\\s+\\([\\d.]+\\s*[KMG]i?B\\))?", re.I)
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            tail.append(line); tail = tail[-20:]
            m = pat.search(line)
            if m:
                what, pkg = m.group(1), m.group(2) or ""
                set_progress("gpu", f"{what} {pkg}".strip(), step, steps)
        proc.wait()
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, args,
                                                output="\n".join(tail))

    STEPS = 4
    try:
        if uv:
            cuda_tag = "cu12" if "cu12" in TORCH_INDEX else ("cu13" if "cu13" in TORCH_INDEX else "cu11")
            cublas = f"nvidia-cublas-{cuda_tag}"
            cudnn  = f"nvidia-cudnn-{cuda_tag}"
            stream([uv, "pip", "install", "--python", sys.executable,
                    cublas, cudnn],
                   1, STEPS, "Downloading CUDA runtime libraries")
            # --reinstall forces replacement of ALL packages in this command,
            # including torchvision whose version number hasn't changed but
            # whose compiled .pyd must match the new torch ABI.
            stream([uv, "pip", "install", "--python", sys.executable,
                    "--reinstall"] + TORCH_PKGS,
                   2, STEPS, "Installing GPU compute engine (~2.5 GB, the big one)")
            stream([uv, "pip", "uninstall", "--python", sys.executable,
                    "onnxruntime"],
                   3, STEPS, "Removing CPU audio engine", check=False)
            stream([uv, "pip", "install", "--python", sys.executable,
                    ORT_PKG],
                   4, STEPS, f"Installing GPU audio engine ({ORT_PKG})")
        else:
            cuda_tag = "cu12" if "cu12" in TORCH_INDEX else ("cu13" if "cu13" in TORCH_INDEX else "cu11")
            cublas = f"nvidia-cublas-{cuda_tag}"
            cudnn  = f"nvidia-cudnn-{cuda_tag}"
            stream([sys.executable, "-m", "pip", "install",
                    cublas, cudnn],
                   1, STEPS, "Downloading CUDA runtime libraries")
            stream([sys.executable, "-m", "pip", "install",
                    "--force-reinstall"] + TORCH_PKGS,
                   2, STEPS, "Installing GPU compute engine (~2.5 GB, the big one)")
            stream([sys.executable, "-m", "pip", "uninstall", "-y",
                    "onnxruntime"],
                   3, STEPS, "Removing CPU audio engine", check=False)
            stream([sys.executable, "-m", "pip", "install", ORT_PKG],
                   4, STEPS, f"Installing GPU audio engine ({ORT_PKG})")
        _GPU.update(restart_needed=True, error=None)
        print("  GPU acceleration installed -- restart the helper to use it.")
    except subprocess.CalledProcessError as e:
        _GPU["error"] = ((e.output or str(e)) or "")[-500:]
        print("  GPU install failed (CPU mode still works fine).")
    except Exception as e:
        _GPU["error"] = str(e)
    finally:
        _GPU["installing"] = False
        set_progress("idle")


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
            out = {"ok": True, "name": "mashup-companion",
                   "authed": self._authed(), "consented": has_consent(),
                   "gpu": gpu_state()}
            if not has_consent():
                out["terms"] = CONSENT_TEXT
            return self._json(200, out)

        if not self._authed():
            return self._json(401, {"error": "missing/invalid X-Companion-Token"})

        if url.path == "/whoami":
            # cheap token test for pairing -- no side effects
            return self._json(200, {"ok": True, "consented": has_consent()})

        if url.path == "/progress":
            with _PROGRESS_LOCK:
                return self._json(200, dict(_PROGRESS))

        if not has_consent():
            return self._json(403, {"error": "consent required -- accept the terms in the web app"})

        if url.path == "/search":
            q = (parse_qs(url.query).get("q") or [""])[0]
            if not q:
                return self._json(400, {"error": "q required"})
            return self._run(op_search, q)

        if url.path == "/recommend":
            n = (parse_qs(url.query).get("n") or ["4"])[0]
            try:
                n = max(1, min(8, int(n)))
            except ValueError:
                n = 4
            return self._run(op_recommend, n)

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

        if p == "/consent":
            if body.get("accept") is not True:
                return self._json(400, {"error": "send {\"accept\": true} to accept the terms"})
            record_consent(self.headers.get("Origin"))
            return self._json(200, {"ok": True, "consented": True})

        if not has_consent():
            return self._json(403, {"error": "consent required -- accept the terms in the web app"})

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
        if p == "/auto_mashup":
            return self._run_locked(op_auto_mashup, body["session"])
        if p == "/gpu/enable":
            if _gpu_accelerated():
                return self._json(200, {"ok": True, "already": True})
            if not _gpu_present():
                return self._json(400, {"error": "no NVIDIA GPU detected"})
            if not _GPU["installing"]:
                _GPU.update(installing=True, error=None)
                threading.Thread(target=_gpu_install_worker, daemon=True).start()
            return self._json(200, {"ok": True, "installing": True})
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
            set_progress("idle")
            JOB_LOCK.release()


def _enable_cuda_dlls():
    """Windows: make bundled CUDA DLLs findable so faster-whisper and
    onnxruntime-gpu can use the GPU. Covers two pip layouts:
      - nvidia-cublas-cu12 / nvidia-cudnn-cu12  -> site-packages/nvidia/*/bin
      - CUDA builds of torch (e.g. 2.5.1+cu121) -> site-packages/torch/lib
    No-op on non-Windows or when nothing is installed."""
    if os.name != "nt":
        return
    try:
        import site
        dirs = []
        for sp in site.getsitepackages():
            nv = os.path.join(sp, "nvidia")
            if os.path.isdir(nv):
                for pkg in os.listdir(nv):
                    b = os.path.join(nv, pkg, "bin")
                    if os.path.isdir(b):
                        dirs.append(b)
            tl = os.path.join(sp, "torch", "lib")
            if os.path.isdir(tl):
                dirs.append(tl)
        for d in dirs:
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def _gpu_selfheal():
    """Detect and repair a broken GPU install (mismatched torchvision).

    Older installer versions swapped in CUDA torch but left the CPU-built
    torchvision behind, whose compiled _C.pyd doesn't match the new torch
    ABI -> 'operator torchvision::nms does not exist' / DLL entry-point
    errors at separation time. The venv survives app updates, so shipping
    new code alone can't fix affected machines. Here we verify torchvision
    actually imports in a throwaway subprocess (so a hard DLL crash can't
    take the server down) and, if broken, re-run the installer worker."""
    if not _gpu_accelerated() or _GPU["installing"]:
        return
    # -- Layer 1: metadata check. Instant, imports nothing, loads no DLLs,
    #    so it cannot pop a Windows error dialog. Catches every known case
    #    (CUDA torch + leftover CPU torchvision) before any user action.
    if not _torch_stack_consistent():
        broken = True
    else:
        # -- Layer 2: functional probe in a throwaway subprocess. The child
        #    sets SetErrorMode FIRST so a bad DLL load fails programmatically
        #    (ImportError -> nonzero exit) instead of showing a modal
        #    'Entry Point Not Found' dialog that would block until clicked.
        #    We call nms() for real because 'import torchvision' can succeed
        #    even when its C extension failed to register.
        import subprocess
        import sys
        PROBE = ("import ctypes,os;"
                 "os.name=='nt' and ctypes.windll.kernel32.SetErrorMode(0x8003);"
                 "import torch;from torchvision.ops import nms;"
                 "nms(torch.zeros((1,4)),torch.zeros(1),0.5)")
        try:
            r = subprocess.run([sys.executable, "-c", PROBE],
                               capture_output=True, timeout=180)
            broken = r.returncode != 0
        except subprocess.TimeoutExpired:
            broken = True  # a healthy probe never takes 3 minutes
        except Exception:
            return  # can't verify; don't risk a pointless 2.5 GB reinstall
    if not broken:
        return
    print("  Detected a broken GPU install (mismatched packages).")
    print("  Repairing automatically -- this is a one-time download...")
    print("  Separation jobs will be blocked until the repair finishes.")
    _GPU.update(installing=True, error=None)  # set flag BEFORE thread starts
                                               # so op_prepare gate sees it instantly
    threading.Thread(target=_gpu_install_worker, daemon=True).start()


def main():
    global TOKEN
    _suppress_windows_error_dialogs()  # must run before any DLL can load
    os.makedirs(WORK_DIR, exist_ok=True)
    TOKEN = get_token()
    _enable_cuda_dlls()
    threading.Thread(target=_gpu_selfheal, daemon=True).start()
    try:  # bundle ffmpeg/ffprobe for yt-dlp + whisper (downloads once, ~80 MB)
        import static_ffmpeg
        static_ffmpeg.add_paths()
    except ImportError:
        pass  # fine if ffmpeg is already installed system-wide
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("=" * 60)
    print("  Mashup Studio helper is running.")
    print(f"  [VERSION] companion.py build: 2026-07-23-r2 (gpu-stack-fix)")
    # --- GPU stack diagnostic: printed every launch so logs are unambiguous ---
    try:
        from importlib.metadata import version as _v, PackageNotFoundError as _PNF
        def _safe_ver(p):
            try: return _v(p)
            except _PNF: return "not installed"
        _t   = _safe_ver("torch")
        _tv  = _safe_ver("torchvision")
        _ort = _safe_ver("onnxruntime-gpu")
        _consistent = _torch_stack_consistent()
        print(f"  [GPU] torch={_t}  torchvision={_tv}  onnxruntime-gpu={_ort}")
        print(f"  [GPU] stack consistent: {_consistent}")
        if not _consistent:
            print("  [GPU] *** MISMATCH DETECTED — self-heal will start ***")
    except Exception as _e:
        print(f"  [GPU] diagnostic failed: {_e}")
    # -------------------------------------------------------------------------
    if APP_URL:
        pair_url = f"{APP_URL.rstrip('/')}/#token={TOKEN}"
        print("  Opening the app in your browser...")
        print(f"  (If it doesn't open, visit: {pair_url})")
        threading.Timer(1.0, webbrowser.open, args=(pair_url,)).start()
    else:
        print(f"  Pairing token: {TOKEN}")
        print("  Paste this token into the web app when it asks.")
        print("  (Token also saved to companion_token.txt)")
    if not has_consent():
        print("  First run: you'll be asked to accept the terms in the web app.")
    print("  Keep this window open while you use Mashup Studio.")
    print("=" * 60)
    srv.serve_forever()


if __name__ == "__main__":
    main()
    