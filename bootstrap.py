"""
bootstrap.py -- compiled into "Mashup Studio.exe" (see build instructions below).

What the exe does when double-clicked:
  1. Downloads (or updates) the app from your website's mashup-studio-helper.zip
     into %LOCALAPPDATA%\\MashupStudio\\app  -- the user never sees code files.
  2. First run: installs uv, a private Python 3.11, and all dependencies into
     %LOCALAPPDATA%\\MashupStudio\\venv  (one-time, a few minutes).
  3. Starts the companion, which opens the browser at the web app, pre-paired.

Updates: push a new mashup-studio-helper.zip to your site and every user picks
it up on next launch automatically. The exe itself never needs re-downloading.

BUILD (on Windows, once):
    pip install pyinstaller
    pyinstaller --onefile --name "Mashup Studio" --icon icon.ico bootstrap.py
    -> dist/Mashup Studio.exe   (host it via GitHub Releases or your site)
"""
import hashlib
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

SITE = "https://rahil-maniar.github.io/Mashup-Maker"
ZIP_URL = SITE + "/mashup-studio-helper.zip"

HOME = os.path.join(os.environ.get("LOCALAPPDATA",
                    os.path.expanduser("~")), "MashupStudio")
APP_DIR = os.path.join(HOME, "app")
VENV = os.path.join(HOME, "venv")
VENV_PY = os.path.join(VENV, "Scripts", "python.exe") if os.name == "nt" \
    else os.path.join(VENV, "bin", "python")
ZIP_HASH_FILE = os.path.join(HOME, "app.hash")
DEPS_OK = os.path.join(HOME, "deps.ok")


def say(msg):
    print(f"  {msg}")


def fail(msg):
    print()
    print("  Something went wrong:", msg)
    print("  Take a photo of this window and send it to whoever shared the app.")
    input("  Press Enter to close...")
    sys.exit(1)


def fetch_app():
    """Download the app zip; (re)extract only when it changed."""
    say("Checking for updates...")
    try:
        with urllib.request.urlopen(ZIP_URL, timeout=30) as r:
            blob = r.read()
    except Exception as e:
        if os.path.isdir(APP_DIR):
            say("(offline or site unreachable -- using the installed version)")
            return
        fail(f"couldn't download the app: {e}")

    digest = hashlib.sha256(blob).hexdigest()
    old = open(ZIP_HASH_FILE).read().strip() if os.path.exists(ZIP_HASH_FILE) else ""
    if digest == old and os.path.isdir(APP_DIR):
        return

    say("Installing the latest version...")
    tmp = APP_DIR + ".new"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for info in z.infolist():
            # strip the leading "Mashup Studio/" folder from the zip
            parts = info.filename.split("/", 1)
            rel = parts[1] if len(parts) == 2 else parts[0]
            if not rel:
                continue
            dest = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if not info.is_dir():
                with z.open(info) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
    # keep user data (downloads/ etc.) across updates
    if os.path.isdir(APP_DIR):
        for keep in ("downloads", "companion_token.txt", "companion_consent.json"):
            src = os.path.join(APP_DIR, keep)
            if os.path.exists(src):
                dst = os.path.join(tmp, keep)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        shutil.rmtree(APP_DIR, ignore_errors=True)
    os.replace(tmp, APP_DIR)
    with open(ZIP_HASH_FILE, "w") as f:
        f.write(digest)


def find_uv():
    uv = shutil.which("uv")
    if uv:
        return uv
    local = os.path.join(os.path.expanduser("~"), ".local", "bin",
                         "uv.exe" if os.name == "nt" else "uv")
    return local if os.path.exists(local) else None


def ensure_uv():
    uv = find_uv()
    if uv:
        return uv
    say("First-time setup: installing a small helper (uv)...")
    try:
        if os.name == "nt":
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy",
                            "Bypass", "-Command",
                            "irm https://astral.sh/uv/install.ps1 | iex"],
                           check=True)
        else:
            subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh",
                           shell=True, check=True)
    except subprocess.CalledProcessError as e:
        fail(f"couldn't install uv: {e}")
    uv = find_uv()
    if not uv:
        fail("uv installed but not found")
    return uv


def ensure_deps(uv):
    if os.path.exists(DEPS_OK) and os.path.exists(VENV_PY):
        return
    say("First-time setup: downloading Python and the audio tools.")
    say("This happens ONCE and can take several minutes. Please wait...")
    req = os.path.join(APP_DIR, "requirements.txt")
    try:
        if not os.path.exists(VENV_PY):
            subprocess.run([uv, "venv", VENV, "--python", "3.11"], check=True)
        subprocess.run([uv, "pip", "install", "--python", VENV_PY,
                        "-r", req], check=True)
    except subprocess.CalledProcessError as e:
        fail(f"dependency install failed: {e}")
    with open(DEPS_OK, "w") as f:
        f.write("ok")


def main():
    print("=" * 60)
    print("  Mashup Studio")
    print("=" * 60)
    os.makedirs(HOME, exist_ok=True)
    fetch_app()
    uv = ensure_uv()
    ensure_deps(uv)
    say("Starting... (keep this window open while you use Mashup Studio)")
    try:
        subprocess.run([VENV_PY, "companion.py"], cwd=APP_DIR, check=False)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        fail(str(e))


if __name__ == "__main__":
    main()
    