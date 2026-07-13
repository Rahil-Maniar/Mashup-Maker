#!/bin/bash
# Mashup Studio launcher (macOS / Linux). Double-click to start.
# macOS first time: RIGHT-CLICK this file -> Open (Gatekeeper).
cd "$(dirname "$0")" || exit 1

# Optional: your hosted web app URL (auto-opens the browser; must also be in
# ALLOWED_ORIGINS in companion.py). Uncomment and edit:
# export MASHUP_APP_URL="https://your-app.pages.dev"

fail() {
  echo
  echo "Something went wrong during setup."
  echo "Take a photo of this window and send it to whoever shared this with you."
  read -r -p "Press Enter to close..."
  exit 1
}

# ---- make sure the uv helper exists (installs Python + packages for us) ----
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "First-time setup: installing a small helper (uv)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || fail
fi
export PATH="$HOME/.local/bin:$PATH"

# ---- first run: create a private Python + install everything ----
# (.deps-ok marker is written only after a SUCCESSFUL install, so a failed
#  or interrupted setup automatically retries next time)
if [ ! -f .venv/.deps-ok ]; then
  echo
  echo "First-time setup: downloading Python and the audio tools."
  echo "This happens ONCE and can take several minutes. Please wait..."
  echo
  [ -d .venv ] || uv venv .venv --python 3.11 || fail
  uv pip install -r requirements.txt || fail
  touch .venv/.deps-ok
fi

echo "Starting Mashup Studio... (keep this window open while you use it)"
exec .venv/bin/python companion.py
