@echo off
title Mashup Studio
cd /d "%~dp0"

rem Optional: your hosted web app URL (used for auto-opening the browser,
rem must also be in ALLOWED_ORIGINS in companion.py). Uncomment and edit:
set "MASHUP_APP_URL=https://rahil-maniar.github.io/"

rem ---- make sure the uv helper exists (installs Python + packages for us) ----
where uv >nul 2>nul
if %errorlevel%==0 goto have_uv
if exist "%USERPROFILE%\.local\bin\uv.exe" goto have_uv
echo First-time setup: installing a small helper (uv)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if errorlevel 1 goto err
:have_uv
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

rem ---- first run: create a private Python + install everything ----
rem (.deps-ok marker is written only after a SUCCESSFUL install, so a failed
rem  or interrupted setup automatically retries next time)
if exist ".venv\.deps-ok" goto run
echo.
echo First-time setup: downloading Python and the audio tools.
echo This happens ONCE and can take several minutes. Please wait...
echo.
if not exist ".venv" uv venv .venv --python 3.11
if errorlevel 1 goto err
uv pip install -r requirements.txt
if errorlevel 1 goto err
echo ok> ".venv\.deps-ok"

:run
echo Starting Mashup Studio... (keep this window open while you use it)
".venv\Scripts\python.exe" companion.py
pause
exit /b

:err
echo.
echo Something went wrong during setup.
echo Take a photo of this window and send it to whoever shared this with you.
pause
