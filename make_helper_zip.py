"""
make_helper_zip.py -- builds mashup-studio-helper.zip for the website.

Run from the project folder:  python make_helper_zip.py
Then commit/upload the zip next to index.html in your Pages repo.

Includes exactly what end users need, preserves the executable bit on the
mac launcher, and refuses to ship your personal files (tokens, consent,
downloads, giant CSVs).
"""
import os
import sys
import zipfile

ZIP_NAME = "mashup-studio-helper.zip"
FOLDER = "Mashup Studio"          # folder users see after unzipping

INCLUDE = [
    # core pipeline
    "companion.py", "get_song.py", "llm_dj.py", "mashup_maker.py",
    "studio_logic.py", "dsl_renderer.py", "critic.py", "key_finder.py",
    "recommender.py",
    # setup
    "requirements.txt",
    "Start Mashup Studio.bat",
    "Start Mashup Studio.command",
    # song database for the 🎲 recommender (small chart file only!)
    "chart_tracks.csv",
]

NEVER_SHIP = {"companion_token.txt", "companion_consent.json", "tracks.csv",
              "dj_session.json", "dj_prompt.txt", "llm_mashup.wav"}


def main():
    missing = [f for f in INCLUDE if not os.path.exists(f)]
    if missing:
        print("Missing files (fix before shipping):")
        for f in missing:
            print("  -", f)
        sys.exit(1)

    for f in INCLUDE:
        if f in NEVER_SHIP:
            sys.exit(f"refusing to ship {f}")

    big = [f for f in INCLUDE if os.path.getsize(f) > 25 * 1024 * 1024]
    if big:
        sys.exit(f"these files are suspiciously large for the zip: {big}")

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as z:
        for f in INCLUDE:
            arc = f"{FOLDER}/{f}"
            if f.endswith(".command"):
                # preserve the executable bit so macOS can run it
                info = zipfile.ZipInfo(arc)
                info.external_attr = 0o755 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                with open(f, "rb") as fh:
                    z.writestr(info, fh.read())
            else:
                z.write(f, arc)

    size = os.path.getsize(ZIP_NAME) / 1024
    print(f"Built {ZIP_NAME} ({size:.0f} KB) with {len(INCLUDE)} files.")
    print("Upload it next to index.html in your GitHub Pages repo.")


if __name__ == "__main__":
    main()
    