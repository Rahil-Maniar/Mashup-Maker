import os
import ast
import pandas as pd
import yt_dlp
import librosa
import warnings
import time
from io import StringIO
from key_finder import detect_key  # Using your existing AI key detector
import numpy as np

# Convert key_finder's STRING key ('C Major') -> Spotify-style (int pitch class, mode).
# The recommender does int(row['key']) and get_camelot(key, mode); scraped rows MUST be
# integers or discovery crashes. This bridges the two formats.
_PITCH_TO_INT = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
_FLAT_TO_SHARP = {'Db':'C#','Eb':'D#','Gb':'F#','Ab':'G#','Bb':'A#'}
def key_string_to_spotify(key_str):
    """'C Major' -> (0, 1) ; 'A Minor' -> (9, 0). Returns (pitch_class_int, mode_int)."""
    if not key_str:
        return 0, 1
    parts = str(key_str).split()
    root = parts[0] if parts else 'C'
    scale = parts[1].lower() if len(parts) > 1 else 'major'
    root = _FLAT_TO_SHARP.get(root, root)
    return _PITCH_TO_INT.get(root, 0), (0 if scale.startswith('min') else 1)

def estimate_energy(y):
    """Rough Spotify-style 'energy' (0-1) from loudness. APPROXIMATION on a DIFFERENT scale
    than Spotify's trained value, so thresholds tuned on tracks.csv (e.g. energy>0.55) may
    need recalibration when mixing scraped + dataset songs."""
    rms = float(np.sqrt(np.mean(np.square(y)))) + 1e-9
    loud_db = 20.0 * np.log10(rms)          # ~ -30 (quiet) .. -8 (loud) for normalized audio
    return float(np.clip((loud_db + 30.0) / 22.0, 0.0, 1.0))


# Suppress librosa warnings for clean console output
warnings.filterwarnings("ignore", category=UserWarning)

CSV_PATH = "tracks.csv"
_CURRENT_YEAR = time.localtime().tm_year  # stamp scraped songs so the recommender's year>=2000 filter keeps them
TEMP_DIR = "temp_sniffs"

# ---- OUTPUT: a DEDICATED file of popular/chart songs, separate from the 1.2M tracks.csv ----
CHART_CSV = "chart_tracks.csv"

# ---- CHART SOURCES (Kworb). Country/global dailies share the 'Artist and Title' table format,
# so one parser handles them all. Each daily is ~Top 200, so several countries + global yields
# thousands of genuinely popular songs. VERIFY each URL opens in a browser -- Kworb occasionally
# renames paths. Add/remove freely; more sources = bigger pool.
CHART_URLS = [
    "https://kworb.net/spotify/country/global_daily.html",
    "https://kworb.net/spotify/country/global_weekly.html",
    "https://kworb.net/spotify/country/us_daily.html",
    "https://kworb.net/spotify/country/gb_daily.html",
    "https://kworb.net/spotify/country/in_daily.html",
    "https://kworb.net/spotify/country/ca_daily.html",
    "https://kworb.net/spotify/country/au_daily.html",
    "https://kworb.net/spotify/country/de_daily.html",
    "https://kworb.net/spotify/country/fr_daily.html",
    "https://kworb.net/spotify/country/br_daily.html",
    "https://kworb.net/spotify/country/mx_daily.html",
]
TOP_N_PER_CHART = 200  # Kworb dailies list ~Top 200

def get_safe_ydl_opts(base_opts=None):
    """Generate yt_dlp options with browser headers and cookie support."""
    safe = {
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 30,
        'no_check_certificate': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive'
        }
    }
    if base_opts:
        safe.update(base_opts)

    # Use a local exported cookie file if one exists in the project folder.
    for cookie_path in ('cookies.txt', 'youtube.cookies.txt'):
        if os.path.exists(cookie_path):
            safe['cookiefile'] = cookie_path
            break
    return safe

def normalize_artist_value(value):
    """Normalizes artist values from either plain strings or list-like strings."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list) and parsed:
                # Use primary artist to match scraped chart format.
                return str(parsed[0]).strip()
        except (ValueError, SyntaxError):
            pass

    return text.replace("['", "").replace("']", "").replace("'", "").strip()

def pick_column(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None

def fetch_chart_tracks(urls=CHART_URLS, top_n=TOP_N_PER_CHART):
    """Scrape several Kworb charts and merge into one de-duplicated pool of popular songs.
    Popularity = each song's BEST rank across charts (+ a small boost for appearing on
    multiple country charts, i.e. broadly popular)."""
    import requests
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    pool = {}
    for url in urls:
        try:
            print(f"\U0001f30d Fetching chart: {url}")
            html = requests.get(url, headers=headers, timeout=30).text
            tables = pd.read_html(StringIO(html))
            df_chart = tables[0].head(top_n)
            col = 'Artist and Title'
            if col not in df_chart.columns:   # fall back to the column with the most ' - '
                scores = {c: df_chart[c].astype(str).str.contains(' - ').sum() for c in df_chart.columns}
                col = max(scores, key=scores.get)
            for rank, (_, row) in enumerate(df_chart.iterrows()):
                raw = str(row[col])
                if " - " not in raw:
                    continue
                artist, title = raw.split(" - ", 1)
                k = (artist.strip().lower(), title.strip().lower())
                if k in pool:
                    pool[k]['best_rank'] = min(pool[k]['best_rank'], rank)
                    pool[k]['charts'] += 1
                else:
                    pool[k] = {'artist': artist.strip(), 'title': title.strip(),
                               'best_rank': rank, 'charts': 1}
            time.sleep(1)
        except Exception as e:
            print(f"\u274c chart failed ({url}): {e}")

    songs = []
    for v in pool.values():
        # #1 (rank 0) -> ~100; deeper ranks lower; +2 per extra chart it charted on
        v['popularity'] = int(min(100, max(55, 100 - v['best_rank']) + 2 * (v['charts'] - 1)))
        songs.append(v)
    songs.sort(key=lambda x: x['popularity'], reverse=True)  # most popular first (good if you stop early)
    print(f"\u2705 Gathered {len(songs)} unique popular songs from {len(urls)} charts.")
    return songs

def get_existing_tracks():
    """Loads current database to prevent redundant processing."""
    if not os.path.exists(CSV_PATH):
        # Create an empty CSV with the right headers if it doesn't exist
        pd.DataFrame(columns=['track_name', 'artists', 'tempo', 'key']).to_csv(CSV_PATH, index=False)
        return set()
    
    header = pd.read_csv(CSV_PATH, nrows=0)
    columns = list(header.columns)

    artist_col = pick_column(columns, ['artists', 'artist'])
    track_col = pick_column(columns, ['track_name', 'name', 'title'])

    if not artist_col or not track_col:
        return set()

    existing = set()
    chunk_iter = pd.read_csv(
        CSV_PATH,
        usecols=[artist_col, track_col],
        dtype=str,
        chunksize=50000,
        low_memory=True
    )

    for chunk in chunk_iter:
        artists = chunk[artist_col].fillna("").map(normalize_artist_value).str.lower().str.strip()
        tracks = chunk[track_col].fillna("").str.lower().str.strip()
        valid = (artists != "") & (tracks != "")
        existing.update((artists[valid] + " - " + tracks[valid]).tolist())

    return existing

def sniff_and_analyze(artist, title):
    """Download ~30s of the song and analyze BPM / key / energy.
    Uses an extension-less outtmpl (the pattern that works in app.py) and then GLOBS for the
    produced file, so naming quirks can't make us miss it. Prints the REAL failure reason."""
    import glob
    os.makedirs(TEMP_DIR, exist_ok=True)
    search_query = f"{artist} {title} audio"
    temp_base = os.path.join(TEMP_DIR, "current_sniff")

    def _cleanup():
        for f in glob.glob(temp_base + "*"):
            try:
                os.remove(f)
            except OSError:
                pass

    _cleanup()  # clear leftovers from the previous song
    ydl_opts = get_safe_ydl_opts({
        'format': 'bestaudio/best',
        'outtmpl': temp_base,                  # NO extension (same as app.py); postprocessor adds .wav
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'postprocessor_args': ['-t', '30'],    # cut to 30s to save time/bandwidth
        'noplaylist': True,
    })

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(f"ytsearch1:{search_query}", download=True)

        # Find the produced audio file no matter how yt-dlp named it
        files = glob.glob(temp_base + "*.wav") or glob.glob(temp_base + "*")
        if not files:
            print("   \u26a0\ufe0f  no audio file produced (likely a download block / yt-dlp issue)")
            return None, None, None
        temp_path = files[0]

        y, sr = librosa.load(temp_path, sr=22050)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = tempo if isinstance(tempo, float) else float(tempo[0])
        key = detect_key(y, sr, path=temp_path)
        energy = estimate_energy(y)
        return round(float(bpm), 1), key, energy

    except Exception as e:
        print(f"   \u26a0\ufe0f  failed: {type(e).__name__}: {e}")
        return None, None, None
    finally:
        _cleanup()

def get_existing_chart_tracks():
    """Songs already in CHART_CSV, so a resumed run skips them."""
    if not os.path.exists(CHART_CSV) or os.path.getsize(CHART_CSV) == 0:
        return set()
    try:
        df = pd.read_csv(CHART_CSV, usecols=['track_name', 'artists'], dtype=str)
    except Exception:
        return set()
    a = df['artists'].fillna('').str.lower().str.strip()
    t = df['track_name'].fillna('').str.lower().str.strip()
    return set((a + " - " + t).tolist())

def append_chart_row(row):
    """Append ONE analyzed song immediately -> a multi-hour run is crash/Ctrl-C safe."""
    pd.DataFrame([row]).to_csv(CHART_CSV, mode='a',
                               header=not os.path.exists(CHART_CSV), index=False)

def main():
    print("\U0001f680 Building popular-song pool -> chart_tracks.csv")
    print("   Resumable: progress saved after EVERY song. Ctrl-C anytime; re-run to continue.\n")
    songs = fetch_chart_tracks()
    done = get_existing_chart_tracks()
    todo = [s for s in songs if f"{s['artist'].lower()} - {s['title'].lower()}" not in done]
    print(f"\U0001f4cb {len(songs)} popular songs | {len(done)} already done | {len(todo)} to analyze.\n")

    processed = 0
    t_start = time.time()
    for i, s in enumerate(todo):
        print(f"[{i + 1}/{len(todo)}] \U0001f3b5 {s['artist']} - {s['title']}")
        bpm, key_str, energy = sniff_and_analyze(s['artist'], s['title'])
        if bpm and key_str:
            key_int, mode_int = key_string_to_spotify(key_str)
            append_chart_row({
                'track_name': s['title'],
                'artists': s['artist'],
                'tempo': bpm,
                'year': _CURRENT_YEAR,
                'key': key_int,               # integer pitch class (recommender expects int)
                'mode': mode_int,             # 1 = major, 0 = minor
                'energy': round(energy, 3),   # computed from audio (approx scale)
                'valence': 0.5,               # APPROX: not computable locally
                'speechiness': 0.06,          # APPROX: keeps vocal-bearing songs in vocal pool
                'acousticness': 0.15,         # APPROX: most chart pop is non-acoustic
                'popularity': s['popularity'] # REAL popularity signal from chart rank
            })
            processed += 1
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            eta_h = ((len(todo) - (i + 1)) / rate / 3600.0) if rate > 0 else 0
            print(f"   => {key_str} | {bpm} BPM | energy {energy:.2f} | pop {s['popularity']}  "
                  f"(saved \u2713 | ~{eta_h:.1f}h left)")
        else:
            print("   => skipped (analysis failed)")
        time.sleep(2)  # polite to YouTube; fine since you're running for hours

    print(f"\n\u2705 Done. Added {processed} new songs to {CHART_CSV}.")

if __name__ == "__main__":
    main()
    