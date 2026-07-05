"""
get_song.py -- Search YouTube by song name, pick from a numbered list, download as WAV.

Usage:
    python get_song.py                      # interactive loop: type queries, pick, download
    python get_song.py "save your tears"    # start with a query
    python get_song.py <youtube_url>        # skip search, download directly

Downloads land in downloads/ as clean-named WAVs, ready for:
    python llm_dj.py prepare downloads/SongA.wav downloads/SongB.wav
"""

import os
import re
import sys
import yt_dlp

OUT_DIR = "downloads"
N_RESULTS = 6


def get_safe_ydl_opts(base_opts=None):
    """Same header/cookie pattern as the rest of the project."""
    safe = {
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 30,
        'no_check_certificate': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.5',
        },
    }
    if base_opts:
        safe.update(base_opts)
    for cookie_path in ('cookies.txt', 'youtube.cookies.txt'):
        if os.path.exists(cookie_path):
            safe['cookiefile'] = cookie_path
            break
    return safe


def clean_filename(name, max_len=60):
    name = re.sub(r'[\\/*?:"<>|#]', '', str(name))
    name = re.sub(r'\s+', ' ', name).strip()
    return name[:max_len].strip() or "song"


def fmt_duration(sec):
    if not sec:
        return "?:??"
    sec = int(sec)
    return f"{sec // 60}:{sec % 60:02d}"


def fmt_views(v):
    if not v:
        return "?"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}K"
    return str(v)


def search(query, n=N_RESULTS):
    """Return top-n search results as [{title, url, uploader, duration, views}]."""
    opts = get_safe_ydl_opts({'extract_flat': True, 'noplaylist': True})
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    results = []
    for e in (info.get('entries') or []):
        results.append({
            'title': e.get('title', '(untitled)'),
            'url': e.get('url') or f"https://www.youtube.com/watch?v={e.get('id')}",
            'uploader': e.get('uploader') or e.get('channel') or '?',
            'duration': e.get('duration'),
            'views': e.get('view_count'),
        })
    return results


def download(url, name_hint=None):
    """Download bestaudio -> WAV in OUT_DIR. Returns the final path."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if name_hint is None:  # fetch title for the filename
        opts = get_safe_ydl_opts({'noplaylist': True})
        with yt_dlp.YoutubeDL(opts) as ydl:
            name_hint = ydl.extract_info(url, download=False).get('title', 'song')
    base = os.path.join(OUT_DIR, clean_filename(name_hint))
    target = base + ".wav"
    if os.path.exists(target):
        print(f"  already downloaded: {target}")
        return target

    opts = get_safe_ydl_opts({
        'format': 'bestaudio/best',
        'outtmpl': base,  # extension-less: postprocessor appends .wav
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'noplaylist': True,
        'quiet': False,
        'no_warnings': False,
    })
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    if os.path.exists(target):
        print(f"  saved -> {target}")
        return target
    # find whatever landed, in case of naming quirks
    import glob
    hits = glob.glob(base + "*")
    if hits:
        print(f"  saved -> {hits[0]}")
        return hits[0]
    print("  download produced no file (check yt-dlp/cookies).")
    return None


def pick_and_download(query):
    print(f'\nSearching: "{query}" ...')
    try:
        results = search(query)
    except Exception as e:
        print(f"  search failed: {type(e).__name__}: {e}")
        return
    if not results:
        print("  no results.")
        return
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['title']}")
        print(f"      {r['uploader']}  |  {fmt_duration(r['duration'])}  |  {fmt_views(r['views'])} views")
    choice = input(f"Pick 1-{len(results)} (Enter = 1, s = skip): ").strip().lower()
    if choice == 's':
        return
    idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(results) else 0
    r = results[idx]
    print(f'Downloading: {r["title"]}')
    download(r['url'], name_hint=r['title'])


def main():
    arg = " ".join(sys.argv[1:]).strip()
    if arg.startswith("http"):
        download(arg)
        return
    if arg:
        pick_and_download(arg)
    while True:
        q = input('\nSong to search (Enter to quit): ').strip()
        if not q:
            break
        if q.startswith("http"):
            download(q)
        else:
            pick_and_download(q)


if __name__ == "__main__":
    main()
    