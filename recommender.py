import pandas as pd
import random
import os
import ast
import math
import time

# Streamlit is only needed when this runs inside the old Streamlit app (app.py).
# Under the Companion there's no streamlit -- use a no-op cache shim instead.
try:
    import streamlit as st
except ImportError:
    class _NoStreamlit:
        @staticmethod
        def cache_data(fn):
            return fn
    st = _NoStreamlit()

# Prefer the scraped popular-song pool if it exists; the big tracks.csv is fallback depth only.
CHART_CSV = "chart_tracks.csv"

def get_safe_ydl_opts(base_opts=None):
    """Generate yt_dlp options with browser headers and cookies to bypass bot detection."""
    safe = {
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 30,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        },
        'extract_flat': True,
        'no_check_certificate': True,
    }
    if base_opts:
        safe.update(base_opts)

    # Use a local exported cookie file if one exists in the project folder.
    for cookie_path in ('cookies.txt', 'youtube.cookies.txt'):
        if os.path.exists(cookie_path):
            safe['cookiefile'] = cookie_path
            break
    return safe


class MashupRecommender:
    def __init__(self, csv_path="tracks.csv"):
        self.csv_path = csv_path
        self.df = self.load_data()
        self.major_map = {0:8, 1:3, 2:10, 3:5, 4:12, 5:7, 6:2, 7:9, 8:4, 9:11, 10:6, 11:1}
        self.minor_map = {0:5, 1:12, 2:7, 3:2, 4:9, 5:4, 6:11, 7:6, 8:1, 9:8, 10:3, 11:10}

    @st.cache_data
    def load_data(_self):
        # Use the scraped CHART pool if present (it has REAL popularity); otherwise fall back to
        # the configured csv (e.g. the 1.2M audio-features dump, which has NO popularity data).
        path = CHART_CSV if (os.path.exists(CHART_CSV) and os.path.getsize(CHART_CSV) > 0) else _self.csv_path
        if not os.path.exists(path): return pd.DataFrame()
        print(f"\U0001f4c0 Recommender pool: {path}")

        df = pd.read_csv(path, low_memory=False)
        
        if 'name' in df.columns: 
            df.rename(columns={'name': 'track_name'}, inplace=True)
            
        if 'popularity' not in df.columns: 
            df['popularity'] = 80 
            
        cols =['key', 'mode', 'tempo', 'energy', 'speechiness', 'popularity', 'valence', 'year']
        for c in cols: 
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            
        if 'year' in df.columns:
            df = df[df['year'] >= 2000]
            
        if 'tempo' in df.columns:
            df = df[df['tempo'] > 50.0] 
            
        df = df.dropna(subset=['track_name', 'artists', 'tempo', 'energy', 'valence'])

        # --- FIX MOJIBAKE: repair UTF-8 text that was mangled through Latin-1
        # ("daÃ±o" -> "daño", "JacareÌ" -> "Jacaré"). Matters for YouTube queries.
        def _demojibake(s):
            if isinstance(s, str) and ('Ã' in s or 'Ì' in s or 'â' in s or 'Â' in s):
                try:
                    return s.encode('latin-1').decode('utf-8')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    return s
            return s
        for col in ('track_name', 'artists'):
            df[col] = df[col].map(_demojibake)

        # --- THE DYNAMIC FILTER & SANITIZER (UPDATED) ---
        # The Top 300 Artist filter has been removed! We now rely on track popularity.
        
        # 1. KILL THE MEGAMIXES: Drop tracks with insanely long artist lists (like 50 Deep)
        df = df[df['artists'].str.len() < 80]
        
        # 2. CLEAN THE TITLES: Remove "(feat. XYZ)" and "- Remastered" for clean UI/YouTube Search
        df['track_name'] = df['track_name'].str.replace(r' \(feat\..*?\)', '', regex=True)
        df['track_name'] = df['track_name'].str.replace(r' - .*', '', regex=True)
        
        # 3. PRIORITIZE HITS: Sort the database by popularity so the best tracks are sampled first
        df = df.sort_values(by='popularity', ascending=False)
        
        return df
    
    def check_youtube_views(self, query):
        """Silently searches YouTube and returns the video's view count"""
        import yt_dlp
        ydl_opts = get_safe_ydl_opts({'noplaylist': True})
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch1:{query}", download=False)
                if 'entries' in info and len(info['entries']) > 0:
                    views = info['entries'][0].get('view_count', 0)
                    return views if views is not None else 0
        except Exception as e:
            time.sleep(1)  # Backoff on error
        return 0

    def clean_artist(self, s):
        s = str(s).strip()
        if s.startswith("['"):
            import ast
            try: 
                # Convert string list to actual Python list, keep max 2 artists
                artists_list = ast.literal_eval(s)
                return ", ".join(artists_list[:2]) 
            except: pass
        # Fallback if eval fails
        clean_str = s.replace("['", "").replace("']", "").replace("'", "")
        return ", ".join(clean_str.split(',')[:2])

    @staticmethod
    def same_song(name_a, name_b):
        """True if two titles are the same song in disguise (remix, RMX,
        '(w/ X)' credit variant, sped up/slowed, live, etc.) -- pairing a
        song with its own alternate version is never a useful mashup."""
        import re
        def base(t):
            t = str(t).lower()
            t = re.sub(r"[\(\[].*?[\)\]]", " ", t)           # drop (…) and […] blocks
            t = re.sub(r"\b(remix|rmx|remaster(ed)?|version|edit|live|acoustic|"
                       r"sped\s*up|slowed(\s*\+?\s*reverb)?|instrumental|"
                       r"radio|extended|vip|deluxe|w)\b", " ", t)
            t = re.sub(r"[^a-z0-9]+", " ", t).strip()
            return t
        a, b = base(name_a), base(name_b)
        if not a or not b:
            return str(name_a).lower() == str(name_b).lower()
        return a == b or a in b or b in a

    def get_camelot(self, k, m):
        h = self.major_map.get(int(k)) if m == 1 else self.minor_map.get(int(k))
        return h, ('B' if m == 1 else 'A')

    def get_vibe_distance(self, row_a, row_b):
        # 1. Acoustic Penalty
        if abs(row_a.get('acousticness', 0) - row_b.get('acousticness', 0)) > 0.6:
            return 100 
        # 2. Vibe Vector
        dist = math.sqrt((row_a['energy'] - row_b['energy'])**2 + (row_a['valence'] - row_b['valence'])**2)
        return (dist / 1.41) * 30

    def discover_mashups(self, n=5, exclude=None):
        """exclude: set of pair_id tuples (sorted track-name pairs) to skip,
        so repeated rolls in one session don't show the same pairs."""
        exclude = exclude or set()
        if self.df.empty: return []
        
        # --- POOLS ---
        df_voc = self.df[(self.df['speechiness'] > 0.04) & (self.df['popularity'] > 45)]
        df_beat = self.df[(self.df['energy'] > 0.55) & (self.df['popularity'] > 45)]
        
        # 1. Guarantee highly popular tracks get sampled into our pools
        df_voc_pop = df_voc[df_voc['popularity'] >= 75]
        df_beat_pop = df_beat[df_beat['popularity'] >= 75]
        
        pool_a = pd.concat([
            df_voc.sample(n=min(100, len(df_voc)), replace=False),
            df_voc_pop.sample(n=min(50, len(df_voc_pop)), replace=False)
        ]).drop_duplicates()
        
        pool_b = pd.concat([
            df_beat.sample(n=min(100, len(df_beat)), replace=False),
            df_beat_pop.sample(n=min(50, len(df_beat_pop)), replace=False)
        ]).drop_duplicates()
        
        candidates = []
        
        for _, a in pool_a.iterrows():
            for _, b in pool_b.iterrows():
                if self.same_song(a['track_name'], b['track_name']): continue
                
                # --- ULTRA-SAFE PRE-FILTER ---
                # Keep only near-1:1 tempo matches to reduce downstream QC warnings.
                ratio = b['tempo'] / a['tempo']
                is_standard_safe = 0.95 <= ratio <= 1.05

                if not is_standard_safe:
                    continue

                # Reject pairs that need large semitone moves based on metadata keys.
                try:
                    key_a = int(a['key'])
                    key_b = int(b['key'])
                except Exception:
                    continue

                semitone_shift = ((key_a - key_b + 6) % 12) - 6
                if abs(semitone_shift) > 1:
                    continue

                # --- STRICT HARMONIC LIMIT (±1 Semitone) ---
                best_harm = 0
                for shift in range(-1, 2): 
                    kb_shift = (int(b['key']) + shift) % 12
                    ha, _ = self.get_camelot(a['key'], a['mode'])
                    hb, _ = self.get_camelot(kb_shift, b['mode'])
                    if ha is None or hb is None:
                        continue
                    dist = min(abs(ha - hb), 12 - abs(ha - hb))
                    
                    s = 100 if dist == 0 else (85 if dist == 1 else 0)
                    s -= abs(shift) * 10 
                    best_harm = max(best_harm, s)
                
                if best_harm < 70: continue
                
                # Vibe Check
                vibe_pen = self.get_vibe_distance(a, b)
                final = best_harm - vibe_pen
                
                if final > 75:
                    candidates.append({'score': final, 'a': a, 'b': b})

        # --- SELECTION ---
        # Chart rank / dataset popularity IS our popularity signal, so we no longer make a
        # live YouTube view-count call per candidate (that was the slow, rate-limit-prone part).
        # Rank by match score blended with combined popularity, then shuffle the top slice so
        # "Roll New Pairs" feels fresh between rolls.
        for c in candidates:
            pa = float(c['a'].get('popularity', 50) or 50)
            pb = float(c['b'].get('popularity', 50) or 50)
            c['rank_score'] = c['score'] + 0.25 * ((pa + pb) / 2.0)   # weight popularity lightly

        candidates.sort(key=lambda x: x['rank_score'], reverse=True)

        # A wide slice keeps quality high while giving rolls real variety
        # (the old n*3 slice made every roll draw from the same ~12 pairs).
        top_slice = candidates[: max(n * 10, 40)]
        random.shuffle(top_slice)

        final_list = []
        used = set()
        for c in top_slice:
            if len(final_list) >= n:
                break
            ta, tb = c['a']['track_name'], c['b']['track_name']
            aa, bb = self.clean_artist(c['a']['artists']), self.clean_artist(c['b']['artists'])
            pair_id = tuple(sorted([ta, tb]))
            if pair_id in used or pair_id in exclude:
                continue
            used.add(pair_id)
            pop = int((float(c['a'].get('popularity', 50) or 50) +
                       float(c['b'].get('popularity', 50) or 50)) / 2)
            # Fire = both songs are highly popular (avg pop >= 75, the dataset's "high-pop" bar).
            # Driven by the popularity field now, NOT a live YouTube view lookup. Lower 75 to see it more often.
            emoji = "\U0001f525" if pop >= 75 else "\U0001f3b5"
            final_list.append({
                "name": f"{aa} - {ta} x {bb} - {tb}",
                "description": f"{emoji} Match: {int(c['score'])}% | Pop: {pop} | {int(c['a']['tempo'])}\u2194{int(c['b']['tempo'])} BPM",
                "song_a": {"title": ta, "search_query": f"{aa} {ta} audio"},
                "song_b": {"title": tb, "search_query": f"{bb} {tb} audio"}
            })

        return final_list
    