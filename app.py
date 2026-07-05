import collections
import collections.abc
import sys
import streamlit as st
import os
import time
import traceback
import yt_dlp
import librosa
from mashup_maker import analyze_structure_advanced, generate_mashup_set
from key_finder import detect_key, get_pitch_shift_steps
from recommender import MashupRecommender

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

# PATCH
if not hasattr(collections, 'MutableSequence'): collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(collections, 'Iterable'): collections.Iterable = collections.abc.Iterable
if not hasattr(collections, 'Callable'): collections.Callable = collections.abc.Callable
sys.modules['collections'].MutableSequence = collections.abc.MutableSequence
sys.modules['collections'].Iterable = collections.abc.Iterable
sys.modules['collections'].Callable = collections.abc.Callable

st.set_page_config(page_title="Magic Switch: Pro", layout="wide", page_icon="🎛️")
WORK_DIR = "downloads"
os.makedirs(WORK_DIR, exist_ok=True)

st.markdown("""
    <style>
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; }
    .warning-box { border: 1px solid #ff4b4b; background-color: #ff4b4b22; padding: 10px; border-radius: 5px; color: #ff4b4b; }
    .success-box { border: 1px solid #09ab3b; background-color: #09ab3b22; padding: 10px; border-radius: 5px; color: #09ab3b; }
    </style>
""", unsafe_allow_html=True)

if 'song_a_data' not in st.session_state: st.session_state.song_a_data = None
if 'song_b_data' not in st.session_state: st.session_state.song_b_data = None
if 'selected_url_a' not in st.session_state: st.session_state.selected_url_a = ""
if 'selected_url_b' not in st.session_state: st.session_state.selected_url_b = ""

rec_engine = MashupRecommender("tracks.csv")

def download_track(url, label):
    import hashlib
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:8]
    filename_base = os.path.join(WORK_DIR, f"{label}_{url_hash}")
    target_path = filename_base + ".wav"
    if os.path.exists(target_path): return target_path
    
    ydl_opts = get_safe_ydl_opts({
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
        'outtmpl': filename_base,
        'noplaylist': True
    })
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try: ydl.download([url]); return target_path
        except: return None

def analyze_track(path):
    y, sr = librosa.load(path, sr=44100, duration=120) 
    analysis = analyze_structure_advanced(path)
    return {"path": path, "y": y, "sr": sr, "bpm": analysis['bpm'], "key": detect_key(y, sr), "anchor_point": analysis['anchor_point']}

def auto_process_pair(url_a, url_b, title_a, title_b):
    try:
        start_total = time.time()
        with st.status("🚀 Launching Engine...", expanded=True) as status:
            # --- TRACK A ---
            t0 = time.time()
            status.write(f"⬇️ Downloading: {title_a}...")
            path_a = download_track(url_a, "A")
            t1 = time.time()
            status.write(f"⏱️ *Download A took: {t1-t0:.2f}s*")
            
            status.write(f"🧠 Analyzing: {title_a}...")
            st.session_state.song_a_data = analyze_track(path_a)
            t2 = time.time()
            status.write(f"⏱️ *Analysis A took: {t2-t1:.2f}s*")
            
            # --- TRACK B ---
            status.write(f"⬇️ Downloading: {title_b}...")
            path_b = download_track(url_b, "B")
            t3 = time.time()
            status.write(f"⏱️ *Download B took: {t3-t2:.2f}s*")
            
            status.write(f"🧠 Analyzing: {title_b}...")
            st.session_state.song_b_data = analyze_track(path_b)
            t4 = time.time()
            status.write(f"⏱️ *Analysis B took: {t4-t3:.2f}s*")
            
            total_prep = t4 - start_total
            status.update(label=f"✅ Ready! (Total Prep: {total_prep:.2f}s)", state="complete", expanded=False)
            
        time.sleep(0.5); st.rerun()
    except Exception as e: st.error(f"Error: {e}")

def search_youtube(query):
    try:
        ydl_opts = get_safe_ydl_opts({'extract_flat': True, 'dump_single_json': True, 'noplaylist': True})
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
            if 'entries' in info and info['entries']: return info['entries'][0]['url']
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("🧬 DNA Matcher")
    if st.button("Roll New Pairs (Strict Mode)", type="primary"):
        st.session_state.discovery_cache = rec_engine.discover_mashups(5)
    
    if 'discovery_cache' in st.session_state:
        for i, c in enumerate(st.session_state.discovery_cache):
            st.markdown(f"**{c['name']}**")
            st.caption(c['description'])
            if st.button("Load Pair", key=f"load_{i}"):
                with st.spinner("Resolving..."):
                    u_a = search_youtube(c['song_a']['search_query'])
                    u_b = search_youtube(c['song_b']['search_query'])
                    if u_a and u_b: auto_process_pair(u_a, u_b, c['song_a']['title'], c['song_b']['title'])
            st.divider()

# --- MAIN UI ---
st.title("🎛️ Magic Switch: Pro")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Source A (Vocals)")
    if not st.session_state.song_a_data:
        u_a = st.text_input("YouTube URL A")
        if u_a and st.button("Analyze A"): auto_process_pair(u_a, st.session_state.selected_url_b or "", "Song A", "Song B")
    else:
        d = st.session_state.song_a_data
        st.success(f"{d['key']} | {d['bpm']:.1f} BPM")
        st.markdown("**🎵 Original Audio A**")
        st.audio(d['path'])
        d['anchor_point'] = st.slider("Anchor A", 0.0, 120.0, float(d['anchor_point']))

with col2:
    st.subheader("Source B (Beat)")
    if not st.session_state.song_b_data:
        u_b = st.text_input("YouTube URL B")
    else:
        d = st.session_state.song_b_data
        st.success(f"{d['key']} | {d['bpm']:.1f} BPM")
        st.markdown("**🎵 Original Audio B**")
        st.audio(d['path'])
        d['anchor_point'] = st.slider("Anchor B", 0.0, 120.0, float(d['anchor_point']))

if st.session_state.song_a_data and st.session_state.song_b_data:
    st.divider()
    st.subheader("⚗️ Quality Control & Generation")
    
    # --- INTELLIGENT QC LOGIC ---
    shift = get_pitch_shift_steps(st.session_state.song_a_data['key'], st.session_state.song_b_data['key'])
    bpm_a, bpm_b = st.session_state.song_a_data['bpm'], st.session_state.song_b_data['bpm']
    ratio = bpm_b / bpm_a
    
    # 1. Pitch Check
    bad_pitch = abs(shift) > 2
    
    # 2. Time Check (Smart)
    # Allows: Standard (0.85-1.15), Double Time (1.8-2.2), Half Time (0.45-0.55)
    is_standard = 0.85 <= ratio <= 1.15
    is_double = 1.8 <= ratio <= 2.2
    is_half = 0.45 <= ratio <= 0.55
    
    bad_time = not (is_standard or is_double or is_half)
    
    c1, c2 = st.columns(2)
    with c1:
        if bad_pitch: 
            st.markdown(f"<div class='warning-box'>⚠️ <b>High Pitch Shift: {shift:+} semitones</b><br>Vocals will sound unnatural.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='success-box'>✅ <b>Perfect Harmonic Match</b><br>Shift: {shift:+} semitones</div>", unsafe_allow_html=True)
            
    with c2:
        if bad_time:
             st.markdown(f"<div class='warning-box'>⚠️ <b>High Time Stretch: {int(ratio*100)}%</b><br>Audio artifacts likely.</div>", unsafe_allow_html=True)
        elif is_double:
             st.markdown(f"<div class='success-box'>🔥 <b>Double Time Match!</b><br>Trap/DnB Mode (200% Speed)</div>", unsafe_allow_html=True)
        elif is_half:
             st.markdown(f"<div class='success-box'>🔥 <b>Half Time Match!</b><br>Chill Mode (50% Speed)</div>", unsafe_allow_html=True)
        else:
             st.markdown(f"<div class='success-box'>✅ <b>Clean Rhythm Match</b><br>Speed Change: {int(ratio*100)}%</div>", unsafe_allow_html=True)

    # Style Selection
    styles = {
        "Standard (A on B)": "Classic Mix", 
        "The Flip (B on A)": "Reverse Mix", 
        "The Magic Switch": "Trading 4-Bars (Interleaved)",
        "Rhythm Swap": "Hybrid Beat",
        "Deep Mode": "Instrumental Shift (Safe)",
        "Hype Mode": "Octave Shift (Creative)"
    }
    
    sel = st.multiselect("Select Output Styles:", options=list(styles.keys()), default=["Standard (A on B)", "The Flip (B on A)"])
    
    if st.button("Generate Mashups", type="primary"):
        start_gen = time.time() # Start the master clock
        
        with st.status("🎛️ Mixing (AI Separation in Progress)...") as status:
            res = generate_mashup_set(WORK_DIR, st.session_state.song_a_data, st.session_state.song_b_data, shift, sel)
            
            for n, p in res.items():
                st.write(f"✅ {n}")
                st.audio(p, format='audio/wav')
                with open(p, 'rb') as f: st.download_button(f"⬇️ {n}", f, file_name=os.path.basename(p))
                
            end_gen = time.time() # Stop the master clock
            total_gen_time = end_gen - start_gen
            
            status.update(label=f"✨ Complete! (Total AI & Mix Time: {total_gen_time:.2f}s)", state="complete")
            