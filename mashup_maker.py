import collections
import collections.abc
import sys
import numpy as np

# PATCH
if not hasattr(collections, 'MutableSequence'): collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(collections, 'Iterable'): collections.Iterable = collections.abc.Iterable
if not hasattr(collections, 'Callable'): collections.Callable = collections.abc.Callable
sys.modules['collections'].MutableSequence = collections.abc.MutableSequence
sys.modules['collections'].Iterable = collections.abc.Iterable
sys.modules['collections'].Callable = collections.abc.Callable
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
# -----

import os
import hashlib
import soundfile as sf
import yt_dlp
import librosa
from scipy.signal import savgol_filter
from audio_separator.separator import Separator
from pedalboard import Pedalboard, Compressor, HighpassFilter, Limiter, Gain
from scipy.signal import savgol_filter, butter, filtfilt

# MODELS
try:
    from BeatNet.BeatNet import BeatNet
    HAS_BEATNET = True
except: HAS_BEATNET = False

try:
    import pyrubberband as pyrb
    HAS_RUBBERBAND = True
except: HAS_RUBBERBAND = False

# UTILS
def analyze_structure_advanced(path):
    print(f"🧠 Analyzing: {os.path.basename(path)}...")
    bpm, anchor_point = 128.0, 0.0
    beat_times = []
    
    if HAS_BEATNET:
        try:
            estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
            output = estimator.process(path)
            if output is not None and len(output) > 0:
                beat_times = output[:, 0]
                intervals = np.diff(beat_times)
                if len(intervals) > 0: bpm = 60.0 / np.median(intervals)
                downbeats = output[output[:, 1] == 1][:, 0]
                valid = [d for d in downbeats if d > 10.0]
                anchor_point = valid[0] if valid else (downbeats[0] if len(downbeats) > 0 else 0.0)
                return {"bpm": bpm, "anchor_point": anchor_point, "beat_times": beat_times, "segments": []}
        except: pass

    y, sr = librosa.load(path, sr=44100, duration=120)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = tempo if np.isscalar(tempo) else tempo[0]
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    peak_time = librosa.frames_to_time(np.argmax(savgol_filter(rms, 21, 3)), sr=sr)
    
    if len(beat_times) > 0: anchor_point = beat_times[(np.abs(beat_times - peak_time)).argmin()]
    else: anchor_point = peak_time

    return {"bpm": bpm, "anchor_point": anchor_point, "beat_times": beat_times, "segments": []}

def _file_hash(path, _bufsize=1 << 20):
    """Stable content hash of a file, so identical audio reuses the same cache."""
    import hashlib as _h
    h = _h.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(_bufsize), b''):
            h.update(chunk)
    return h.hexdigest()[:12]

def _log_execution_provider():
    """Loudly report whether ONNX is on the GPU. A silent CPU fallback is the
    usual reason a machine with a GPU still takes 5-10 min to separate."""
    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            print("\u2705 onnxruntime: CUDA (GPU) execution available.")
        else:
            print("\u26a0\ufe0f  onnxruntime: GPU NOT available -> running on CPU. "
                  "This is the usual cause of very slow separation. Install "
                  "'onnxruntime-gpu' + matching CUDA to use your GTX 1650.")
    except Exception as e:
        print(f"(could not check onnxruntime providers: {e})")

def separate_full_song(source_path, output_dir):
    """Separate a FULL source song ONCE and cache stems by content hash.
    The same songs recur across many rolled pairs, so every repeat is now an
    instant cache hit instead of a fresh, minutes-long separation. We separate
    the whole track (not an anchor-specific chunk) so the result is reusable
    for any anchor / any mashup that uses this song."""
    # --- CACHE KEY = content of the SOURCE song (anchor-independent, reusable) ---
    try:
        key = _file_hash(source_path)
    except Exception:
        key = os.path.basename(source_path).replace('.wav', '')
    stem_dir = os.path.join(output_dir, "stems_4", key)
    os.makedirs(stem_dir, exist_ok=True)

    required = ["vocals", "drums", "bass", "other"]
    existing = os.listdir(stem_dir)
    found = {k: next((f for f in existing if k in f.lower()), None) for k in required}
    if all(found.values()):
        print(f"\u26a1 Stem cache HIT: {os.path.basename(source_path)} (skipping separation)")
        return {k: os.path.join(stem_dir, v) for k, v in found.items()}

    print(f"\U0001f9e0 Separating (cache miss): {os.path.basename(source_path)}")
    _log_execution_provider()
    try:
        # 'UVR-MDX-NET-Inst_HQ_3.onnx' is an MDX model, so MDX params apply -- NOT the
        # demucs_params the old code passed (which were silently ignored). Lower 'overlap'
        # = faster with negligible quality change; batch_size=1 stays within 4 GB VRAM.
        # NOTE: exact param NAMES depend on your audio-separator version -- VERIFY against
        # your installed version (older releases used a flatter kwargs style).
        sep = Separator(
            output_dir=stem_dir,
            model_file_dir=os.path.join(output_dir, "models"),
            mdx_params={
                "hop_length": 1024,
                "segment_size": 256,
                "overlap": 0.10,
                "batch_size": 1,
                "enable_denoise": False,
            },
        )

        model_name = 'UVR-MDX-NET-Inst_HQ_3.onnx'  # your 2-stem ONNX model
        sep.load_model(model_filename=model_name)

        output = sep.separate(source_path)
        res = {}
        for f in output:
            path = os.path.join(stem_dir, f)
            l = f.lower()
            if "vocals" in l: res["vocals"] = path
            elif "drums" in l: res["drums"] = path
            elif "bass" in l: res["bass"] = path
            elif "other" in l: res["other"] = path
            elif "instrumental" in l: res["instrumental"] = path

        # 2-stem model (vocals + instrumental): map 'instrumental' into the 4-stem layout.
        # bass/other point at instrumental but are silenced at load time so it
        # isn't played three times.
        if "instrumental" in res and "drums" not in res:
            res["drums"] = res["instrumental"]
            res["bass"] = res["instrumental"]
            res["other"] = res["instrumental"]

        return res
    except Exception as e:
        print(f"Separation Error: {e}")
        return None

# Back-compat alias: anything still calling separate_stems_4 keeps working.
# (It now separates + caches the full song rather than a per-anchor chunk.)
separate_stems_4 = separate_full_song

def process_audio(y, rate, shift_semitones, sr):
    if y.ndim == 1: y = np.vstack([y, y])
    if abs(rate - 1.0) > 0.01:
        if HAS_RUBBERBAND:
            try: y = pyrb.time_stretch(y.T, sr, rate).T
            except: y = librosa.effects.time_stretch(y, rate=rate)
        else: y = librosa.effects.time_stretch(y, rate=rate)
    if shift_semitones != 0:
        if HAS_RUBBERBAND:
            try: y = pyrb.pitch_shift(y.T, sr, shift_semitones).T
            except: y = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift_semitones)
        else: y = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift_semitones)
    return y

def normalize_stem(y, target_db=-20.0):
    if y is None: return None
    rms = np.sqrt(np.mean(y**2))
    if rms < 1e-5: return y
    current_db = 20 * np.log10(rms)
    gain_linear = 10 ** ((target_db - current_db) / 20)
    return y * gain_linear

# --- NEW: DYNAMIC ENVELOPE GENERATOR ---
def create_dynamic_mask(vocals, sr, duck_amount=0.3):
    """
    Creates a 'Sidechain' curve.
    When vocals are loud, the curve goes down.
    When vocals are silent, the curve goes up.
    duck_amount: 0.0 to 1.0 (How much volume to remove)
    """
    if vocals is None: return 1.0
    
    # 1. Calculate Envelope (RMS)
    # Use a hop length of 512 for resolution
    hop_length = 512
    frame_len = 1024
    
    # Convert to mono for analysis
    v_mono = np.mean(vocals, axis=0)
    
    # Calculate RMS energy per frame
    rms = librosa.feature.rms(y=v_mono, frame_length=frame_len, hop_length=hop_length)[0]
    
    # Smooth the envelope (Attack/Release simulation)
    # We smooth heavily so it doesn't jitter
    envelope = savgol_filter(rms, 51, 3)
    
    # Normalize envelope 0.0 to 1.0
    if np.max(envelope) > 0:
        envelope = envelope / np.max(envelope)
    
    # Invert to create Ducking Curve
    # Logic: 1.0 - (Envelope * DuckAmount)
    # If envelope is 1.0 (Loud), output is 0.7 (Ducked)
    # If envelope is 0.0 (Silent), output is 1.0 (Full)
    ducking_curve = 1.0 - (envelope * duck_amount)
    
    # Interpolate back to sample rate length
    ducking_curve_full = np.interp(
        np.arange(0, vocals.shape[1]),
        np.arange(0, len(ducking_curve)) * hop_length,
        ducking_curve
    )
    
    return ducking_curve_full

# --- NEW: SURGICAL EQ ENGINE ---
def apply_dynamic_eq(stem, mask, sr, low_freq=500.0, high_freq=4000.0):
    """
    Splits the audio into 'Vocal Frequencies' and 'Everything Else'.
    Applies the ducking mask ONLY to the vocal frequencies.
    """
    nyq = 0.5 * sr
    low = low_freq / nyq
    high = high_freq / nyq
    
    # 1. Isolate the Vocal Range (Mid-band)
    b_band, a_band = butter(4, [low, high], btype='band')
    mid_band = filtfilt(b_band, a_band, stem, axis=-1)
    
    # 2. Isolate the Rest (Sub-bass & High-end air)
    b_stop, a_stop = butter(4, [low, high], btype='bandstop')
    outside_band = filtfilt(b_stop, a_stop, stem, axis=-1)
    
    # 3. Duck ONLY the mid-band when vocals are singing
    ducked_mid = mid_band * mask
    
    # 4. Glue them back together perfectly
    return outside_band + ducked_mid

def mix_multistem(stems_dict, sr):
    mixed = None
    max_len = 0
    # 1. Calc Max Length
    for s in stems_dict.values():
        if s is not None:
            if s.ndim == 1: s = s[np.newaxis, :]
            max_len = max(max_len, s.shape[1])
            
    # 2. Generate Dynamic Mask (Sidechain)
    # We generate this ONCE based on the vocals
    dynamic_mask = None
    if 'vocals' in stems_dict:
        # Pad vocals first for analysis
        v_temp = stems_dict['vocals']
        if v_temp.ndim == 1: v_temp = v_temp[np.newaxis, :]
        pad_len = max_len - v_temp.shape[1]
        if pad_len > 0: v_temp = np.pad(v_temp, ((0,0), (0, pad_len)))
        
        # Create mask (Ducks by 50% when vocals are active - pushed higher because it's EQ now!)
        dynamic_mask = create_dynamic_mask(v_temp, sr, duck_amount=0.50)

    # 3. Mix
    for name, s in stems_dict.items():
        if s is None: continue
        if s.ndim == 1: s = s[np.newaxis, :]
        s_pad = np.pad(s, ((0,0), (0, max_len - s.shape[1]))) if s.shape[1] < max_len else s
        
        if name == 'vocals':
             s_pad = normalize_stem(s_pad, target_db=-14.0) 
             s_pad = Pedalboard([HighpassFilter(150), Compressor(-20, 3.0), Gain(0)])(s_pad, sr)
        else:
             s_pad = normalize_stem(s_pad, target_db=-22.0)
             
             # --- APPLY SURGICAL DYNAMIC EQ (NEW) ---
             # We carve out ONLY the vocal frequencies (500Hz-4kHz) in the Bass and Melodies.
             # The kick drum, sub-bass, and high-hats stay at 100% energy!
             if dynamic_mask is not None and name in ['bass', 'other']:
                 s_pad = apply_dynamic_eq(s_pad, dynamic_mask, sr)
             # --------------------------

             if name == 'bass': 
                s_mono = np.mean(s_pad, axis=0)
                s_pad = np.vstack([s_mono, s_mono])
             elif name == 'drums': 
                s_pad = Pedalboard([Compressor(-10, 2.5)])(s_pad, sr)
        
        if mixed is None: mixed = s_pad
        else: mixed += s_pad
        
    return Pedalboard([Limiter(-1.0)])(mixed, sr)

def get_smart_rate(bpm_source, bpm_target):
    raw_rate = bpm_target / bpm_source
    if 1.9 <= raw_rate <= 2.1: return 1.0
    if 0.45 <= raw_rate <= 0.55: return 1.0
    return raw_rate

def _load_slice_stems(stem_paths, sr, anchor, bpm, bars=96, dummy_keys=()):
    """Load cached FULL-length stems and slice them to a ~`bars`-bar window around
    the anchor (drop). This mirrors the old prepare_micro_chunk window exactly, so
    the downstream mixing math is unchanged -- we've only moved the separation out
    in front and cached it. Returns (stems_dict, window_start_sec)."""
    sec_per_beat = 60.0 / bpm
    chunk_duration = sec_per_beat * 4 * bars          # 4/4 assumed, same as before
    out = {}
    start_sec = None
    s0 = s1 = 0
    for k in ('vocals', 'drums', 'bass', 'other'):
        y, _ = librosa.load(stem_paths[k], sr=sr, mono=False)
        if y.ndim == 1:
            y = np.vstack([y, y])
        if start_sec is None:                         # compute the window once
            total = y.shape[-1] / sr
            start_sec = max(0.0, anchor - chunk_duration * 0.25)   # 8 bars before drop
            end_sec = min(total, start_sec + chunk_duration)
            s0, s1 = int(start_sec * sr), int(end_sec * sr)
        seg = y[:, s0:s1]
        if k in dummy_keys:
            seg = np.zeros_like(seg)                   # silence dummies (was is_dummy=True)
        out[k] = seg
    return out, start_sec

def generate_mashup_set(output_dir, data_a, data_b, shift_semitones=0, selected_styles=[]):
    print(f"🚀 Styles Requested: {selected_styles}") 
    sr_a, sr_b = data_a['sr'], data_b['sr']
    
    # 1. Separate each FULL source song ONCE (cached by content hash). The same
    #    songs recur across rolled pairs, so repeats become instant cache hits.
    full_a = separate_full_song(data_a['path'], output_dir)
    full_b = separate_full_song(data_b['path'], output_dir)
    if not full_a or not full_b: return {}

    # 2. Load cached full stems and SLICE to the ~3-min window around the drop
    #    (96 bars). Same per-mashup compute as before, but separation is reused.
    S_A, offset_a = _load_slice_stems(full_a, sr_a, data_a['anchor_point'], data_a['bpm'],
                                      bars=96, dummy_keys=('bass', 'other'))
    S_B, offset_b = _load_slice_stems(full_b, sr_b, data_b['anchor_point'], data_b['bpm'],
                                      bars=96, dummy_keys=('bass', 'other'))

    # 3. Make anchor points relative to the sliced window (same as before)
    data_a['anchor_point'] = max(0.0, data_a['anchor_point'] - offset_a)
    data_b['anchor_point'] = max(0.0, data_b['anchor_point'] - offset_b)

    bpm_a, bpm_b = data_a['bpm'], data_b['bpm']
    rate_a_to_b = get_smart_rate(bpm_a, bpm_b)
    rate_b_to_a = get_smart_rate(bpm_b, bpm_a)
    
    anc_a, anc_b = int(data_a['anchor_point'] * sr_a), int(data_b['anchor_point'] * sr_b)
    output_files = {}

    # --- NEW: CACHING SYSTEM TO PREVENT REDUNDANT PROCESSING ---
    processed_cache = {}
    def get_processed(stem, rate, shift, sr):
        # Create a unique key for this exact operation
        cache_key = (id(stem), rate, shift, sr)
        if cache_key not in processed_cache:
            processed_cache[cache_key] = process_audio(stem, rate, shift, sr)
        return processed_cache[cache_key]

    if any(x in selected_styles for x in["Standard (A on B)"]):
        v_a = get_processed(S_A['vocals'], rate_a_to_b, shift_semitones, sr_b)
        off = anc_b - int(anc_a * rate_a_to_b)
        len_b = S_B['drums'].shape[1]
        v_al = np.zeros((2, len_b))
        if off >= 0:
            l = min(v_a.shape[1], len_b - off)
            v_al[:, off:off+l] = v_a[:, :l]
        else:
            s = -off
            l = min(v_a.shape[1] - s, len_b)
            v_al[:, :l] = v_a[:, s:s+l]

        mix = mix_multistem({'vocals': v_al, 'drums': S_B['drums'], 'bass': S_B['bass'], 'other': S_B['other']}, sr_b)
        p = os.path.join(output_dir, "01_Standard.wav")
        sf.write(p, mix.T, sr_b)
        output_files["Standard (A on B)"] = p

    if any(x in selected_styles for x in ["The Flip (B on A)"]):
        v_b = get_processed(S_B['vocals'], rate_b_to_a, -shift_semitones, sr_a)
        off = anc_a - int(anc_b * rate_b_to_a)
        len_a = S_A['drums'].shape[1]
        v_bl = np.zeros((2, len_a))
        if off >= 0:
            l = min(v_b.shape[1], len_a - off)
            v_bl[:, off:off+l] = v_b[:, :l]
        else:
            s = -off
            l = min(v_b.shape[1] - s, len_a)
            v_bl[:, :l] = v_b[:, s:s+l]

        mix = mix_multistem({'vocals': v_bl, 'drums': S_A['drums'], 'bass': S_A['bass'], 'other': S_A['other']}, sr_a)
        p = os.path.join(output_dir, "02_Flip.wav")
        sf.write(p, mix.T, sr_a)
        output_files["The Flip (B on A)"] = p

    if any(x in selected_styles for x in ["The Magic Switch"]):
        # 1. Get BOTH vocals. 
        # Shift Vocal A to match B. Vocal B is already native to B, so no shift needed.
        v_a = get_processed(S_A['vocals'], rate_a_to_b, shift_semitones, sr_b)
        v_b = get_processed(S_B['vocals'], 1.0, 0, sr_b) 
        
        len_b = S_B['drums'].shape[1]
        
        # Helper function to align stems to the anchor point
        def align_stem(stem, anc_src, rate_src, anc_tgt, tgt_len):
            off = anc_tgt - int(anc_src * rate_src)
            out = np.zeros((2, tgt_len))
            if off >= 0:
                l = min(stem.shape[1], tgt_len - off)
                out[:, off:off+l] = stem[:, :l]
            else:
                s = -off
                l = min(stem.shape[1] - s, tgt_len)
                out[:, :l] = stem[:, s:s+l]
            return out

        # 2. Align both vocals to the beat drop
        v_al = align_stem(v_a, anc_a, rate_a_to_b, anc_b, len_b)
        v_bl = align_stem(v_b, anc_b, 1.0, anc_b, len_b)
        
        # 3. Calculate exactly how long 4 musical bars are in audio samples
        # 1 Beat = (60 / BPM) seconds. 4 Bars = 16 Beats.
        samples_per_beat = sr_b * (60.0 / bpm_b)
        samples_per_4_bars = int(samples_per_beat * 16)
        
        # 4. Create the "Crossfader" Mask
        mask_a = np.zeros((2, len_b))
        
        # Alternate every 4 bars: 1 is Vocal A, 0 is Vocal B
        for i in range(0, len_b, samples_per_4_bars * 2):
            end_a = min(i + samples_per_4_bars, len_b)
            mask_a[:, i:end_a] = 1.0 
            
        mask_b = 1.0 - mask_a # The exact opposite of Mask A
        
        # 5. Splicing magic: Multiply the vocals by the masks to interleave them
        v_interleaved = (v_al * mask_a) + (v_bl * mask_b)
        
        # Mix the interleaved vocals with Beat B
        mix = mix_multistem({'vocals': v_interleaved, 'drums': S_B['drums'], 'bass': S_B['bass'], 'other': S_B['other']}, sr_b)
        p = os.path.join(output_dir, "06_Magic_Switch.wav")
        sf.write(p, mix.T, sr_b)
        output_files["The Magic Switch"] = p

    if any(x in selected_styles for x in ["Rhythm Swap"]):
        v_a = get_processed(S_A['vocals'], rate_a_to_b, shift_semitones, sr_b)
        d_a = get_processed(S_A['drums'], rate_a_to_b, shift_semitones, sr_b)
        off = anc_b - int(anc_a * rate_a_to_b)
        len_b = S_B['drums'].shape[1]
        def align(stem):
            out = np.zeros((2, len_b))
            if off >= 0:
                l = min(stem.shape[1], len_b - off)
                out[:, off:off+l] = stem[:, :l]
            else:
                s = -off
                l = min(stem.shape[1] - s, len_b)
                out[:, :l] = stem[:, s:s+l]
            return out
        mix = mix_multistem({'vocals': align(v_a), 'drums': align(d_a), 'bass': S_B['bass'], 'other': S_B['other']}, sr_b)
        p = os.path.join(output_dir, "03_Swap.wav")
        sf.write(p, mix.T, sr_b)
        output_files["Rhythm Swap"] = p
        
    if any(x in selected_styles for x in ["Deep Mode"]):
        v_a = get_processed(S_A['vocals'], rate_a_to_b, 0, sr_b) 
        inv = -shift_semitones
        d_b = get_processed(S_B['drums'], 1.0, inv, sr_b)
        b_b = get_processed(S_B['bass'], 1.0, inv, sr_b)
        o_b = get_processed(S_B['other'], 1.0, inv, sr_b)
        off = anc_b - int(anc_a * rate_a_to_b)
        len_b = d_b.shape[1]
        v_al = np.zeros((2, len_b))
        if off >= 0:
            l = min(v_a.shape[1], len_b - off)
            v_al[:, off:off+l] = v_a[:, :l]
        else:
            s = -off
            l = min(v_a.shape[1] - s, len_b)
            v_al[:, :l] = v_a[:, s:s+l]
        mix = mix_multistem({'vocals': v_al, 'drums': d_b, 'bass': b_b, 'other': o_b}, sr_b)
        p = os.path.join(output_dir, "04_Deep.wav")
        sf.write(p, mix.T, sr_b)
        output_files["Deep Mode"] = p

    if any(x in selected_styles for x in ["Hype Mode"]):
        v_a = get_processed(S_A['vocals'], rate_a_to_b, shift_semitones + 12, sr_b)
        off = anc_b - int(anc_a * rate_a_to_b)
        len_b = S_B['drums'].shape[1]
        v_al = np.zeros((2, len_b))
        if off >= 0:
            l = min(v_a.shape[1], len_b - off)
            v_al[:, off:off+l] = v_a[:, :l]
        else:
            s = -off
            l = min(v_a.shape[1] - s, len_b)
            v_al[:, :l] = v_a[:, s:s+l]
        mix = mix_multistem({'vocals': v_al, 'drums': S_B['drums'], 'bass': S_B['bass'], 'other': S_B['other']}, sr_b)
        p = os.path.join(output_dir, "05_Hype.wav")
        sf.write(p, mix.T, sr_b)
        output_files["Hype Mode"] = p

    return output_files

def prepare_micro_chunk(data_dict, bars=32):
    """
    Slices the audio to only process the specific bars needed around the anchor.
    """
    input_path = data_dict['path']
    bpm = data_dict['bpm']
    anchor = data_dict['anchor_point']
    
    # Load original audio
    y, sr = librosa.load(input_path, sr=None, mono=False)
    
    # Calculate duration of 32 bars in seconds
    sec_per_beat = 60.0 / bpm
    chunk_duration = sec_per_beat * 4 * bars # Assuming 4/4 time signature
    
    # Crop around the anchor point (drop/chorus)
    start_sec = max(0.0, anchor - (chunk_duration * 0.25)) # 8 bars before drop
    end_sec = min(y.shape[-1] / sr if y.ndim > 1 else len(y) / sr, start_sec + chunk_duration)
    
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    
    if y.ndim > 1:
        chunk_y = y[:, start_sample:end_sample].T # Transpose for soundfile
    else:
        chunk_y = y[start_sample:end_sample]
        
    chunk_path = input_path.replace('.wav', '_chunk.wav')
    sf.write(chunk_path, chunk_y, sr)
    
    return chunk_path, start_sec # We return start_sec so we can offset the final mix later
