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
import soundfile as sf
import yt_dlp
import librosa
from scipy.signal import savgol_filter
from audio_separator.separator import Separator
from pedalboard import Pedalboard, Compressor, HighpassFilter, Limiter, Gain

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

def separate_stems_4(input_path, output_dir):
    filename = os.path.basename(input_path).replace('.wav', '')
    stem_dir = os.path.join(output_dir, "stems_4", filename)
    os.makedirs(stem_dir, exist_ok=True)
    
    required = ["vocals", "drums", "bass", "other"]
    existing = os.listdir(stem_dir)
    found = {k: next((f for f in existing if k in f), None) for k in required}
    if all(found.values()): return {k: os.path.join(stem_dir, v) for k, v in found.items()}

    try:
        sep = Separator(output_dir=stem_dir, model_file_dir=os.path.join(output_dir, "models"))
        sep.load_model(model_filename='htdemucs_ft.yaml') 
        output = sep.separate(input_path)
        res = {}
        for f in output:
            path = os.path.join(stem_dir, f)
            l = f.lower()
            if "vocals" in l: res["vocals"] = path
            elif "drums" in l: res["drums"] = path
            elif "bass" in l: res["bass"] = path
            elif "other" in l: res["other"] = path
        return res
    except: return None

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
        
        # Create mask (Ducks by 25% when vocals are active)
        dynamic_mask = create_dynamic_mask(v_temp, sr, duck_amount=0.25)

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
             
             # --- APPLY DYNAMIC MASK ---
             # We only duck the Bass and 'Other' (Melodies)
             # We leave Drums Punchy!
             if dynamic_mask is not None and name in ['bass', 'other']:
                 s_pad = s_pad * dynamic_mask
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

def generate_mashup_set(output_dir, data_a, data_b, shift_semitones=0, selected_styles=[]):
    print(f"🚀 Styles Requested: {selected_styles}") 
    sr_a, sr_b = data_a['sr'], data_b['sr']
    
    stems_a = separate_stems_4(data_a['path'], output_dir)
    stems_b = separate_stems_4(data_b['path'], output_dir)
    if not stems_a or not stems_b: return {}

    def load(p, sr): 
        y, _ = librosa.load(p, sr=sr, mono=False)
        return np.vstack([y, y]) if y.ndim == 1 else y

    S_A = {k: load(v, sr_a) for k, v in stems_a.items()}
    S_B = {k: load(v, sr_b) for k, v in stems_b.items()}

    bpm_a, bpm_b = data_a['bpm'], data_b['bpm']
    rate_a_to_b = get_smart_rate(bpm_a, bpm_b)
    rate_b_to_a = get_smart_rate(bpm_b, bpm_a)
    
    anc_a, anc_b = int(data_a['anchor_point'] * sr_a), int(data_b['anchor_point'] * sr_b)
    output_files = {}

    if any(x in selected_styles for x in ["Standard (A on B)"]):
        v_a = process_audio(S_A['vocals'], rate_a_to_b, shift_semitones, sr_b)
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
        output_files["Standard"] = p

    if any(x in selected_styles for x in ["The Flip (B on A)"]):
        v_b = process_audio(S_B['vocals'], rate_b_to_a, -shift_semitones, sr_a)
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
        output_files["The Flip"] = p

    if any(x in selected_styles for x in ["Rhythm Swap (Hybrid)"]):
        v_a = process_audio(S_A['vocals'], rate_a_to_b, shift_semitones, sr_b)
        d_a = process_audio(S_A['drums'], rate_a_to_b, shift_semitones, sr_b)
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
        
    if any(x in selected_styles for x in ["Deep Mode (Inst Shift)"]):
        v_a = process_audio(S_A['vocals'], rate_a_to_b, 0, sr_b) 
        inv = -shift_semitones
        d_b = process_audio(S_B['drums'], 1.0, inv, sr_b)
        b_b = process_audio(S_B['bass'], 1.0, inv, sr_b)
        o_b = process_audio(S_B['other'], 1.0, inv, sr_b)
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

    if any(x in selected_styles for x in ["Hype Mode (Nightcore)"]):
        v_a = process_audio(S_A['vocals'], rate_a_to_b, shift_semitones + 12, sr_b)
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
