import numpy as np

# NumPy 2.x compatibility for older dependencies (madmom)
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "complex"):
    np.complex = complex
import os
import collections
import collections.abc
import sys

# Compatibility shim for older audio libs on newer Python versions.
if not hasattr(collections, 'MutableSequence'):
    collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable
sys.modules['collections'].MutableSequence = collections.abc.MutableSequence
sys.modules['collections'].Iterable = collections.abc.Iterable
sys.modules['collections'].Callable = collections.abc.Callable

# --- 1. SOTA DEEP LEARNING IMPORTS ---
try:
    import madmom
    HAS_MADMOM = True
    print("✅ Madmom (Deep Learning) Key Detector Loaded")
except Exception as e:
    HAS_MADMOM = False
    print(f"⚠️ Madmom unavailable ({e}). Using Math Fallback.")

# --- 2. MATH FALLBACK IMPORTS ---
import librosa
from scipy.stats import pearsonr

# --- CONSTANTS ---
PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Schmuckler Profiles (The Best Non-AI Math)
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

CAMELOT_MAP = {
    'Ab Minor': '1A', 'G# Minor': '1A', 'B Major': '1B', 'Cb Major': '1B',
    'Eb Minor': '2A', 'D# Minor': '2A', 'F# Major': '2B', 'Gb Major': '2B',
    'Bb Minor': '3A', 'A# Minor': '3A', 'Db Major': '3B', 'C# Major': '3B',
    'F Minor': '4A', 'Ab Major': '4B', 'G# Major': '4B',
    'C Minor': '5A', 'Eb Major': '5B', 'D# Major': '5B',
    'G Minor': '6A', 'Bb Major': '6B', 'A# Major': '6B',
    'D Minor': '7A', 'F Major': '7B',
    'A Minor': '8A', 'C Major': '8B',
    'E Minor': '9A', 'G Major': '9B',
    'B Minor': '10A', 'D Major': '10B',
    'F# Minor': '11A', 'Gb Minor': '11A', 'A Major': '11B',
    'Db Minor': '12A', 'C# Minor': '12A', 'E Major': '12B'
}

def detect_key_madmom(path):
    """
    SOTA: Uses a Convolutional Neural Network (CNN) to detect key.
    """
    try:
        # 1. Create the Neural Network Processor
        proc = madmom.features.key.CNNKeyRecognitionProcessor()
        
        # 2. Run Inference
        key_probs = proc(path)
        
        # 3. Decode Result (Madmom returns probabilities for all keys)
        # We assume the most probable key is the global key
        key_label = madmom.features.key.key_prediction_to_label(key_probs)
        
        return key_label # Returns e.g., "C major"
    except Exception as e:
        print(f"Madmom Error: {e}")
        return None

def detect_key_math(y, sr):
    """
    Fallback: Manual Krumhansl-Schmuckler Implementation.
    """
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_sum = np.sum(chroma, axis=1)
        
        major_corrs = []
        minor_corrs = []
        
        for i in range(12):
            rotated = np.roll(chroma_sum, -i)
            major_corrs.append(pearsonr(rotated, KS_MAJOR)[0])
            minor_corrs.append(pearsonr(rotated, KS_MINOR)[0])
            
        best_major_idx = np.argmax(major_corrs)
        best_minor_idx = np.argmax(minor_corrs)
        
        if major_corrs[best_major_idx] > minor_corrs[best_minor_idx]:
            return f"{PITCH_CLASS[best_major_idx]} Major"
        else:
            return f"{PITCH_CLASS[best_minor_idx]} Minor"
    except: return "C Major"

def detect_key(y, sr, path=None):
    # 1. Try Deep Learning (Madmom)
    if HAS_MADMOM and path and os.path.exists(path):
        key = detect_key_madmom(path)
        if key: return key.title() # Convert "C major" to "C Major"

    # 2. Fallback to Math
    return detect_key_math(y, sr)

def get_camelot_code(key_str):
    if not key_str: return None
    key_str = key_str.replace('Flat', 'b').replace('Sharp', '#')
    parts = key_str.split(' ')
    if len(parts) >= 2:
        root = parts[0]
        scale = parts[1].lower() 
        normalized = f"{root} {scale.title()}"
        return CAMELOT_MAP.get(normalized)
    return None

def get_pitch_shift_steps(key_a, key_b):
    pitch_map = {k: v for v, k in enumerate(PITCH_CLASS)}
    try:
        code_a = get_camelot_code(key_a)
        code_b = get_camelot_code(key_b)
        
        if code_a and code_b:
            h_a, h_b = int(code_a[:-1]), int(code_b[:-1])
            if abs(h_a - h_b) <= 1 or abs(h_a - h_b) == 11: return 0

        root_a = key_a.split(' ')[0]
        root_b = key_b.split(' ')[0]
        rep = {'Db':'C#', 'Eb':'D#', 'Gb':'F#', 'Ab':'G#', 'Bb':'A#'}
        for k,v in rep.items():
            root_a = root_a.replace(k,v)
            root_b = root_b.replace(k,v)

        val_a, val_b = pitch_map.get(root_a, 0), pitch_map.get(root_b, 0)
        diff = val_a - val_b
        
        if diff > 6: diff -= 12
        elif diff < -6: diff += 12
        return diff
    except: return 0