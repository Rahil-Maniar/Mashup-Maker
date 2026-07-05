"""Functional test: build synthetic 4-stem songs where each stem is a unique
sine frequency, render the user's scenario, then verify each output region
contains exactly the expected frequencies."""
import numpy as np
from scipy.io import wavfile
import os, shutil
from dsl_renderer import render_plan, validate_plan, resolve_stem_paths

SR = 22050
BPM = 120.0                    # bar = 2.0s
BAR = int(2.0 * SR)

def tone(freq, n):             # constant sine, stereo
    t = np.arange(n) / SR
    y = 0.4 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return np.vstack([y, y])

os.makedirs("test_stems", exist_ok=True)
def write(name, y):
    p = f"test_stems/{name}.wav"
    wavfile.write(p, SR, (y.T * 32767).astype(np.int16)); return p

N = 40 * BAR  # 40-bar songs
# Song A stems: vocals=440Hz, drums=100Hz, bass=60Hz, other=1000Hz
A = {"stems": {"vocals": write("A_voc", tone(440, N)), "drums": write("A_dr", tone(100, N)),
               "bass":   write("A_ba",  tone(60,  N)), "other": write("A_ot", tone(1000, N))},
     "bpm": BPM, "grid_start": 0.0, "shift": 0}
# Song B: 2-STEM ALIAS CASE -> drums/bass/other all point at ONE instrumental file (2kHz)
b_inst = write("B_inst", tone(2000, N))
B = {"stems": {"vocals": write("B_voc", tone(660, N)), "drums": b_inst,
               "bass": b_inst, "other": b_inst},
     "bpm": BPM, "grid_start": 0.0, "shift": 0}
songs = {"A": A, "B": B}

# The user's scenario in 4-stem terms:
# seg1: vocals A + instrumental B (16 bars)  [tests alias-dedup: B inst must sum ONCE]
# seg2: bass A alone (8 bars)                 [the 'solo stem interlude']
# seg3: vocals B + instrumental A (16 bars, crossfade)
plan = {"target_bpm": BPM, "timeline": [
  {"bars": 16, "transition": "cut",
   "layers": [{"song":"A","stem":"vocals","src_bars":[9,25],"gain":0},
              {"song":"B","stem":"instrumental","src_bars":[5,21],"gain":0}]},
  {"bars": 8, "transition": "cut",
   "layers": [{"song":"A","stem":"bass","src_bars":[33,41],"gain":0}]},
  {"bars": 16, "transition": "crossfade",
   "layers": [{"song":"B","stem":"vocals","src_bars":[25,41],"gain":0},
              {"song":"A","stem":"instrumental","src_bars":[9,25],"gain":-1}]},
]}

# --- validation checks ---
errs, warns = validate_plan(plan, songs, SR)
print("validation errors:", errs)
print("validation warnings:")
for w in warns: print("  ", w)
assert not errs

# alias-dedup check: B 'instrumental' must resolve to ONE path
assert len(resolve_stem_paths(B, "instrumental")) == 1, "alias dedup FAILED"
assert len(resolve_stem_paths(A, "instrumental")) == 3
print("alias dedup: OK (B instrumental = 1 file, A instrumental = 3 files)")

# --- a bad plan must be rejected with useful errors ---
bad = {"target_bpm": BPM, "timeline": [
  {"bars": 4, "transition": "cut",
   "layers": [{"song":"A","stem":"piano","src_bars":[1,5],"gain":0},   # no such stem
              {"song":"C","stem":"vocals","src_bars":[1,5],"gain":0}]}]}  # no such song
e2, _ = validate_plan(bad, songs, SR)
print("bad-plan errors caught:", len(e2)); assert len(e2) == 2

# --- render ---
out, report = render_plan(plan, songs, "test_out.wav", sr=SR)
for r in report: print("  ", r)

# --- verify content per region via FFT peaks ---
_, y = wavfile.read("test_out.wav"); y = y.astype(np.float32).T / 32767.0
xfade = 1 * BAR
exp_len = (16 + 8 + 16) * BAR - xfade   # crossfade overlaps 1 bar
print(f"length: got {y.shape[1]} expected {exp_len}"); assert abs(y.shape[1] - exp_len) < 10

def peaks(seg):
    m = np.mean(seg, axis=0)
    sp = np.abs(np.fft.rfft(m * np.hanning(len(m))))
    fr = np.fft.rfftfreq(len(m), 1/SR)
    found = set()
    for f in (60, 100, 440, 660, 1000, 2000):
        band = sp[(fr > f*0.97) & (fr < f*1.03)].max()
        if band > 0.005 * len(m): found.add(f)
    return found

mid = lambda a, b: y[:, (a*BAR)+BAR//2 : (b*BAR)-BAR//2]  # avoid edges
s1 = peaks(mid(0, 16));  print("seg1 freqs:", sorted(s1))
assert s1 == {440, 2000}, "seg1 should be vocals A (440) + inst B (2000) only"
s2 = peaks(mid(16, 24)); print("seg2 freqs:", sorted(s2))
assert s2 == {60}, "seg2 should be bass A (60) only"
s3 = peaks(y[:, 25*BAR: 39*BAR]); print("seg3 freqs:", sorted(s3))
assert s3 == {660, 60, 100, 1000}, "seg3 should be vocals B + full A instrumental"

# alias amplitude check: if B's instrumental summed 3x, seg1's 2kHz would be ~3x seg1's 440Hz
m1 = np.mean(mid(0,16), axis=0); sp = np.abs(np.fft.rfft(m1*np.hanning(len(m1)))); fr = np.fft.rfftfreq(len(m1),1/SR)
a440  = sp[(fr>430)&(fr<450)].max(); a2000 = sp[(fr>1950)&(fr<2050)].max()
ratio = a2000/a440
print(f"seg1 amplitude ratio inst/voc = {ratio:.2f} (should be ~1.0, would be ~3.0 if alias bug)")
assert ratio < 1.5

print("\nALL TESTS PASSED")
