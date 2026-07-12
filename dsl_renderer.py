"""
dsl_renderer.py -- Render an LLM-composed mashup timeline into audio. (v2: DJ toolkit)

Timing is in BARS -- the engine owns all sample math.

PLAN format
-----------
{
  "target_bpm": 100,
  "timeline": [
    { "bars": 16,
      "transition": "cut" | "crossfade" | {"type": "crossfade", "bars": 4},
      "layers": [
        { "song": "A", "stem": "vocals", "src_bars": [9, 25], "gain": 0,
          "fade_in": 2, "fade_out": 2,        # optional volume ramps (bars)
          "filter_in": 4, "filter_out": 4 },  # optional EQ blends (bars):
                                              #   filter_in: enters thin, opens to full body
                                              #   filter_out: loses its body/bass at the end
        { "song": "FX", "stem": "riser" }     # synthesized FX layers (no src_bars):
                                              #   riser (swell into next segment),
                                              #   impact (boom at segment start),
                                              #   sweep_down (noise tail from start)
      ] }
  ]
}
- src_bars = [start, end), 1-based.
- "instrumental" = all non-vocal stems, DEDUPLICATED by file path.
- gain is dB.
Notes: filter_in/out are implemented as a blend between the full and a
high-passed (600 Hz) copy -- reads as a DJ EQ move, not a true swept filter.
"""

import os
import numpy as np
from scipy.signal import butter, sosfiltfilt

try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

try:
    import pyrubberband as pyrb
    HAS_RUBBERBAND = True
except Exception:
    HAS_RUBBERBAND = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except Exception:
    HAS_SOUNDFILE = False
    from scipy.io import wavfile as _wavfile

BEATS_PER_BAR = 4
FX_STEMS = ("riser", "impact", "sweep_down")
MAX_XFADE_BARS = 16


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def _load_audio(path, sr):
    if HAS_LIBROSA:
        y, _ = librosa.load(path, sr=sr, mono=False)
    else:
        file_sr, y = _wavfile.read(path)
        if file_sr != sr:
            raise ValueError(f"{path}: sr {file_sr} != {sr} (no resampler available)")
        y = y.astype(np.float32)
        if y.ndim == 2:
            y = y.T
        if np.max(np.abs(y)) > 1.5:
            y = y / 32768.0
    if y.ndim == 1:
        y = np.vstack([y, y])
    return y.astype(np.float32)


def _write_audio(path, y, sr):
    y = np.clip(y, -1.0, 1.0)
    if HAS_SOUNDFILE:
        sf.write(path, y.T, sr)
    else:
        _wavfile.write(path, sr, (y.T * 32767).astype(np.int16))


# --------------------------------------------------------------------------
# DSP
# --------------------------------------------------------------------------
def _time_stretch(y, rate, sr):
    if abs(rate - 1.0) < 0.01:
        return y
    if HAS_RUBBERBAND:
        try:
            return pyrb.time_stretch(y.T, sr, rate).T
        except Exception:
            pass
    if HAS_LIBROSA:
        return librosa.effects.time_stretch(y, rate=rate)
    raise RuntimeError("No time-stretch backend (need librosa or pyrubberband).")


def _pitch_shift(y, sr, semitones):
    if semitones == 0:
        return y
    if HAS_RUBBERBAND:
        try:
            return pyrb.pitch_shift(y.T, sr, semitones).T
        except Exception:
            pass
    if HAS_LIBROSA:
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    raise RuntimeError("No pitch-shift backend (need librosa or pyrubberband).")


def _db_to_lin(db):
    return float(10.0 ** (db / 20.0))


def _volume_envelope(points, n_samples, bar_samps):
    """Breakpoint volume automation -> per-sample linear gain curve.
    points: [{"bar": float, "db": float}], bars relative to segment start.
    Interpolated in dB (perceptually smooth); first/last values held."""
    pts = sorted(points, key=lambda p: float(p["bar"]))
    xs = np.array([float(p["bar"]) * bar_samps for p in pts])
    ys = np.array([float(p["db"]) for p in pts])
    db_curve = np.interp(np.arange(n_samples), xs, ys)   # np.interp holds ends
    return (10.0 ** (db_curve / 20.0)).astype(np.float32)


def _thin(y, sr, cutoff=600.0):
    """High-passed copy: the 'body removed' version used by the EQ blends."""
    sos = butter(2, cutoff / (sr / 2), btype='high', output='sos')
    return sosfiltfilt(sos, y, axis=-1).astype(np.float32)


def _apply_shape(layer, sr, bar_samps, fade_in=0, fade_out=0, filter_in=0, filter_out=0):
    """Volume ramps + EQ blends, all expressed in bars on the segment timeline."""
    n = layer.shape[1]
    if fade_in:
        L = min(n, int(fade_in * bar_samps))
        layer[:, :L] *= np.linspace(0.0, 1.0, L, dtype=np.float32)
    if fade_out:
        L = min(n, int(fade_out * bar_samps))
        layer[:, n - L:] *= np.linspace(1.0, 0.0, L, dtype=np.float32)
    if filter_in or filter_out:
        thin = _thin(layer, sr)
        blend = np.zeros(n, dtype=np.float32)          # 0 = full, 1 = thin
        if filter_in:
            L = min(n, int(filter_in * bar_samps))
            blend[:L] = np.linspace(1.0, 0.0, L, dtype=np.float32)
        if filter_out:
            L = min(n, int(filter_out * bar_samps))
            blend[n - L:] = np.maximum(blend[n - L:], np.linspace(0.0, 1.0, L, dtype=np.float32))
        layer = layer * (1.0 - blend) + thin * blend
    return layer


def _swept_noise(L, sr, f_from, f_to, rng):
    """White noise through a time-varying lowpass sweeping f_from -> f_to.
    Implemented as 8 pre-filtered banks blended over time (click-free)."""
    noise = rng.standard_normal(L).astype(np.float32)
    K = 8
    cuts = np.geomspace(max(120.0, min(f_from, f_to)),
                        min(sr / 2 * 0.9, max(f_from, f_to)), K)
    if f_from > f_to:
        cuts = cuts[::-1]
    banks = []
    for c in cuts:
        sos = butter(4, c / (sr / 2), btype='low', output='sos')
        banks.append(sosfiltfilt(sos, noise).astype(np.float32))
    banks = np.stack(banks)                              # (K, L)
    pos = np.linspace(0, K - 1, L, dtype=np.float32)
    i0 = np.clip(pos.astype(np.int32), 0, K - 2)
    frac = pos - i0
    idx = np.arange(L)
    sig = banks[i0, idx] * (1 - frac) + banks[i0 + 1, idx] * frac
    return sig / (np.abs(sig).max() + 1e-9)


def _synth_fx(stem, seg_len, sr, bar_samps, length_bars=None):
    """Synthesized transition FX, placed within the segment. Noise is always
    filter-swept (never raw broadband static)."""
    out = np.zeros((2, seg_len), dtype=np.float32)
    rng = np.random.default_rng(7)
    if stem == "riser":                     # filtered swell over the LAST bars:
        L = min(seg_len, int((length_bars or 4) * bar_samps))
        t = np.linspace(0.0, 1.0, L, dtype=np.float32)
        sig = _swept_noise(L, sr, 350.0, 9000.0, rng)     # opens up as it rises
        sig *= (t ** 2.5) * 0.4                            # late, smooth swell
        out[:, seg_len - L:] = sig
    elif stem == "impact":                  # boom on the segment's first beat
        L = min(seg_len, int(1.5 * sr))
        t = np.arange(L) / sr
        boom = np.sin(2 * np.pi * 55 * t) * np.exp(-t / 0.35)
        nb = min(L, int(0.05 * sr))
        burst = np.zeros(L, dtype=np.float32)
        burst[:nb] = rng.standard_normal(nb).astype(np.float32) * np.exp(-np.arange(nb) / (0.012 * sr))
        sos = butter(4, 2500.0 / (sr / 2), btype='low', output='sos')  # tame the click
        burst = sosfiltfilt(sos, burst).astype(np.float32)
        out[:, :L] = (0.9 * boom + 0.5 * burst) * 0.8
    elif stem == "sweep_down":              # filtered tail closing down from the start
        L = min(seg_len, int((length_bars or 2) * bar_samps))
        t = np.linspace(1.0, 0.0, L, dtype=np.float32)
        sig = _swept_noise(L, sr, 9000.0, 350.0, rng)      # darkens as it falls
        out[:, :L] = sig * (t ** 2) * 0.3
    return out


# --------------------------------------------------------------------------
# Stem resolution (incl. the 2-stem alias trap)
# --------------------------------------------------------------------------
def resolve_stem_paths(song, stem_name):
    stems = song["stems"]
    if stem_name == "instrumental":
        seen, paths = set(), []
        for k, p in stems.items():
            if k == "vocals" or p is None:
                continue
            rp = os.path.realpath(p)
            if rp not in seen:
                seen.add(rp)
                paths.append(p)
        return paths
    if stem_name in stems and stems[stem_name] is not None:
        return [stems[stem_name]]
    return []


def stem_alias_warnings(song, song_id):
    paths = {}
    for k, p in song["stems"].items():
        if p is None:
            continue
        paths.setdefault(os.path.realpath(p), []).append(k)
    warns = []
    for rp, keys in paths.items():
        if len(keys) > 1:
            warns.append(f"song {song_id}: stems {keys} are the SAME file (2-stem model); "
                         f"requesting any of them plays the full instrumental.")
    return warns


# --------------------------------------------------------------------------
# Transition parsing
# --------------------------------------------------------------------------
def _xfade_bars(transition):
    """0 = cut; N = overlap the previous segment by N bars."""
    if transition is None or transition == "cut":
        return 0
    if transition == "crossfade":
        return 1
    if isinstance(transition, dict) and transition.get("type") == "crossfade":
        return int(np.clip(int(transition.get("bars", 1)), 1, MAX_XFADE_BARS))
    return None  # invalid


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_plan(plan, songs, sr=44100):
    errors, warnings = [], []
    if "target_bpm" not in plan or not (40 <= float(plan["target_bpm"]) <= 220):
        errors.append("plan.target_bpm missing or implausible (need 40-220).")
    timeline = plan.get("timeline", [])
    if not timeline:
        errors.append("plan.timeline is empty.")

    for sid, song in songs.items():
        warnings.extend(stem_alias_warnings(song, sid))

    for i, seg in enumerate(timeline, 1):
        bars = seg.get("bars", 0)
        if not (isinstance(bars, int) and 1 <= bars <= 128):
            errors.append(f"segment {i}: bars must be int in 1..128 (got {bars!r}).")
        if _xfade_bars(seg.get("transition", "cut")) is None:
            errors.append(f'segment {i}: transition must be "cut", "crossfade", '
                          f'or {{"type":"crossfade","bars":N}}.')
        layers = seg.get("layers", [])
        if not layers:
            errors.append(f"segment {i}: no layers.")
        for j, ly in enumerate(layers, 1):
            tag = f"segment {i} layer {j}"
            sid = ly.get("song")

            vol = ly.get("volume")
            if vol is not None:
                if (not isinstance(vol, list) or not vol
                        or not all(isinstance(p, dict) and "bar" in p and "db" in p for p in vol)):
                    errors.append(f'{tag}: volume must be a non-empty list of {{"bar","db"}} points.')
                else:
                    for p in vol:
                        if not (0 <= float(p["bar"]) <= (bars if isinstance(bars, int) else 128)):
                            errors.append(f"{tag}: volume point bar {p['bar']} outside segment (0..{bars}).")
                        if not (-40 <= float(p["db"]) <= 6):
                            errors.append(f"{tag}: volume point {p['db']} dB out of range (-40..+6).")

            if sid == "FX":
                if ly.get("stem") not in FX_STEMS:
                    errors.append(f"{tag}: FX stem must be one of {FX_STEMS}.")
                continue

            if sid not in songs:
                errors.append(f"{tag}: unknown song {sid!r}.")
                continue
            song = songs[sid]
            stem = ly.get("stem", "")
            if not resolve_stem_paths(song, stem):
                errors.append(f"{tag}: stem {stem!r} not available for song {sid} "
                              f"(have: {list(song['stems'].keys())} + 'instrumental').")
            sb = ly.get("src_bars")
            if (not isinstance(sb, (list, tuple)) or len(sb) != 2
                    or not all(isinstance(v, int) for v in sb) or sb[0] < 1 or sb[1] <= sb[0]):
                errors.append(f"{tag}: src_bars must be [start,end) 1-based ints, start<end.")
            else:
                n_src = sb[1] - sb[0]
                if isinstance(bars, int) and n_src != bars:
                    warnings.append(f"{tag}: {n_src} source bars into a {bars}-bar segment "
                                    f"-> will be truncated/padded.")
            g = ly.get("gain", 0)
            if not (-40 <= float(g) <= 12):
                errors.append(f"{tag}: gain {g} dB out of range (-40..+12).")
            for fld in ("fade_in", "fade_out", "filter_in", "filter_out"):
                v = ly.get(fld, 0)
                if not (isinstance(v, int) and 0 <= v <= MAX_XFADE_BARS):
                    errors.append(f"{tag}: {fld} must be int bars 0..{MAX_XFADE_BARS}.")
                elif isinstance(bars, int) and v > bars:
                    warnings.append(f"{tag}: {fld}={v} exceeds segment length {bars}.")
    return errors, warnings


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _extract_bars(y, sr, bpm, grid_start, bar_a, bar_b):
    bar_dur = BEATS_PER_BAR * 60.0 / bpm
    s0 = int(round((grid_start + (bar_a - 1) * bar_dur) * sr))
    s1 = int(round((grid_start + (bar_b - 1) * bar_dur) * sr))
    s0 = max(0, min(s0, y.shape[1]))
    s1 = max(s0, min(s1, y.shape[1]))
    return y[:, s0:s1]


def _fit_length(y, n):
    if y.shape[1] >= n:
        return y[:, :n]
    return np.pad(y, ((0, 0), (0, n - y.shape[1])))


def render_plan(plan, songs, output_path, sr=44100, verbose=True):
    report = []
    errors, warnings = validate_plan(plan, songs, sr)
    report.extend("WARN: " + w for w in warnings)
    if errors:
        raise ValueError("Plan invalid:\n" + "\n".join(" - " + e for e in errors))

    target_bpm = float(plan["target_bpm"])
    bar_samps = int(round(BEATS_PER_BAR * 60.0 / target_bpm * sr))

    audio_cache = {}

    def get_audio(sid, path):
        key = (sid, os.path.realpath(path))
        if key not in audio_cache:
            audio_cache[key] = _load_audio(path, sr)
        return audio_cache[key]

    rendered = []
    for i, seg in enumerate(plan["timeline"], 1):
        n_bars = seg["bars"]
        seg_len = n_bars * bar_samps
        mix = np.zeros((2, seg_len), dtype=np.float32)

        for ly in seg["layers"]:
            if ly.get("song") == "FX":
                fx = _synth_fx(ly["stem"], seg_len, sr, bar_samps, ly.get("length_bars"))
                if ly.get("volume"):
                    fx *= _volume_envelope(ly["volume"], fx.shape[1], bar_samps)
                mix += fx * _db_to_lin(float(ly.get("gain", 0)))
                continue

            song = songs[ly["song"]]
            bpm, grid = float(song["bpm"]), float(song.get("grid_start", 0.0))
            shift = int(song.get("shift", 0))
            rate = target_bpm / bpm

            layer = None
            for p in resolve_stem_paths(song, ly["stem"]):
                y = get_audio(ly["song"], p)
                part = _extract_bars(y, sr, bpm, grid, *ly["src_bars"])
                if layer is None:
                    layer = part
                else:
                    m = max(layer.shape[1], part.shape[1])
                    layer = _fit_length(layer, m) + _fit_length(part, m)

            layer = _time_stretch(layer, rate, sr)
            layer = _pitch_shift(layer, sr, shift)
            layer = _fit_length(layer, seg_len).copy()

            ef = min(int(0.015 * sr), layer.shape[1] // 4)   # declick edges
            if ef > 0:
                ramp = np.linspace(0, 1, ef, dtype=np.float32)
                layer[:, :ef] *= ramp
                layer[:, -ef:] *= ramp[::-1]

            layer = _apply_shape(layer, sr, bar_samps,
                                 fade_in=int(ly.get("fade_in", 0)),
                                 fade_out=int(ly.get("fade_out", 0)),
                                 filter_in=int(ly.get("filter_in", 0)),
                                 filter_out=int(ly.get("filter_out", 0)))
            if ly.get("volume"):
                layer *= _volume_envelope(ly["volume"], layer.shape[1], bar_samps)
            mix += layer * _db_to_lin(float(ly.get("gain", 0)))

        peak = np.max(np.abs(mix))
        if peak > 1.0:
            mix /= peak * 1.02
            report.append(f"segment {i}: peak-limited (sum exceeded 0 dBFS).")
        rendered.append((mix, _xfade_bars(seg.get("transition", "cut"))))
        if verbose:
            print(f"  segment {i}: {n_bars} bars, {len(seg['layers'])} layer(s), "
                  f"{seg_len / sr:.1f}s, xfade_in={_xfade_bars(seg.get('transition', 'cut'))} bars")

    total = sum(m.shape[1] for m, _ in rendered)
    out = np.zeros((2, total), dtype=np.float32)
    cursor = 0
    for k, (mix, xb) in enumerate(rendered):
        if k > 0 and xb:
            ov = min(xb * bar_samps, cursor, mix.shape[1])
            if ov > 0:
                fade = np.sqrt(np.linspace(0, 1, ov, dtype=np.float32))
                # bass swap: the outgoing tail progressively loses its low end
                # so the two songs' basslines never collide (kills the mud)
                if ov > sr // 8:
                    tail = out[:, cursor - ov:cursor]
                    hp = _thin(tail, sr, cutoff=220.0)
                    w = np.linspace(0.0, 1.0, ov, dtype=np.float32) ** 0.7
                    out[:, cursor - ov:cursor] = tail * (1.0 - w) + hp * w
                out[:, cursor - ov:cursor] *= fade[::-1]
                mix = mix.copy()
                mix[:, :ov] *= fade
                cursor -= ov
        out[:, cursor:cursor + mix.shape[1]] += mix
        cursor += mix.shape[1]
    out = out[:, :cursor]

    peak = np.max(np.abs(out)) + 1e-9
    out = out / peak * 0.95
    _write_audio(output_path, out, sr)
    report.append(f"rendered {cursor / sr:.1f}s to {output_path}")
    return output_path, report
