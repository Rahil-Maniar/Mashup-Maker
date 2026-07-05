"""
studio_logic.py -- Pure logic for the Mashup Studio UI (no streamlit imports,
so it's unit-testable): blend compatibility verdict + automatic plan generation.
"""

import hashlib
import os


# ---------------------------------------------------------------------------
# Blend verdict: "do these two songs go together?"
# ---------------------------------------------------------------------------
def blend_report(key_a, key_b, bpm_a, bpm_b, shift):
    """Score + human verdict from the compatibility stack (key/tempo levels)."""
    ratio = bpm_b / bpm_a if bpm_a else 1.0
    checks = []
    score = 100

    # Tempo (near-1:1 is the musically-correct zone; user-validated)
    if 0.95 <= ratio <= 1.05:
        checks.append(("Tempo", "ok", f"{bpm_a:.1f} vs {bpm_b:.1f} BPM ({ratio*100:.0f}%) - near-identical"))
    elif 0.88 <= ratio <= 1.14:
        checks.append(("Tempo", "warn", f"{ratio*100:.0f}% - stretch audible but workable"))
        score -= 20
    else:
        checks.append(("Tempo", "bad", f"{ratio*100:.0f}% - groove/feel will fight"))
        score -= 55

    # Key / harmonic (Camelot-based shift the pipeline computed)
    if shift == 0:
        checks.append(("Key", "ok", f"{key_a} vs {key_b} - compatible with no pitch shift"))
    elif abs(shift) <= 2:
        checks.append(("Key", "warn", f"needs {shift:+d} semitones - vocals slightly unnatural"))
        score -= 15
    else:
        checks.append(("Key", "bad", f"needs {shift:+d} semitones - vocals will sound wrong"))
        score -= 45

    if score >= 85:
        verdict = "Strong blend - keys and tempo line up. Chord loops and spectral room decide the rest."
    elif score >= 60:
        verdict = "Workable - expect some friction; a good arrangement can hide it."
    else:
        verdict = "Poor blend - the fundamentals fight. Consider a different pair."
    return {"score": max(0, score), "verdict": verdict, "checks": checks}


def file_hash(path, _bufsize=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Auto-plan: a decent default arrangement from grids + lyrics (no LLM needed)
# ---------------------------------------------------------------------------
def _best_window(grid, length, metric="vocal", lo=1, hi=None, min_vocal=0):
    """1-based start bar of the `length`-bar window maximizing the summed metric."""
    vals = grid[metric]
    n = grid["n_bars"]
    hi = min(hi or n, n)
    best, best_s = -1.0, lo
    for s in range(lo, max(lo + 1, hi - length + 1)):
        w = vals[s - 1: s - 1 + length]
        if len(w) < length:
            break
        if min_vocal and min(grid["vocal"][s - 1: s - 1 + length]) < min_vocal:
            continue
        tot = float(sum(w))
        if tot > best:
            best, best_s = tot, s
    return best_s


def _quietest_vocal_window(grid, length):
    """Verse finder: lowest-energy window that still has vocals throughout."""
    n = grid["n_bars"]
    best, best_s = float("inf"), 1
    for s in range(1, max(2, n - length + 1)):
        voc = grid["vocal"][s - 1: s - 1 + length]
        if len(voc) < length or min(voc) < 1:
            continue
        e = float(sum(grid["energy"][s - 1: s - 1 + length]))
        if e < best:
            best, best_s = e, s
    return best_s


def _snap_to_phrase(start, lyrics):
    """Snap a window start to the floor of the nearest phrase start (if lyrics exist)."""
    if not lyrics:
        return start
    best = min(lyrics, key=lambda l: abs(l["bar_start"] - start))
    return max(1, int(best["bar_start"]))


def auto_plan(session, seg_len=8):
    """Build a 4-part default arrangement: A verse over B groove -> A hook over
    B hook -> B hook over A hook (the flip) -> outro walk-down. Uses the DJ
    recipes (EQ blends, riser+impact) and lyric snapping when available."""
    grids = session["grids"]
    lyrics = session.get("lyrics", {"A": [], "B": []})
    ga, gb = grids["A"], grids["B"]

    a_verse = _snap_to_phrase(_quietest_vocal_window(ga, seg_len), lyrics.get("A"))
    a_hook = _snap_to_phrase(_best_window(ga, seg_len, "vocal", min_vocal=1), lyrics.get("A"))
    b_hook = _snap_to_phrase(_best_window(gb, seg_len, "vocal", min_vocal=1), lyrics.get("B"))
    b_groove = _best_window(gb, seg_len, "energy")

    def clamp(s, grid):
        return max(1, min(s, grid["n_bars"] - seg_len))

    a_verse, a_hook = clamp(a_verse, ga), clamp(a_hook, ga)
    b_hook, b_groove = clamp(b_hook, gb), clamp(b_groove, gb)

    L = seg_len
    return {
        "target_bpm": float(session["songs"]["B"]["bpm"]),
        "comment": "Auto-arranged default: A verse over B groove, trading hooks, walk-down outro.",
        "timeline": [
            {"bars": L, "transition": "cut", "layers": [
                {"song": "A", "stem": "vocals", "src_bars": [a_verse, a_verse + L], "gain": 0},
                {"song": "B", "stem": "instrumental", "src_bars": [b_groove, b_groove + L],
                 "gain": -3, "filter_in": 2, "filter_out": 2},
            ]},
            {"bars": L, "transition": {"type": "crossfade", "bars": 2}, "layers": [
                {"song": "A", "stem": "vocals", "src_bars": [a_hook, a_hook + L], "gain": 0},
                {"song": "B", "stem": "instrumental", "src_bars": [b_hook, b_hook + L],
                 "gain": -3, "filter_in": 2},
                {"song": "FX", "stem": "impact", "gain": -8},
            ]},
            {"bars": L, "transition": {"type": "crossfade", "bars": 4}, "layers": [
                {"song": "B", "stem": "vocals", "src_bars": [b_hook, b_hook + L], "gain": 0},
                {"song": "A", "stem": "instrumental", "src_bars": [a_hook, a_hook + L],
                 "gain": -3, "filter_in": 4, "filter_out": 2},
            ]},
            {"bars": L, "transition": {"type": "crossfade", "bars": 2}, "layers": [
                {"song": "B", "stem": "vocals", "src_bars": [b_hook, b_hook + L], "gain": 0},
                {"song": "B", "stem": "instrumental", "src_bars": [b_hook, b_hook + L],
                 "gain": -3,
                 "volume": [{"bar": 0, "db": 0}, {"bar": L - 3, "db": 0}, {"bar": L, "db": -14}]},
            ]},
        ],
    }
    