"""
critic.py -- Closes the feedback loop. Two tools:

1. check_phrase_integrity(plan, session)   [PRE-render]
   Compares every vocal window in the plan against the timestamped lyric lines
   and reports any phrase that would be clipped mid-sentence -- the #1 observed
   failure -- before any audio is rendered.

2. critique_render(wav_path, plan, session) [POST-render]
   Measures the rendered mashup (per-segment loudness, arc shape, dead air) and
   combines it with phrase issues into a CRITIC REPORT formatted to paste back
   to the LLM in revision mode.
"""

import numpy as np

BEATS_PER_BAR = 4

# ---------------------------------------------------------------------------
# Energy-arc templates: shared by the prompt (descriptions guide the LLM's
# design) and the critic (curves verify the render delivered the arc).
# points = (position 0..1, target level 0..1)
# ---------------------------------------------------------------------------
ARC_TEMPLATES = {
    "slow_burn": {
        "description": ("open sparse and restrained, build continuously with no early "
                        "peaks, save the biggest payoff for the final quarter, short resolve"),
        "points": [(0.0, 0.30), (0.6, 0.60), (0.85, 1.00), (1.0, 0.90)],
    },
    "instant_banger": {
        "description": ("hit hard from bar one, carve out a mid-edit breakdown for "
                        "contrast, then return to full power for the finish"),
        "points": [(0.0, 1.00), (0.2, 0.85), (0.5, 0.55), (0.8, 0.95), (1.0, 1.00)],
    },
    "rollercoaster": {
        "description": ("two distinct peaks with a real valley between them -- an early "
                        "payoff, a stripped middle, and a bigger second payoff near the end"),
        "points": [(0.0, 0.40), (0.3, 1.00), (0.55, 0.45), (0.9, 1.00), (1.0, 0.70)],
    },
    "waves": {
        "description": ("gentle repeated swells -- energy breathes up and down every few "
                        "segments, never fully exploding, hypnotic rather than dramatic"),
        "points": [(0.0, 0.50), (0.25, 0.90), (0.5, 0.50), (0.75, 0.95), (1.0, 0.60)],
    },
}


def arc_target(name, positions):
    """Sample a template curve at the given 0..1 positions."""
    pts = ARC_TEMPLATES[name]["points"]
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    return np.interp(np.asarray(positions, dtype=float), xs, ys)


def check_arc(levels_db, arc_name):
    """Compare measured per-segment loudness to the requested arc template.
    Returns (ok: bool|None, message). None = not enough data to judge."""
    if arc_name not in ARC_TEMPLATES:
        return None, f"unknown arc template {arc_name!r}"
    lv = np.asarray(levels_db, dtype=float)
    n = len(lv)
    if n < 3 or float(np.std(lv)) < 0.5:
        return None, "too few segments (or a flat mix) to judge the arc"
    meas = (lv - lv.min()) / (lv.max() - lv.min())
    mids = (np.arange(n) + 0.5) / n
    tgt = arc_target(arc_name, mids)
    r = float(np.corrcoef(meas, tgt)[0, 1])
    meas_s = " ".join(f"{v:.1f}" for v in meas)
    tgt_s = " ".join(f"{v:.1f}" for v in tgt)
    if r >= 0.5:
        return True, (f"requested arc '{arc_name}' delivered (shape match r={r:.2f}; "
                      f"measured [{meas_s}] vs target [{tgt_s}])")
    return False, (f"requested arc '{arc_name}' NOT delivered (r={r:.2f}); "
                   f"measured [{meas_s}] vs target [{tgt_s}] -- adjust segment gains/"
                   f"volume automation so the loudness shape follows the target.")
CLIP_TOLERANCE = 0.2   # bars; shaving less than this off a phrase edge is inaudible


# ---------------------------------------------------------------------------
# 1. Pre-render: phrase integrity
# ---------------------------------------------------------------------------
def check_phrase_integrity(plan, session):
    """Return a list of human-readable issues: vocal windows that clip phrases."""
    lyrics = session.get("lyrics", {})
    issues = []
    for i, seg in enumerate(plan.get("timeline", []), 1):
        for j, ly in enumerate(seg.get("layers", []), 1):
            if ly.get("song") == "FX" or ly.get("stem") != "vocals":
                continue
            lines = lyrics.get(ly.get("song"), [])
            if not lines or not ly.get("src_bars"):
                continue
            s, e = float(ly["src_bars"][0]), float(ly["src_bars"][1])
            for ln in lines:
                b0, b1 = float(ln["bar_start"]), float(ln["bar_end"])
                if b1 <= s or b0 >= e:
                    continue                                   # outside window
                head_cut = s - b0                              # window starts inside phrase
                tail_cut = b1 - e                              # window ends inside phrase
                if head_cut > CLIP_TOLERANCE:
                    issues.append(
                        f'segment {i} layer {j}: window [{int(s)},{int(e)}) BEHEADS '
                        f'"{ln["text"][:45]}" (phrase starts at bar {b0}) -- '
                        f'start the window at bar {int(np.floor(b0))} or earlier.')
                if tail_cut > CLIP_TOLERANCE:
                    issues.append(
                        f'segment {i} layer {j}: window [{int(s)},{int(e)}) CUTS OFF '
                        f'"{ln["text"][:45]}" (phrase ends at bar {b1}) -- '
                        f'end the window at bar {int(np.ceil(b1))} or later.')
    return issues


# ---------------------------------------------------------------------------
# 2. Post-render: measured critique
# ---------------------------------------------------------------------------
def _xfade_bars_of(transition):
    if transition in (None, "cut"):
        return 0
    if transition == "crossfade":
        return 1
    if isinstance(transition, dict) and transition.get("type") == "crossfade":
        return max(1, min(16, int(transition.get("bars", 1))))
    return 0


def _segment_spans(plan, sr):
    """Sample spans of each segment in the FINAL rendered file (xfades overlap)."""
    bar = int(round(BEATS_PER_BAR * 60.0 / float(plan["target_bpm"]) * sr))
    spans, cursor = [], 0
    for k, seg in enumerate(plan["timeline"]):
        if k > 0:
            cursor -= min(_xfade_bars_of(seg.get("transition", "cut")) * bar, cursor)
        length = seg["bars"] * bar
        spans.append((cursor, cursor + length))
        cursor += length
    return spans


def _rms_db(x):
    r = float(np.sqrt(np.mean(np.square(x)))) + 1e-9
    return 20.0 * np.log10(r)


def critique_render(wav_path, plan, session, sr_hint=None):
    """Measure the render and produce {'metrics': [...], 'flags': [...], 'report': str}."""
    from scipy.io import wavfile
    sr, y = wavfile.read(wav_path)
    y = y.astype(np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if np.max(np.abs(y)) > 1.5:
        y = y / 32768.0

    spans = _segment_spans(plan, sr)
    metrics, flags = [], []
    levels = []
    for i, (a, b) in enumerate(spans, 1):
        seg = y[a:min(b, len(y))]
        if len(seg) == 0:
            flags.append(f"segment {i}: fell outside the rendered file (length mismatch?).")
            levels.append(-60.0)
            continue
        db = _rms_db(seg)
        levels.append(db)
        # dead air inside the segment: any half-bar window below -45 dB
        bar = int(round(BEATS_PER_BAR * 60.0 / float(plan["target_bpm"]) * sr))
        half = max(1, bar // 2)
        quiet = 0
        for w0 in range(0, max(1, len(seg) - half), half):
            if _rms_db(seg[w0:w0 + half]) < -45:
                quiet += 1
        metrics.append({"segment": i, "rms_db": round(db, 1),
                        "quiet_halfbars": quiet, "bars": plan["timeline"][i - 1]["bars"]})
        if quiet >= 2:
            flags.append(f"segment {i}: {quiet} half-bars of near-silence -- dead air; "
                         f"check src_bars point at active material.")

    # Arc-shape checks
    if levels:
        peak_seg = int(np.argmax(levels)) + 1
        if peak_seg == 1 and len(levels) > 2:
            flags.append("arc: the FIRST segment is the loudest -- the edit opens at max "
                         "and can only decay; pull segment 1 down or move the payoff later.")
        if len(levels) >= 3:
            for i in range(1, len(levels) - 1):
                if levels[i] < min(levels[i - 1], levels[i + 1]) - 6:
                    flags.append(f"segment {i+1}: sits {min(levels[i-1],levels[i+1])-levels[i]:.0f} dB "
                                 f"below its neighbors -- if not an intentional breakdown, raise its gain.")
        if levels[-1] > max(levels[:-1] or [0]) - 1 and len(levels) > 2:
            pass  # ending on the peak is fine
        span = max(levels) - min(levels)
        if span < 3:
            flags.append(f"arc: only {span:.1f} dB of movement across the whole edit -- "
                         "it will feel flat; deepen a breakdown or push a payoff.")

    brief = session.get("brief") or {}
    brief_lines = []
    if brief.get("arc"):
        ok, msg = check_arc(levels, brief["arc"])
        if ok is False:
            flags.append("BRIEF: " + msg)
        elif ok is True:
            brief_lines.append("BRIEF CHECK: " + msg)
        else:
            brief_lines.append("BRIEF CHECK: " + msg)

    phrase_issues = check_phrase_integrity(plan, session)

    lines = ["=== CRITIC REPORT (auto-measured from the render) ==="]
    lines.append("Per-segment loudness: " +
                 "  ".join(f"S{m['segment']}:{m['rms_db']}dB" for m in metrics))
    lines += brief_lines
    if phrase_issues:
        lines.append("\nPHRASE INTEGRITY:")
        lines += ["- " + p for p in phrase_issues]
    if flags:
        lines.append("\nMIX FLAGS:")
        lines += ["- " + f for f in flags]
    if not phrase_issues and not flags:
        lines.append("\nNo mechanical issues found. Remaining judgment is taste: listen and "
                     "give the DJ subjective notes (mood, energy, which lines land).")
    else:
        lines.append("\nTO REVISE: paste this report to your LLM (revision mode) -- it should "
                     "keep everything not mentioned and fix only the flagged items.")
    return {"metrics": metrics, "flags": flags, "phrase_issues": phrase_issues,
            "report": "\n".join(lines)}
    