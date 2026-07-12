"""
llm_dj.py -- Glue for the LLM-as-DJ loop. Two commands:

  1) python llm_dj.py prepare songA.wav songB.wav
       - analyzes both songs (BPM, key, downbeat anchor)  [uses your existing modules]
       - separates stems via your cached separate_full_song
       - computes a per-bar feature grid from the stems
       - writes:  dj_session.json  (machine data for the render step)
                  dj_prompt.txt    (paste this into claude.ai / any LLM)

  2) python llm_dj.py render dj_session.json plan.json
       - validates the LLM's plan (prints errors you can paste back to the LLM)
       - renders llm_mashup.wav via dsl_renderer

Optional overrides for prepare (if auto-analysis misfires):
  --bpm-a 128 --bpm-b 100 --key-a "C Major" --key-b "A Minor"
"""

import os
import sys
import json
import numpy as np

from dsl_renderer import render_plan, validate_plan, _load_audio, resolve_stem_paths, stem_alias_warnings

SR_ANALYSIS = 22050
BEATS_PER_BAR = 4


# ---------------------------------------------------------------------------
# Analysis (uses your existing pipeline; guarded so overrides can substitute)
# ---------------------------------------------------------------------------
def analyze_song(path, bpm_override=None, key_override=None):
    bpm, anchor, key = bpm_override, 0.0, key_override
    if bpm is None or key is None:
        from mashup_maker import analyze_structure_advanced      # your module
        from key_finder import detect_key                        # your module
        import librosa
        a = analyze_structure_advanced(path)
        bpm = bpm or float(a["bpm"])
        anchor = float(a["anchor_point"])
        if key is None:
            y, sr = librosa.load(path, sr=SR_ANALYSIS, duration=120)
            key = detect_key(y, sr, path=path)
    return {"bpm": float(bpm), "anchor": float(anchor), "key": key}


def compute_shifts(key_a, key_b):
    """Song B is the tonal anchor (shift_B = 0); A moves to B by your QC math."""
    try:
        from key_finder import get_pitch_shift_steps
        return int(get_pitch_shift_steps(key_a, key_b)), 0
    except Exception:
        return 0, 0


# ---------------------------------------------------------------------------
# Per-bar feature grid (energy / vocal / bass activity, quantized 0-9)
# ---------------------------------------------------------------------------
def bar_grid(song, sr=SR_ANALYSIS, max_bars=160):
    """One row per bar: total energy, vocal activity, bass activity (0-9 each),
    computed from the separated stems on the song's uniform bar grid."""
    bpm, grid_start = song["bpm"], song["grid_start"]
    bar_len = int(round(BEATS_PER_BAR * 60.0 / bpm * sr))

    def per_bar_rms(paths):
        y = None
        seen = set()
        for p in paths:
            rp = os.path.realpath(p)
            if rp in seen:
                continue
            seen.add(rp)
            a = _load_audio(p, sr)
            m = np.mean(a, axis=0)
            y = m if y is None else (np.pad(y, (0, max(0, len(m) - len(y))))
                                     + np.pad(m, (0, max(0, len(y) - len(m)))))
        if y is None:
            return np.zeros(1)
        start = int(round(grid_start * sr))
        n_bars = min(max_bars, max(1, (len(y) - start) // bar_len))
        vals = np.zeros(n_bars)
        for b in range(n_bars):
            seg = y[start + b * bar_len: start + (b + 1) * bar_len]
            vals[b] = np.sqrt(np.mean(seg ** 2)) if len(seg) else 0.0
        return vals

    voc = per_bar_rms([song["stems"]["vocals"]])
    inst = per_bar_rms(resolve_stem_paths(song, "instrumental"))
    bass = per_bar_rms([song["stems"]["bass"]]) if song["stems"].get("bass") else inst
    n = min(len(voc), len(inst), len(bass))
    voc, inst, bass = voc[:n], inst[:n], bass[:n]
    total = voc + inst

    def q(v):  # quantize to 0-9 relative to this song's own max
        m = v.max() + 1e-9
        return np.clip((v / m * 9.999).astype(int), 0, 9)

    return {"n_bars": int(n), "energy": q(total).tolist(),
            "vocal": q(voc).tolist(), "bass": q(bass).tolist()}


def grid_to_text(grid, step=2):
    """Compact text table (one row per `step` bars) for the prompt."""
    rows = ["bar | energy vocal bass"]
    for b in range(0, grid["n_bars"], step):
        rows.append(f"{b+1:>3} |   {grid['energy'][b]}      {grid['vocal'][b]}     {grid['bass'][b]}")
    return "\n".join(rows)



# ---------------------------------------------------------------------------
# Lyrics with timestamps (Whisper on the ISOLATED VOCAL STEM)
# ---------------------------------------------------------------------------
def transcribe_lyrics(vocal_path):
    """Transcribe the vocal stem -> [{'start': s, 'end': s, 'text': str}].
    Prefers faster-whisper (fits a GTX 1650); falls back to openai-whisper;
    returns [] with a notice if neither is installed."""
    model_size = os.environ.get("DJ_WHISPER_MODEL", "small")
    try:
        from faster_whisper import WhisperModel

        def _run(device, compute):
            print(f"  📝 Transcribing lyrics on {device.upper()} "
                  f"(model '{model_size}'; first run downloads it, then this can "
                  f"take a few minutes on CPU)...")
            model = WhisperModel(model_size, device=device, compute_type=compute)
            segs, _ = model.transcribe(vocal_path, vad_filter=True)
            # consume the generator HERE so CUDA errors surface inside this call
            out = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
                   for s in segs if s.text.strip()]
            print(f"  📝 Lyrics done: {len(out)} lines.")
            return out

        try:
            return _run("cuda", "int8_float16")
        except Exception as e:
            # CUDA libs missing/broken (e.g. cublas64_12.dll) -- fall back to CPU.
            print(f"  (GPU whisper unavailable: {type(e).__name__}: {e} -- using CPU)")
            return _run("cpu", "int8")
    except ImportError:
        pass
    try:
        import whisper
        model = whisper.load_model(model_size)
        out = model.transcribe(vocal_path)
        return [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
                for s in out.get("segments", []) if s["text"].strip()]
    except ImportError:
        print("  (no whisper installed -- skipping lyrics. pip install faster-whisper)")
        return []


def lyrics_to_bar_lines(segments, bpm, grid_start, max_lines=60):
    """Convert second-timestamps to DECIMAL bar positions on the song's grid.
    Decimal matters: phrases often start on a pickup before the barline."""
    bar_dur = BEATS_PER_BAR * 60.0 / bpm
    lines = []
    for s in segments[:max_lines]:
        b0 = (s["start"] - grid_start) / bar_dur + 1.0
        b1 = (s["end"] - grid_start) / bar_dur + 1.0
        if b1 < 1.0:
            continue
        text = s["text"][:70]
        lines.append({"bar_start": round(b0, 1), "bar_end": round(b1, 1), "text": text})
    return lines


def lyrics_to_text(lines):
    if not lines:
        return "(no transcription available)"
    return "\n".join(f"bars {l['bar_start']:>5.1f}-{l['bar_end']:>5.1f}: {l['text']}"
                      for l in lines)

# ---------------------------------------------------------------------------
# The DJ prompt
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are an expert mashup DJ and arranger. Your job: design a mashup
arrangement of two songs, as a JSON "timeline plan" that an audio engine will render.
You cannot hear the songs; you are given their musical data instead. Reason like an
arranger reading a score.

=== SONG DATA ===
Each song lists: key, BPM, available stems, and a PER-BAR grid.
Grid columns (each 0-9, relative to that song's own maximum):
  energy = overall loudness of the bar
  vocal  = vocal activity (0 = no vocals in this bar)
  bass   = low-end activity
Rows are printed every 2 bars; assume smooth values in between.
Use the grid to find structure yourself: intros (low energy, vocal 0), verses
(mid energy, vocal>0), hooks/drops (energy 8-9), breakdowns (energy dips), etc.

HOW TO READ THE DATA LIKE A DJ:
* Hooks: a lyric line that REPEATS at multiple positions is almost certainly the hook;
  confirm with vocal 7-9 and energy 8-9 at those bars. Hooks are your payoffs.
* Drop-outs: a bar where bass/energy dips while vocal stays (e.g. bass 2 amid 8s) is a
  pre-chorus launch ramp -- cutting or dropping ON the bar after it hits hardest.
* Lyrics are MACHINE-TRANSCRIBED and imperfect: words may be misheard, and several sung
  lines sometimes merge into one long entry. Treat each entry's boundaries as real pauses
  (safe cut points) even when the words look wrong, and cross-check against the vocal
  column -- vocal 0 bars are safe cut zones regardless of what the lyric sheet implies.

DESIGN PROCESS (in this order):
1. Read both lyric sheets and name the RELATIONSHIP between the songs (argument, echo,
   answer, same story from two sides). This is the mashup's reason to exist.
2. Pick 2-3 pillar lines -- the lyric moments the whole edit is built to deliver.
3. Sketch the arc: establish -> build -> payoff -> (contrast) -> final payoff -> outro.
4. Only then fill in segments, transitions, and mix moves.

--- SONG A: {name_a} ---
key: {key_a}   BPM: {bpm_a}   pitch-shift applied by engine: {shift_a:+d} semitones
stems you may reference: {stems_a}
{grid_a}

LYRICS A (phrase -> bar positions; decimals = phrase starts/ends mid-bar):
{lyrics_a}

--- SONG B: {name_b} ---
key: {key_b}   BPM: {bpm_b}   pitch-shift applied by engine: {shift_b:+d} semitones
stems you may reference: {stems_b}
{grid_b}

LYRICS B (phrase -> bar positions; decimals = phrase starts/ends mid-bar):
{lyrics_b}

{alias_note}

{brief_section}=== DJ TOOLKIT (transitions & FX) ===
Segment "transition" controls how it enters over the PREVIOUS segment:
  "cut"  -- hard downbeat hit;  "crossfade" -- 1-bar blend;
  {{"type":"crossfade","bars":N}} -- N-bar overlap where BOTH segments play (2-8 typical).
Per-layer optional controls (values are BARS):
  "fade_in": N / "fade_out": N  -- volume ramp at the layer's start/end
  "filter_in": N  -- enters thin (no lows) and opens to full body: classic EQ blend-in
  "filter_out": N -- loses its body/bass over its final bars: classic EQ-out
  "volume": [{{"bar": 0, "db": -6}}, {{"bar": 4, "db": 0}}, ...] -- VOLUME AUTOMATION.
     Breakpoints in bars RELATIVE TO THE SEGMENT START (decimals allowed), dB -40..+6,
     interpolated smoothly between points, held after the last. Stacks on top of "gain".
FX layers (synthesized, no src_bars): {{"song":"FX","stem":"riser","gain":-6}}
  "riser" -- noise swell over the LAST 4 bars of its segment (build into the next drop)
  "impact" -- boom on this segment's FIRST beat (mark a drop)
  "sweep_down" -- noise tail decaying from this segment's start

RECIPES (use these -- hard instrumental swaps sound amateurish):
* Smooth music swap: outgoing segment's instrumental gets "filter_out": 4 (+ optional
  FX riser); incoming segment uses transition {{"type":"crossfade","bars":4}} and its
  instrumental gets "filter_in": 4. The basses trade places instead of colliding.
* Dramatic drop: keep transition "cut", but put an FX "impact" layer on the incoming
  segment and an FX "riser" on the outgoing one.
Reserve plain unadorned "cut" for at most one deliberate dramatic moment.

MIX LIKE A PRODUCER (volume automation is what separates a mix from a bounce):
* Duck the instrumental 3-6 dB under the most important lyric line of a segment,
  then restore it -- the line will cut through and the restore feels like a lift.
* Swell into hooks: instrumental rising 3-4 dB over the 2 bars before a payoff.
* Sink outros: walk the last layers down over the final bars instead of stopping flat.
* Ride FX: a riser that also swells via automation hits harder than a static one.
Every segment longer than 8 bars should have at least one deliberate volume move.

=== OUTPUT FORMAT ===
Think first: you may reason in prose about structure, pillar lines, and transitions.
Then END your reply with the complete plan as ONE JSON object -- it must be the LAST
thing in your reply. Schema:
{{
  "target_bpm": <number: MUST be exactly {bpm_a} or {bpm_b}>,
  "comment": "<one sentence describing your creative idea>",
  "timeline": [
    {{ "bars": <int 4-32>, "transition": "cut" | "crossfade" | {{"type":"crossfade","bars":N}},
       "layers": [
         {{ "song": "A" | "B", "stem": "<one of the listed stems>",
            "src_bars": [<start>, <end>],   // 1-based, end-exclusive; (end-start) SHOULD equal "bars"
            "gain": <dB, -12..0> }}
       ] }}
  ]
}}

=== RULES ===
1. src_bars must stay within each song's grid (Song A has {nbars_a} bars, Song B has {nbars_b}).
2. Only reference bars where the stem is actually active (vocal layer needs vocal > 0 there).
3. 1-3 layers per segment. Total arrangement length: 32-80 bars.
4. Do not put two vocal layers in the same segment unless intentionally trading lines.
5. Pitch/tempo are handled by the engine -- never compensate for them yourself.
6. Think structure: establish -> build -> payoff -> outro. Use energy 8-9 bars as payoffs.
7. Transitions: "cut" hits hard on a downbeat; "crossfade" blends over 1 bar.
8. NEVER cut a vocal mid-phrase. When a layer uses vocals, its src_bars window must
   contain complete lyric lines: start at or before floor(phrase bar_start) and end
   after the last phrase's bar_end (e.g. phrase at 16.8-20.3 -> window [16, 21) or wider).
9. Use the MEANING of the lyrics: pick lines that answer, echo, or argue with each
   other across songs. Quote the key lyric of each segment in your comment.
10. Every instrumental change must use a blended transition (crossfade 2-8 bars with
    filter_in/filter_out) or an FX-marked drop -- never a bare unadorned cut.
11. Use "volume" automation to set mood: no long segment should sit at one static level.
    Anchor your moves to specific lyric lines (say which line in your comment).

BEFORE YOU OUTPUT, CHECK YOUR PLAN:
[ ] Every vocal window contains complete lyric entries (no mid-phrase starts/ends).
[ ] The arc moves: it does NOT open at maximum energy, and it ends deliberately
    (walk-down or clean resolve), never just stopping flat.
[ ] No segment overlaps two instrumentals at full level -- overlapping instrumentals
    must trade via filter_in/filter_out.
[ ] Total length 40-80 bars; 4-8 segments; segments mostly 8+ bars (choppier = amateur).
[ ] Vocals sit at gain 0; instrumentals at -2 to -4 under them.
[ ] Each pillar line has a mix move (duck, drop, or swell) that spotlights it.

REVISION MODE: if the user replies with feedback on a rendered plan (e.g. "the duck at
segment 3 is too deep", "that cut lands early"), keep everything they didn't mention,
change only what they did, and output the complete revised JSON again.

Design something with a clear creative idea (e.g. A's hook answered by B's hook,
a stripped breakdown using one stem alone, verse/chorus source alternation).
"""


def build_prompt(songs, grids, names, lyrics=None, brief=None):
    lyrics = lyrics or {"A": [], "B": []}
    brief_section = ""
    if brief and (brief.get("text") or brief.get("arc")):
        from critic import ARC_TEMPLATES
        parts = ["=== CREATIVE BRIEF (from the user -- this overrides your own taste) ==="]
        if brief.get("text"):
            parts.append(f"Direction: {brief['text']}")
        if brief.get("arc") in ARC_TEMPLATES:
            parts.append(f"Target energy arc '{brief['arc']}': "
                         f"{ARC_TEMPLATES[brief['arc']]['description']}.")
            parts.append("Design segment gains and volume automation so per-segment "
                         "loudness follows this arc -- it WILL be measured on the render.")
        parts.append("State in your comment how the plan delivers the brief.")
        brief_section = "\n".join(parts) + "\n\n"
    warns = []
    for sid in ("A", "B"):
        warns += stem_alias_warnings(songs[sid], sid)
    alias_note = ""
    if warns:
        alias_note = ("NOTE: " + " ".join(warns) +
                      " Therefore treat the stem palette as exactly: vocals, instrumental.")
        stems_txt = {sid: "vocals, instrumental" for sid in ("A", "B")}
    else:
        stems_txt = {sid: ", ".join(list(songs[sid]["stems"].keys()) + ["instrumental"])
                     for sid in ("A", "B")}
    return PROMPT_TEMPLATE.format(
        name_a=names["A"], name_b=names["B"],
        key_a=songs["A"]["key"], key_b=songs["B"]["key"],
        bpm_a=round(songs["A"]["bpm"], 1), bpm_b=round(songs["B"]["bpm"], 1),
        shift_a=songs["A"]["shift"], shift_b=songs["B"]["shift"],
        stems_a=stems_txt["A"], stems_b=stems_txt["B"],
        grid_a=grid_to_text(grids["A"]), grid_b=grid_to_text(grids["B"]),
        nbars_a=grids["A"]["n_bars"], nbars_b=grids["B"]["n_bars"],
        alias_note=alias_note,
        lyrics_a=lyrics_to_text(lyrics.get("A", [])),
        lyrics_b=lyrics_to_text(lyrics.get("B", [])),
        brief_section=brief_section,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_prepare(path_a, path_b, overrides):
    from mashup_maker import separate_full_song
    work_dir = "downloads"
    os.makedirs(work_dir, exist_ok=True)

    songs, grids, names = {}, {}, {}
    for sid, path in (("A", path_a), ("B", path_b)):
        print(f"== Song {sid}: {os.path.basename(path)}")
        meta = analyze_song(path, overrides.get(f"bpm_{sid.lower()}"),
                            overrides.get(f"key_{sid.lower()}"))
        print(f"   {meta['key']} | {meta['bpm']:.1f} BPM | anchor {meta['anchor']:.1f}s")
        stems = separate_full_song(path, work_dir)
        if not stems:
            sys.exit(f"Separation failed for {path}")
        songs[sid] = {"stems": stems, "bpm": meta["bpm"],
                      "grid_start": meta["anchor"] % (BEATS_PER_BAR * 60.0 / meta["bpm"]),
                      "key": meta["key"], "shift": 0}
        names[sid] = os.path.splitext(os.path.basename(path))[0]

    sa, sb = compute_shifts(songs["A"]["key"], songs["B"]["key"])
    songs["A"]["shift"], songs["B"]["shift"] = sa, sb
    print(f"== Pitch plan: A {sa:+d} st, B {sb:+d} st (B is tonal anchor)")

    lyrics = {}
    for sid in ("A", "B"):
        print(f"== Bar grid {sid}...")
        grids[sid] = bar_grid(songs[sid])
        print(f"== Transcribing vocals {sid} (Whisper)...")
        segs = transcribe_lyrics(songs[sid]["stems"]["vocals"])
        lyrics[sid] = lyrics_to_bar_lines(segs, songs[sid]["bpm"], songs[sid]["grid_start"])
        print(f"   {len(lyrics[sid])} lyric lines")

    with open("dj_session.json", "w") as f:
        json.dump({"songs": songs, "names": names, "lyrics": lyrics}, f, indent=2)
    with open("dj_prompt.txt", "w", encoding="utf-8") as f:
        f.write(build_prompt(songs, grids, names, lyrics))
    print("\nWrote dj_session.json and dj_prompt.txt")
    print("NEXT: paste dj_prompt.txt into an LLM, save its JSON reply as plan.json,")
    print("      then run:  python llm_dj.py render dj_session.json plan.json")


def extract_plan_json(raw):
    """Pull the plan JSON out of an LLM reply that may contain reasoning prose,
    markdown fences, or trailing commentary. Finds the LAST occurrence of
    "target_bpm", backtracks to its opening brace, and brace-matches forward."""
    raw = raw.strip()
    key = raw.rfind('"target_bpm"')
    if key == -1:
        return json.loads(raw)                     # plain JSON or fail loudly
    start = raw.rfind("{", 0, key)
    if start == -1:
        raise ValueError("Found target_bpm but no opening brace before it.")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start:i + 1])
    raise ValueError("Unbalanced braces in plan JSON.")


def cmd_render(session_path, plan_path):
    with open(session_path) as f:
        session = json.load(f)
    plan = extract_plan_json(open(plan_path).read())

    songs = session["songs"]
    sr = int(os.environ.get("DJ_SR", "44100"))
    errors, warnings = validate_plan(plan, songs, sr=sr)
    for w in warnings:
        print("WARN:", w)
    if errors:
        print("\nPLAN REJECTED -- paste these errors back to the LLM and ask it to fix the JSON:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    from critic import check_phrase_integrity, critique_render
    for p in check_phrase_integrity(plan, session):
        print("PHRASE:", p)

    print("Plan valid. Idea:", plan.get("comment", "(none)"))
    out, report = render_plan(plan, songs, "llm_mashup.wav", sr=sr)
    for r in report:
        print("  ", r)

    res = critique_render(out, plan, session)
    print("\n" + res["report"])
    print(f"\nDone -> {out}")


def cmd_reprompt(session_path, brief=None):
    """Rebuild dj_prompt.txt from an existing session (no re-analysis, no re-whisper).
    Optional brief: {"text": str, "arc": str} -- persisted into the session so the
    critic verifies the arc on the next render."""
    with open(session_path) as f:
        session = json.load(f)
    songs, names = session["songs"], session["names"]
    lyrics = session.get("lyrics", {"A": [], "B": []})
    if brief is not None:
        session["brief"] = brief
        with open(session_path, "w") as f:
            json.dump(session, f, indent=2)
    grids = {}
    for sid in ("A", "B"):
        print(f"== Bar grid {sid} (recompute)...")
        grids[sid] = bar_grid(songs[sid])
    with open("dj_prompt.txt", "w", encoding="utf-8") as f:
        f.write(build_prompt(songs, grids, names, lyrics, session.get("brief")))
    print("Wrote dj_prompt.txt" + (" (with creative brief)." if session.get("brief") else "."))


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 3 and args[0] == "prepare":
        ov = {}
        rest = args[3:]
        for i in range(0, len(rest) - 1, 2):
            k = rest[i].lstrip("-").replace("-", "_")
            ov[k] = float(rest[i + 1]) if "bpm" in k else rest[i + 1]
        cmd_prepare(args[1], args[2], ov)
    elif len(args) == 3 and args[0] == "render":
        cmd_render(args[1], args[2])
    elif args and args[0] == "reprompt" and len(args) >= 2:
        brief, rest = {}, args[2:]
        i = 0
        while i < len(rest) - 1:
            if rest[i] == "--brief":
                brief["text"] = rest[i + 1]
            elif rest[i] == "--arc":
                brief["arc"] = rest[i + 1]
            i += 2
        cmd_reprompt(args[1], brief or None)
    else:
        print(__doc__)
        