"""
studio.py -- Mashup Studio (localhost). One app for the whole loop:
roll compatible pairs OR search any two songs -> download -> prepare
(separate/analyze/lyrics) -> blend verdict -> LLM DJ prompt -> paste plan
(or auto-plan) -> render -> listen.

Run:  streamlit run studio.py
"""

import os
import json
import time
import streamlit as st

from studio_logic import blend_report, auto_plan, file_hash
from dsl_renderer import validate_plan, render_plan
from get_song import search as yt_search, download as yt_download

st.set_page_config(page_title="Mashup Studio", layout="wide", page_icon="🎛️")
WORK_DIR = "downloads"
os.makedirs(WORK_DIR, exist_ok=True)

SS = st.session_state
for k, v in {"pair": {}, "session": None, "prompt": None,
             "search_A": [], "search_B": [], "roll": None}.items():
    if k not in SS:
        SS[k] = v


# ---------------------------------------------------------------------------
# Pipeline steps (heavy imports kept lazy so the app opens instantly)
# ---------------------------------------------------------------------------
def prepare_pair(path_a, path_b, name_a, name_b):
    from llm_dj import analyze_song, compute_shifts, bar_grid, \
        transcribe_lyrics, lyrics_to_bar_lines, build_prompt
    from mashup_maker import separate_full_song

    if file_hash(path_a) == file_hash(path_b):
        st.error("Both slots contain the SAME audio file - load two different songs.")
        return

    songs, grids, lyrics, names = {}, {}, {}, {"A": name_a, "B": name_b}
    with st.status("🚀 Preparing pair...", expanded=True) as status:
        for sid, path in (("A", path_a), ("B", path_b)):
            t0 = time.time()
            status.write(f"🧠 Analyzing {sid}: {names[sid]}")
            meta = analyze_song(path)
            status.write(f"   {meta['key']} | {meta['bpm']:.1f} BPM  ({time.time()-t0:.1f}s)")

            t0 = time.time()
            status.write(f"🎚️ Separating {sid} (cached after first time)...")
            stems = separate_full_song(path, WORK_DIR)
            if not stems:
                st.error(f"Separation failed for {names[sid]}")
                return
            status.write(f"   stems ready ({time.time()-t0:.1f}s)")

            songs[sid] = {"stems": stems, "bpm": meta["bpm"],
                          "grid_start": meta["anchor"] % (4 * 60.0 / meta["bpm"]),
                          "key": meta["key"], "shift": 0}

        sa, sb = compute_shifts(songs["A"]["key"], songs["B"]["key"])
        songs["A"]["shift"], songs["B"]["shift"] = sa, sb

        for sid in ("A", "B"):
            t0 = time.time()
            status.write(f"📊 Bar grid + 🎤 lyrics {sid}...")
            grids[sid] = bar_grid(songs[sid])
            segs = transcribe_lyrics(songs[sid]["stems"]["vocals"])
            lyrics[sid] = lyrics_to_bar_lines(segs, songs[sid]["bpm"], songs[sid]["grid_start"])
            status.write(f"   {grids[sid]['n_bars']} bars, {len(lyrics[sid])} lyric lines "
                         f"({time.time()-t0:.1f}s)")

        SS.session = {"songs": songs, "names": names, "grids": grids, "lyrics": lyrics}
        SS.prompt = build_prompt(songs, grids, names, lyrics)
        with open("dj_session.json", "w") as f:
            json.dump(SS.session, f, indent=2)
        status.update(label="✅ Pair ready", state="complete", expanded=False)


def load_by_query(query, slot):
    """Search YouTube, download top pick, put it in slot A/B."""
    with st.spinner(f'Searching "{query}"...'):
        results = yt_search(query + " audio")
    if not results:
        st.error("No results.")
        return
    r = results[0]
    with st.spinner(f"Downloading: {r['title']}"):
        path = yt_download(r["url"], name_hint=r["title"])
    if path:
        SS.pair[slot] = {"path": path, "name": r["title"]}


# ---------------------------------------------------------------------------
# SIDEBAR: roll pairs + manual search
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🧬 Roll a pair")
    if st.button("Roll compatible pairs", type="primary"):
        from recommender import MashupRecommender
        rec = MashupRecommender("tracks.csv")
        SS.roll = rec.discover_mashups(5)
    if SS.roll:
        for i, c in enumerate(SS.roll):
            st.markdown(f"**{c['name']}**")
            st.caption(c["description"])
            if st.button("Load this pair", key=f"roll_{i}"):
                load_by_query(c["song_a"]["search_query"], "A")
                load_by_query(c["song_b"]["search_query"], "B")
                st.rerun()
            st.divider()

    st.header("🔎 Or pick songs")
    for slot in ("A", "B"):
        q = st.text_input(f"Song {slot} (name or artist+title)", key=f"q_{slot}")
        if st.button(f"Search {slot}", key=f"s_{slot}") and q:
            SS[f"search_{slot}"] = yt_search(q + " audio")
        if SS[f"search_{slot}"]:
            opts = [f"{r['title']}  ({(r['duration'] or 0)//60}:{(r['duration'] or 0)%60:02d})"
                    for r in SS[f"search_{slot}"]]
            pick = st.selectbox(f"Pick {slot}", range(len(opts)),
                                format_func=lambda i: opts[i], key=f"pick_{slot}")
            if st.button(f"Download {slot}", key=f"d_{slot}"):
                r = SS[f"search_{slot}"][pick]
                with st.spinner("Downloading..."):
                    path = yt_download(r["url"], name_hint=r["title"])
                if path:
                    SS.pair[slot] = {"path": path, "name": r["title"]}
                    st.rerun()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
st.title("🎛️ Mashup Studio")

c1, c2 = st.columns(2)
for slot, col in (("A", c1), ("B", c2)):
    with col:
        st.subheader(f"Song {slot}" + (" (vocal lead)" if slot == "A" else " (tonal anchor)"))
        if slot in SS.pair:
            st.success(SS.pair[slot]["name"])
            st.audio(SS.pair[slot]["path"])
        else:
            st.info("Load via sidebar: roll a pair or search.")

if "A" in SS.pair and "B" in SS.pair:
    if st.button("⚗️ Prepare pair (separate + analyze + lyrics)", type="primary"):
        prepare_pair(SS.pair["A"]["path"], SS.pair["B"]["path"],
                     SS.pair["A"]["name"], SS.pair["B"]["name"])

if SS.session:
    s = SS.session["songs"]

    # --- Blend verdict ---
    st.divider()
    st.subheader("🔬 Blend verdict")
    rep = blend_report(s["A"]["key"], s["B"]["key"], s["A"]["bpm"], s["B"]["bpm"], s["A"]["shift"])
    st.metric("Compatibility", f"{rep['score']}/100")
    st.write(rep["verdict"])
    for name, level, msg in rep["checks"]:
        icon = {"ok": "✅", "warn": "⚠️", "bad": "❌"}[level]
        st.write(f"{icon} **{name}:** {msg}")

    # --- DJ loop ---
    st.divider()
    st.subheader("🎧 The DJ")

    from critic import ARC_TEMPLATES
    bc1, bc2, bc3 = st.columns([3, 2, 1])
    brief_text = bc1.text_input("Creative brief (optional)",
                                placeholder='e.g. "melancholic, like a late-night drive" '
                                            'or "frame it as an argument"')
    arc_names = ["(none)"] + list(ARC_TEMPLATES.keys())
    arc_pick = bc2.selectbox("Energy arc", arc_names,
                             help={k: v["description"] for k, v in ARC_TEMPLATES.items()}.get(
                                 "slow_burn", "shape of loudness over the edit"))
    if bc3.button("Apply brief"):
        from llm_dj import build_prompt
        brief = {}
        if brief_text.strip():
            brief["text"] = brief_text.strip()
        if arc_pick != "(none)":
            brief["arc"] = arc_pick
        SS.session["brief"] = brief
        SS.prompt = build_prompt(SS.session["songs"], SS.session["grids"],
                                 SS.session["names"], SS.session["lyrics"], brief)
        with open("dj_session.json", "w") as f:
            json.dump(SS.session, f, indent=2)
        st.success("Brief applied - prompt rebuilt (arc will be verified after render)."
                   if brief else "Brief cleared.")

    with st.expander("1) Copy this prompt into your LLM (claude.ai etc.)"):
        st.code(SS.prompt, language=None)

    plan_text = st.text_area("2) Paste the plan JSON here", height=200,
                             placeholder='{"target_bpm": ..., "timeline": [...]}')
    b1, b2, b3 = st.columns(3)
    plan = None
    if b1.button("Validate plan") and plan_text.strip():
        from llm_dj import extract_plan_json
        try:
            plan = extract_plan_json(plan_text)
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Could not find a valid plan JSON in that text: {e}")
        if plan:
            errs, warns = validate_plan(plan, s)
            for w in warns:
                st.warning(w)
            if errs:
                st.error("Plan rejected - paste these back to the LLM:\n" +
                         "\n".join("- " + e for e in errs))
            else:
                from critic import check_phrase_integrity
                phrase_issues = check_phrase_integrity(plan, SS.session)
                if phrase_issues:
                    st.warning("Plan is renderable, but clips these lyric phrases "
                               "(paste to the LLM to fix, or render anyway):\n" +
                               "\n".join("- " + p for p in phrase_issues))
                st.success("Plan valid. Idea: " + plan.get("comment", "(none)"))
                SS["plan_ok"] = plan

    if b2.button("🎲 No LLM? Auto-arrange"):
        SS["plan_ok"] = auto_plan(SS.session)
        st.success("Auto-plan built: " + SS["plan_ok"]["comment"])

    if b3.button("🎛️ Render mashup", type="primary"):
        plan = SS.get("plan_ok")
        if not plan:
            st.error("Validate a plan (or auto-arrange) first.")
        else:
            with st.status("Rendering...") as status:
                out, report = render_plan(plan, s, "llm_mashup.wav")
                for r in report:
                    status.write(r)
                status.update(label="✨ Rendered", state="complete")
            st.audio(out)
            with open(out, "rb") as f:
                st.download_button("⬇️ Download mashup", f, file_name="mashup.wav")
            from critic import critique_render
            res = critique_render(out, plan, SS.session)
            with st.expander("🧾 Critic report (auto-measured -- paste to your LLM to revise)",
                             expanded=bool(res["flags"] or res["phrase_issues"])):
                st.code(res["report"], language=None)
                