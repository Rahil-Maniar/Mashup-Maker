"""
list_all_pairs.py -- enumerate EVERY pair the recommender could ever suggest.

Unlike the app (which randomly samples pools each roll), this walks the FULL
vocal-pool x beat-pool grid with the exact same filters, so you can see the
complete universe of qualifying pairs and why some repeat so often.

Usage:
    python list_all_pairs.py                 # uses chart_tracks.csv / tracks.csv
    python list_all_pairs.py mydata.csv      # explicit database
    python list_all_pairs.py --full          # don't cap huge pools (slow!)

Writes all pairs to all_pairs.csv (sorted by score) and prints a summary.
"""
import sys
import csv as csvmod
from collections import Counter

from recommender import MashupRecommender

POOL_CAP = 800   # per-pool cap for giant databases (override with --full)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full = "--full" in sys.argv
    src = args[0] if args else "tracks.csv"

    rec = MashupRecommender(src)
    df = rec.df
    if df.empty:
        print("No song database found (need chart_tracks.csv or tracks.csv).")
        return

    df_voc = df[(df["speechiness"] > 0.04) & (df["popularity"] > 45)]
    df_beat = df[(df["energy"] > 0.55) & (df["popularity"] > 45)]

    if not full:
        if len(df_voc) > POOL_CAP:
            df_voc = df_voc.nlargest(POOL_CAP, "popularity")
            print(f"(vocal pool capped to top {POOL_CAP} by popularity; --full to disable)")
        if len(df_beat) > POOL_CAP:
            df_beat = df_beat.nlargest(POOL_CAP, "popularity")
            print(f"(beat pool capped to top {POOL_CAP} by popularity; --full to disable)")

    print(f"Vocal pool: {len(df_voc)} tracks | Beat pool: {len(df_beat)} tracks "
          f"-> scanning {len(df_voc) * len(df_beat):,} combos...")

    pairs, seen = [], set()
    beat_rows = list(df_beat.itertuples())
    for a in df_voc.itertuples():
        a_d = a._asdict()
        for b in beat_rows:
            if rec.same_song(a.track_name, b.track_name):
                continue
            pid = tuple(sorted([a.track_name, b.track_name]))
            if pid in seen:
                continue

            # --- identical filters to discover_mashups ---
            ratio = b.tempo / a.tempo if a.tempo else 0.0
            if not (0.95 <= ratio <= 1.05):
                continue
            try:
                ka, kb = int(a.key), int(b.key)
            except Exception:
                continue
            if abs(((ka - kb + 6) % 12) - 6) > 1:
                continue
            best_harm = 0
            for shift in (-1, 0, 1):
                ha, _ = rec.get_camelot(ka, a.mode)
                hb, _ = rec.get_camelot((kb + shift) % 12, b.mode)
                if ha is None or hb is None:
                    continue
                dist = min(abs(ha - hb), 12 - abs(ha - hb))
                s = 100 if dist == 0 else (85 if dist == 1 else 0)
                best_harm = max(best_harm, s - abs(shift) * 10)
            if best_harm < 70:
                continue
            score = best_harm - rec.get_vibe_distance(a_d, b._asdict())
            if score <= 75:
                continue

            seen.add(pid)
            pairs.append({
                "score": int(score),
                "pop": int((float(a.popularity or 50) + float(b.popularity or 50)) / 2),
                "song_a": f"{rec.clean_artist(a.artists)} - {a.track_name}",
                "song_b": f"{rec.clean_artist(b.artists)} - {b.track_name}",
                "bpm": f"{int(a.tempo)}<->{int(b.tempo)}",
            })

    pairs.sort(key=lambda p: (p["score"], p["pop"]), reverse=True)

    with open("all_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csvmod.DictWriter(f, fieldnames=["score", "pop", "song_a", "song_b", "bpm"])
        w.writeheader()
        w.writerows(pairs)

    print(f"\n{len(pairs)} unique qualifying pairs total -> saved to all_pairs.csv\n")
    print("Top 30 by score:")
    for p in pairs[:30]:
        print(f"  [{p['score']:>3}] pop {p['pop']:>3} | {p['song_a']}  x  {p['song_b']}  ({p['bpm']} BPM)")

    # which songs dominate? (the repetition drivers)
    cnt = Counter()
    for p in pairs:
        cnt[p["song_a"]] += 1
        cnt[p["song_b"]] += 1
    print("\nMost-connected songs (appear in the most pairs -> most likely to repeat):")
    for name, c in cnt.most_common(10):
        print(f"  {c:>4} pairs: {name}")


if __name__ == "__main__":
    main()
    