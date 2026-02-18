import pandas as pd
import random
import streamlit as st
import os
import ast
import math

class MashupRecommender:
    def __init__(self, csv_path="tracks.csv"):
        self.csv_path = csv_path
        self.df = self.load_data()
        self.major_map = {0:8, 1:3, 2:10, 3:5, 4:12, 5:7, 6:2, 7:9, 8:4, 9:11, 10:6, 11:1}
        self.minor_map = {0:5, 1:12, 2:7, 3:2, 4:9, 5:4, 6:11, 7:6, 8:1, 9:8, 10:3, 11:10}

    @st.cache_data
    def load_data(_self):
        if not os.path.exists(_self.csv_path): return pd.DataFrame()
        df = pd.read_csv(_self.csv_path)
        cols = ['key', 'mode', 'tempo', 'energy', 'speechiness', 'popularity', 'valence', 'acousticness']
        for c in cols: 
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['track_name', 'artists', 'tempo', 'energy', 'valence'])

    def clean_artist(self, s):
        s = str(s).strip()
        if s.startswith("['") and s.endswith("']"):
            try: return ", ".join(ast.literal_eval(s))
            except: pass
        return s.replace(";", ", ").replace("['", "").replace("']", "")

    def get_camelot(self, k, m):
        h = self.major_map.get(int(k)) if m == 1 else self.minor_map.get(int(k))
        return h, ('B' if m == 1 else 'A')

    def get_vibe_distance(self, row_a, row_b):
        # 1. Acoustic Penalty
        if abs(row_a.get('acousticness', 0) - row_b.get('acousticness', 0)) > 0.6:
            return 100 
        # 2. Vibe Vector
        dist = math.sqrt((row_a['energy'] - row_b['energy'])**2 + (row_a['valence'] - row_b['valence'])**2)
        return (dist / 1.41) * 30

    def discover_mashups(self, n=5):
        if self.df.empty: return []
        
        # --- POOLS ---
        df_voc = self.df[(self.df['speechiness'] > 0.04) & (self.df['popularity'] > 45)]
        df_beat = self.df[(self.df['energy'] > 0.55) & (self.df['popularity'] > 45)]
        
        pool_a = df_voc.sample(n=min(150, len(df_voc)), replace=False)
        pool_b = df_beat.sample(n=min(150, len(df_beat)), replace=False)
        
        candidates = []
        
        for _, a in pool_a.iterrows():
            for _, b in pool_b.iterrows():
                if a['track_name'] == b['track_name']: continue
                
                # --- 1. STRICT BPM LIMIT (±10%) ---
                ratio = a['tempo'] / b['tempo']
                is_safe = 0.90 <= ratio <= 1.10
                is_trap = 1.98 <= ratio <= 2.02
                is_dub = 0.49 <= ratio <= 0.51
                
                if not (is_safe or is_trap or is_dub): continue
                
                # --- 2. STRICT HARMONIC LIMIT (±2 Semitones) ---
                best_harm = 0
                for shift in range(-2, 3): 
                    kb_shift = (int(b['key']) + shift) % 12
                    ha, _ = self.get_camelot(a['key'], a['mode'])
                    hb, _ = self.get_camelot(kb_shift, b['mode'])
                    dist = min(abs(ha - hb), 12 - abs(ha - hb))
                    
                    s = 100 if dist == 0 else (85 if dist == 1 else 0)
                    s -= abs(shift) * 10 
                    best_harm = max(best_harm, s)
                
                if best_harm < 70: continue
                
                # Vibe Check
                vibe_pen = self.get_vibe_distance(a, b)
                final = best_harm - vibe_pen
                
                if final > 75:
                    candidates.append({'score': final, 'a': a, 'b': b})

        # --- SELECTION ---
        candidates.sort(key=lambda x: x['score'], reverse=True)
        top = candidates[:60]
        random.shuffle(top)
        
        final_list = []
        used = set()
        
        for c in top:
            aa, bb = self.clean_artist(c['a']['artists']), self.clean_artist(c['b']['artists'])
            ta, tb = c['a']['track_name'], c['b']['track_name']
            
            # Dedupe
            pair_id = tuple(sorted([ta, tb]))
            if pair_id in used: continue
            
            ap, bp = aa.split(',')[0], bb.split(',')[0]
            if ap in used or bp in used: continue
            
            # FIXED LINE HERE: Removed 'b_prim='
            used.add(pair_id); used.add(ap); used.add(bp)
            
            final_list.append({
                "name": f"{aa} - {ta} x {bb} - {tb}",
                "description": f"Match: {int(c['score'])}% | {int(c['a']['tempo'])}↔{int(c['b']['tempo'])} BPM",
                "song_a": {"title": ta, "search_query": f"{aa} {ta} Audio"},
                "song_b": {"title": tb, "search_query": f"{bb} {tb} Audio"}
            })
            if len(final_list) >= n: break
            
        return final_list
    