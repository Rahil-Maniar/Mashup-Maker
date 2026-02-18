import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st

class SpotifyMatcher:
    def __init__(self, client_id, client_secret):
        try:
            self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            ))
            self.auth_success = True
        except Exception as e:
            self.auth_success = False
            print(f"Spotify Auth Failed: {e}")

    def get_spotify_key_code(self, key_str):
        """Maps 'C Major' -> (0, 1) for Spotify API"""
        # Pitch class map
        pitch_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        
        try:
            root, scale = key_str.split(' ')
            key_code = pitch_map.get(root, 0)
            mode_code = 1 if scale.lower() == 'major' else 0
            return key_code, mode_code
        except:
            return None, None

    def get_recommendations(self, target_key_str, target_bpm, limit=3):
        if not self.auth_success:
            return []

        key_code, mode_code = self.get_spotify_key_code(target_key_str)
        
        try:
            # We request 10 tracks, then sort by popularity to give the "Best" ones
            recs = self.sp.recommendations(
                seed_genres=['edm', 'house', 'dance', 'pop'], # Generic seeds for drops
                target_key=key_code,
                target_mode=mode_code,
                target_tempo=target_bpm,
                min_energy=0.7, # We want "Drops", so high energy
                limit=10 
            )
            
            tracks = []
            for t in recs['tracks']:
                tracks.append({
                    'name': t['name'],
                    'artist': t['artists'][0]['name'],
                    'image': t['album']['images'][0]['url'] if t['album']['images'] else None,
                    'spotify_url': t['external_urls']['spotify'],
                    'popularity': t['popularity']
                })
            
            # Sort by popularity (Highest first) and take top N
            tracks.sort(key=lambda x: x['popularity'], reverse=True)
            return tracks[:limit]
            
        except Exception as e:
            st.error(f"Spotify API Error: {e}")
            return []