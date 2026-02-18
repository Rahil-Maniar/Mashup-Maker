# presets.py

PRESET_MASHUPS = [
    {
        "name": "The 'Industry' Standard (Pop x EDM)",
        "description": "Lil Nas X vocals on a heavy Big Room drop. Classic high-energy switch.",
        "song_a": {
            "query": "Lil Nas X - Industry Baby Official Audio",
            "url": "https://www.youtube.com/watch?v=UTHLKHL_whs", # Example URL, search usually better
            "cut_time": 35.0 # Approx vocal end before drop
        },
        "song_b": {
            "query": "Martin Garrix - Animals Official Video",
            "url": "https://www.youtube.com/watch?v=gCYcHz2k5x0",
            "drop_time": 88.0 # Approx drop start
        }
    },
    {
        "name": "The Nostalgia Switch (2000s x Modern)",
        "description": "Cascada vocals on a modern Tech House beat. Works perfectly in A Minor.",
        "song_a": {
            "query": "Cascada - Everytime We Touch (Acapella)", # Acapellas work best if found
            "url": "https://www.youtube.com/watch?v=4G6QDNC4jPs", 
            "cut_time": 45.0
        },
        "song_b": {
            "query": "Fisher - Losing It Official Audio",
            "url": "https://www.youtube.com/watch?v=RSZC6er9I1g",
            "drop_time": 60.5
        }
    },
    {
        "name": "Hip Hop x Dubstep (The Headbanger)",
        "description": "Kendrick Lamar vocals on Skrillex. Aggressive energy.",
        "song_a": {
            "query": "Kendrick Lamar - Humble Audio",
            "url": "https://www.youtube.com/watch?v=tvTRZJ-4EyI",
            "cut_time": 15.0
        },
        "song_b": {
            "query": "Skrillex - Bangarang",
            "url": "https://www.youtube.com/watch?v=YJVmu6yttiw",
            "drop_time": 25.5
        }
    }
]