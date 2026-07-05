import os
import pandas as pd
import yt_dlp
import re

# --- PATHS (Adjust these to where your Google Drive Desktop is mounted) ---
# Example for Windows: "G:\My Drive\Mashup_Vault"
DRIVE_DIR = r"https://drive.google.com/drive/folders/19HQ5OMzBEAG6csXYpBcTMcK7h2dVYljL?usp=sharing" 
CSV_PATH = os.path.join(DRIVE_DIR, "tracks.csv")
TO_PROCESS_DIR = os.path.join(DRIVE_DIR, "To_Process")

os.makedirs(TO_PROCESS_DIR, exist_ok=True)

print("📊 Loading dataset...")
df = pd.read_csv(CSV_PATH, low_memory=False)
if 'name' in df.columns: df.rename(columns={'name': 'track_name'}, inplace=True)
df = df.dropna(subset=['track_name', 'artists'])

# Simple deduplication
df['core_title'] = df['track_name'].apply(lambda x: re.sub(r' \(feat\..*?\)', '', str(x).lower()).strip())
df['core_artist'] = df['artists'].apply(lambda x: str(x).replace("['", "").replace("']", "").replace("'", "").split(',')[0].strip().lower())
df = df.drop_duplicates(subset=['core_artist', 'core_title'])

def clean_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

for index, row in df.iterrows():
    artist = row['core_artist'].title()
    title = row['core_title'].title()
    
    search_query = f"{artist} {title} audio"
    file_name = clean_filename(f"{artist} - {title}")
    target_path = os.path.join(TO_PROCESS_DIR, f"{file_name}.wav")
    
    # Check if we already downloaded it or if Colab already processed it
    if os.path.exists(target_path) or os.path.exists(os.path.join(DRIVE_DIR, "stems_4", file_name)):
        continue
        
    print(f"⬇️ Downloading to Drive: {file_name}")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(TO_PROCESS_DIR, f'{file_name}.%(ext)s'),
        'postprocessors':[{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'quiet': True, 'noplaylist': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(f"ytsearch1:{search_query}", download=True)
    except Exception as e:
        print(f"❌ Failed {file_name}")