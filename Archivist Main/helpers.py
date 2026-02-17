import os
import json
import ctypes
import threading
from moviepy import VideoFileClip

SETTINGS_FILE = "user_settings.json"

def resolve_path(path):
    abs_path = os.path.abspath(path)
    try:
        buffer = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(abs_path, buffer, 1024) > 0:
            return buffer.value
    except: pass 
    return "\\\\?\\" + abs_path if len(abs_path) > 240 else abs_path

def generate_clip(video_path, start_time, duration, output_path):
    try:
        with VideoFileClip(video_path) as clip:
            max_dur = clip.duration if clip.duration else (start_time + duration + 100)
            end_t = min(max_dur, start_time + (duration / 2))
            start_t = max(0, start_time - (duration / 2))
            new_clip = clip.subclipped(start_t, end_t)
            new_clip.write_videofile(output_path, codec='libx264', audio=False, preset='ultrafast', threads=4, logger=None)
    except Exception as e:
        print(f"Clip Generation Error: {e}")

def save_user_settings(data):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def load_user_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            try: return json.load(f)
            except: return {}
    return {}