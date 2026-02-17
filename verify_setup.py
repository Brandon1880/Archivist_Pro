import sys
import os
import importlib.util

# --- CONFIGURATION ---
# UPDATE THIS if Tesseract is installed elsewhere, as noted in the docs 
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def print_status(component, status, message=""):
    symbol = "✅" if status else "❌"
    print(f"{symbol} {component}: {message}")

print("--- Archivist Pro v1.0 Environment Verification ---\n")

# 1. CHECK PYTHON VERSION 
py_ver = sys.version_info
if py_ver.major == 3 and (py_ver.minor == 10 or py_ver.minor == 11):
    print_status("Python Version", True, f"{py_ver.major}.{py_ver.minor} (Compatible)")
else:
    print_status("Python Version", False, f"{py_ver.major}.{py_ver.minor} (Docs recommend 3.10 or 3.11)")

# 2. CHECK LIBRARIES IMPORT 
required_libs = [
    "customtkinter", "cv2", "moviepy", "fpdf", 
    "pytesseract", "nudenet", "PIL"
]
missing_libs = []

for lib in required_libs:
    if importlib.util.find_spec(lib) is None:
        missing_libs.append(lib)

if not missing_libs:
    print_status("Library Imports", True, "All libraries installed.")
else:
    print_status("Library Imports", False, f"Missing: {', '.join(missing_libs)}")

# 3. CHECK TESSERACT OCR 
try:
    import pytesseract
    from PIL import Image, ImageDraw, ImageFont
    
    if not os.path.exists(TESSERACT_PATH):
        raise FileNotFoundError(f"Executable not found at {TESSERACT_PATH}")
        
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    
    # Create a dummy image with text to test OCR
    img = Image.new('RGB', (100, 30), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "TEST", fill=(0, 0, 0))
    
    text = pytesseract.image_to_string(img)
    if "TEST" in text:
        print_status("Tesseract OCR", True, "Successfully read text from image.")
    else:
        print_status("Tesseract OCR", False, "Ran, but failed to detect text.")
except Exception as e:
    print_status("Tesseract OCR", False, f"Error: {e}")

# 4. CHECK NUDENET (AI MODEL) 
try:
    print("   ...Initializing NudeNet (this may take a moment)...")
    from nudenet import NudeDetector
    # Initialize detector to trigger model check/load
    detector = NudeDetector()
    print_status("NudeNet AI", True, "Model loaded successfully.")
except Exception as e:
    print_status("NudeNet AI", False, f"Failed to load. Error: {e}")

# 5. CHECK FFMPEG / MOVIEPY 
try:
    from moviepy.config import get_setting
    ffmpeg_path = get_setting("EXEC_FFMPEG")
    if ffmpeg_path:
        print_status("FFmpeg/MoviePy", True, f"Found binary at: {ffmpeg_path}")
    else:
        print_status("FFmpeg/MoviePy", False, "FFmpeg binary not found.")
except Exception as e:
    print_status("FFmpeg/MoviePy", False, f"Error: {e}")

print("\n--- Verification Complete ---")
if "Failed" in str(sys.stdout): 
    print("Please resolve the ❌ errors before running scanner.py.")
else:
    print("System is ready! You can now run the main application.")