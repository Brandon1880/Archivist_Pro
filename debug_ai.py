import os
import urllib.request

# 1. CONFIGURATION
# [cite_start]Matches the filters in your scanner.py exactly [cite: 45]
FILTERS_IN_CODE = [
    "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED", "FEMALE_BREAST_EXPOSED", "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_COVERED", "FEMALE_GENITALIA_COVERED",
    "MALE_GENITALIA_COVERED", "BUTTOCKS_COVERED", "ANUS_COVERED",
    "BELLY_EXPOSED", "BELLY_COVERED", "MALE_BREAST_EXPOSED",
    "FEET_EXPOSED", "FEET_COVERED", "ARMPITS_EXPOSED", "ARMPITS_COVERED"
]

print("--- DIAGNOSTIC START ---")

# 2. LOAD AI
try:
    print("1. Attempting to load NudeNet...")
    from nudenet import NudeDetector
    detector = NudeDetector()
    print("   ✅ NudeNet loaded successfully.")
except Exception as e:
    print(f"   ❌ CRITICAL FAILURE: NudeNet could not load.\n   Error: {e}")
    exit()

# 3. GET TEST IMAGE (With Browser Headers to fix 403 Error)
img_path = "test_debug.jpg"

# If you have your own image, rename it to 'test_debug.jpg' and put it in this folder
if os.path.exists(img_path):
    print(f"\n2. Found existing image: {img_path}")
else:
    print("\n2. Downloading test image (Statue of David)...")
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Michelangelo%27s_David_-_3.jpg/480px-Michelangelo%27s_David_-_3.jpg"
    
    # This header tricks the server into thinking we are a real Chrome browser
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(img_path, 'wb') as out_file:
            out_file.write(response.read())
        print("   ✅ Download complete.")
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        print("   -> WORKAROUND: Manually put ANY image in this folder and name it 'test_debug.jpg'")
        exit()

# 4. RUN DETECTION
print(f"\n3. Scanning image...")
detections = detector.detect(img_path)

# 5. VERIFY MATCHES
print("\n4. Verifying Filter Matches:")
if not detections:
    print("   ⚠ No detections found. (Try a clearer image)")
else:
    for i, det in enumerate(detections):
        label = det['label'] # Raw label from AI
        score = det['score']
        
        # KEY CHECK: Does the raw label exist in your hardcoded filter list?
        if label in FILTERS_IN_CODE:
            status = "✅ MATCH"
        else:
            status = "❌ MISMATCH"

        print(f"   Hit {i+1}: '{label}' (Score: {score:.2f}) -> {status}")
        
        if status == "❌ MISMATCH":
            print(f"      CRITICAL ERROR FOUND:")
            print(f"      The AI output '{label}' (lowercase/different format).")
            print(f"      But scanner.py expects '{label.upper()}' (uppercase).")
            print("      This is why your 12-hour scan found nothing.")

print("\n--- DIAGNOSTIC END ---")