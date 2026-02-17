# --- SIMULATION SCRIPT ---
# This simulates what the AI *would* output if it saw something
# to test if your code's filters are broken.

# 1. These are the filters currently in your scanner.py
FILTERS_IN_CODE = [
    "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED", "FEMALE_BREAST_EXPOSED", "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_COVERED", "FEMALE_GENITALIA_COVERED",
    "MALE_GENITALIA_COVERED", "BUTTOCKS_COVERED", "ANUS_COVERED",
    "BELLY_EXPOSED", "BELLY_COVERED", "MALE_BREAST_EXPOSED",
    "FEET_EXPOSED", "FEET_COVERED", "ARMPITS_EXPOSED", "ARMPITS_COVERED"
]

print("--- FILTER LOGIC TEST ---")

# 2. Simulate what NudeNet actually outputs (lowercase labels)
# NudeNet typically returns labels like: "female_breast_exposed"
simulated_ai_output = "female_breast_exposed" 

print(f"1. AI Detection Simulated: '{simulated_ai_output}'")

# 3. Test if the current code catches it
if simulated_ai_output in FILTERS_IN_CODE:
    print(f"2. Current Code Check: ✅ PASSED (Hit found)")
else:
    print(f"2. Current Code Check: ❌ FAILED (Hit ignored)")
    print(f"   Reason: '{simulated_ai_output}' does not equal '{simulated_ai_output.upper()}'")

# 4. Test the fix (Case-Insensitive Check)
# This simulates the fix I will give you
normalized_filters = [f.lower() for f in FILTERS_IN_CODE]

if simulated_ai_output.lower() in normalized_filters:
    print(f"3. Proposed Fix Check: ✅ PASSED (Hit found with fix)")
else:
    print(f"3. Proposed Fix Check: ❌ FAILED")

print("-------------------------")