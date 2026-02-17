import onnxruntime as ort

print(f"ONNX Runtime Version: {ort.__version__}")
print(f"Available Providers: {ort.get_available_providers()}")

try:
    # Try to force DirectML
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 0 # Verbose logging
    print("\nAttempting to load DirectML...")
    
    # We create a dummy session to test initialization
    # (This doesn't need a real model, just checks the hardware link)
    providers = ['DmlExecutionProvider']
    print(f"Requesting: {providers}")
    
except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")

if 'DmlExecutionProvider' in ort.get_available_providers():
    print("\n✅ SUCCESS: DirectML is installed and visible!")
else:
    print("\n❌ FAILURE: DirectML is NOT visible. You are running on CPU.")