import os
import nudenet
print(f"NudeNet is installed at: {os.path.dirname(nudenet.__file__)}")
print(f"Files in folder: {os.listdir(os.path.dirname(nudenet.__file__))}")

# Try to see what is actually importable
print("\nAttempting to find classes...")
try:
    print(f"Available attributes: {dir(nudenet)}")
except:
    print("Could not dir(nudenet)")