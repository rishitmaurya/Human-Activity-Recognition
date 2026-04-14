# diagnose_pykinect.py
"""
Diagnoses pykinect2 installation issues.
Run: python diagnose_pykinect.py
"""
import sys
import os

print("=" * 60)
print("  PYKINECT2 DIAGNOSTIC TOOL")
print("=" * 60)

# Check Python version
print(f"\nPython: {sys.version}")
print(f"Platform: {sys.platform}")
bits = 8 * 8 if sys.maxsize > 2**32 else 32
print(f"Architecture: {bits}-bit")

# Check comtypes
print("\n--- COMTYPES CHECK ---")
try:
    import comtypes
    print(f"✓ comtypes installed: {comtypes.__version__}")
    
    # Check if version is compatible
    import packaging.version
    ver = packaging.version.parse(comtypes.__version__)
    if ver >= packaging.version.parse("1.2.0"):
        print(f"✗ INCOMPATIBLE VERSION!")
        print(f"  comtypes {comtypes.__version__} is too new for pykinect2")
        print(f"  FIX: pip uninstall comtypes -y && pip install comtypes==1.1.14")
    else:
        print(f"✓ comtypes version is compatible")
        
except ImportError:
    print("✗ comtypes NOT installed")
    print("  FIX: pip install comtypes==1.1.14")
except Exception as e:
    print(f"✗ Error checking comtypes: {e}")

# Check pykinect2
print("\n--- PYKINECT2 CHECK ---")
try:
    import pykinect2
    print(f"✓ pykinect2 installed at: {os.path.dirname(pykinect2.__file__)}")
except ImportError as e:
    print(f"✗ pykinect2 NOT installed: {e}")
    print("  FIX: pip install pykinect2")
except Exception as e:
    print(f"✗ Error: {e}")

# Try importing PyKinectV2
print("\n--- PYKINECTV2 MODULE CHECK ---")
try:
    from pykinect2 import PyKinectV2
    print("✓ PyKinectV2 imports successfully")
except ImportError as e:
    print(f"✗ Cannot import PyKinectV2: {e}")
    if "Wrong version" in str(e):
        print("\n  DIAGNOSIS: comtypes version conflict")
        print("  SOLUTION:")
        print("    pip uninstall comtypes -y")
        print("    pip install comtypes==1.1.14")
        print("    (Then restart terminal)")
except AssertionError as e:
    print(f"✗ Assertion error (64-bit struct issue): {e}")
    print("  Run: python fix_pykinect2_final.py")
except Exception as e:
    print(f"✗ Other error: {type(e).__name__}: {e}")

# Try importing PyKinectRuntime
print("\n--- PYKINECTRUNTIME MODULE CHECK ---")
try:
    from pykinect2 import PyKinectRuntime
    print("✓ PyKinectRuntime imports successfully")
except Exception as e:
    print(f"✗ Cannot import PyKinectRuntime: {e}")

# Check Kinect SDK
print("\n--- KINECT SDK CHECK ---")
sdk_env = os.environ.get('KINECTSDK20_DIR', '')
if sdk_env:
    print(f"✓ SDK env var set: {sdk_env}")
    if os.path.exists(sdk_env):
        print(f"✓ SDK directory exists")
    else:
        print(f"✗ SDK directory doesn't exist")
else:
    print("✗ KINECTSDK20_DIR not set")
    paths = [
        r"C:\Program Files\Microsoft SDKs\Kinect\v2.0_1409",
        r"C:\Program Files (x86)\Microsoft SDKs\Kinect\v2.0_1409",
    ]
    for p in paths:
        if os.path.exists(p):
            print(f"✓ Found SDK at: {p}")
            break
    else:
        print("✗ Kinect SDK 2.0 not found")
        print("  Download: https://www.microsoft.com/en-us/download/details.aspx?id=44561")

# Final verdict
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

all_ok = True
try:
    import comtypes
    import packaging.version
    if packaging.version.parse(comtypes.__version__) >= packaging.version.parse("1.2.0"):
        all_ok = False
        print("✗ ISSUE: comtypes version too new")
except:
    all_ok = False
    print("✗ ISSUE: comtypes not properly installed")

try:
    from pykinect2 import PyKinectV2
    from pykinect2 import PyKinectRuntime
    print("✓ pykinect2 modules import successfully")
except Exception as e:
    all_ok = False
    print(f"✗ ISSUE: pykinect2 import fails: {e}")

if all_ok:
    print("\n✓ ALL CHECKS PASSED!")
    print("  Your pykinect2 installation is ready.")
else:
    print("\n✗ ISSUES FOUND - See above for fixes")

print("=" * 60)