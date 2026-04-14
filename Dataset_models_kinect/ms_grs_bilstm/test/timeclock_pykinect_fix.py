# fix_pykinect_timeclock.py
"""
Fixes time.clock() deprecation in PyKinectRuntime.py
Python 3.8+ removed time.clock(), but pykinect2 still uses it.

Run: python fix_pykinect_timeclock.py
"""
import os
import sys
import shutil

def main():
    print("=" * 70)
    print("  FIXING time.clock() IN PYKINECTRUNTIME")
    print("=" * 70)

    # Find pykinect2
    try:
        import pykinect2
        pkg_dir = os.path.dirname(pykinect2.__file__)
    except ImportError:
        print("[ERROR] pykinect2 not installed")
        sys.exit(1)

    print(f"\nPackage: {pkg_dir}")
    target = os.path.join(pkg_dir, "PyKinectRuntime.py")

    if not os.path.exists(target):
        print(f"[ERROR] {target} not found")
        sys.exit(1)

    # Read file
    with open(target, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Backup
    backup = target + ".time_backup"
    if not os.path.exists(backup):
        shutil.copy2(target, backup)
        print(f"Backup: {backup}")

    # Check if already fixed
    if 'time.perf_counter()' in content or 'TIME_CLOCK_FIX' in content:
        print("\n[INFO] Already fixed!")
    else:
        print("\nApplying fix...")
        
        # Replace time.clock() with time.perf_counter()
        original_content = content
        
        # Method 1: Direct replacement
        content = content.replace('time.clock()', 'time.perf_counter()')
        
        # Method 2: Also handle variations
        content = content.replace('time.clock ()', 'time.perf_counter()')
        
        # Add a marker comment at the top
        if content != original_content:
            # Find the first import statement
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('import '):
                    lines.insert(i + 1, 
                                 '# TIME_CLOCK_FIX: time.clock() → '
                                 'time.perf_counter() for Python 3.8+')
                    break
            content = '\n'.join(lines)
            
            replacements = original_content.count('time.clock()') + \
                          original_content.count('time.clock ()')
            
            # Write patched file
            with open(target, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"  ✓ Replaced {replacements} occurrence(s) of time.clock()")
            print(f"  ✓ Fixed: {target}")
        else:
            print("  [INFO] No time.clock() found (might be different format)")

    # Verification
    print("\n" + "=" * 70)
    print("  VERIFYING FIX")
    print("=" * 70)

    # Clear module cache
    mods_to_clear = [k for k in list(sys.modules.keys()) if 'pykinect' in k]
    for m in mods_to_clear:
        del sys.modules[m]

    try:
        from pykinect2 import PyKinectRuntime
        print("✓ PyKinectRuntime imports successfully!")
        
        print("\n" + "=" * 70)
        print("  ✓✓✓ SUCCESS! ✓✓✓")
        print("=" * 70)
        print("\nNow you can run:")
        print("  python realtime_har_kinect_v2.py --mode mock")
        print("  python realtime_har_kinect_v2.py --mode kinect")
        print("=" * 70)
        
    except AttributeError as e:
        if 'clock' in str(e).lower():
            print(f"✗ time.clock() still present: {e}")
            print("\n[MANUAL FIX REQUIRED]")
            print(f"\n1. Open: {target}")
            print("2. Find all: time.clock()")
            print("3. Replace with: time.perf_counter()")
            print("4. Save and run this script again")
        else:
            print(f"✗ Other error: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"✗ Import error: {type(e).__name__}: {e}")
        print("(This may be OK if Kinect isn't connected)")

if __name__ == "__main__":
    main()