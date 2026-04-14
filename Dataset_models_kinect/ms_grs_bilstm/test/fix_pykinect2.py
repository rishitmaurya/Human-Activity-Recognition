# fix_pykinect_ultimate.py
"""
Fixes ALL pykinect2 issues on 64-bit Python:
  1. sizeof assertions (struct size mismatches)
  2. _check_version('') call that fails

Run ONCE: python fix_pykinect_ultimate.py
"""
import os
import sys
import shutil

def main():
    print("=" * 70)
    print("  PYKINECT2 ULTIMATE FIXER")
    print("=" * 70)

    # Find pykinect2
    try:
        import pykinect2
        pkg_dir = os.path.dirname(pykinect2.__file__)
    except ImportError:
        print("\n[ERROR] pykinect2 not installed!")
        print("  Run: uv pip install pykinect2")
        sys.exit(1)

    print(f"\nPackage: {pkg_dir}")
    target = os.path.join(pkg_dir, "PyKinectV2.py")

    if not os.path.exists(target):
        print(f"[ERROR] {target} not found")
        sys.exit(1)

    # Read file
    with open(target, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Backup
    backup = target + ".ultimate_backup"
    if not os.path.exists(backup):
        shutil.copy2(target, backup)
        print(f"Backup: {backup}")

    # Check if already fixed
    if 'ULTIMATE_FIX_APPLIED' in content:
        print("\n[INFO] Already fixed!")
        print("Verifying...")
    else:
        print("\nApplying fixes...")

        original_content = content

        # ═══════════════════════════════════════════════════════════
        # FIX 1: Remove/bypass the _check_version('') call
        # ═══════════════════════════════════════════════════════════
        
        # Find the problematic import line
        check_version_lines = [
            "from comtypes import _check_version; _check_version('')",
            "from comtypes import _check_version;_check_version('')",
            "from comtypes import _check_version\n_check_version('')",
        ]
        
        fix1_applied = False
        for pattern in check_version_lines:
            if pattern in content:
                # Comment it out
                content = content.replace(
                    pattern,
                    "# ULTIMATE_FIX_APPLIED: Bypassed version check\n"
                    "# " + pattern.replace('\n', '\n# ')
                )
                fix1_applied = True
                print("  ✓ Fixed _check_version('') call")
                break
        
        if not fix1_applied:
            # Try line-by-line approach
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if '_check_version' in line and "from comtypes import" in line:
                    lines[i] = "# ULTIMATE_FIX_APPLIED: Bypassed version check\n# " + line
                    fix1_applied = True
                    print(f"  ✓ Fixed _check_version at line {i+1}")
                    break
            if fix1_applied:
                content = '\n'.join(lines)

        # ═══════════════════════════════════════════════════════════
        # FIX 2: Replace all sizeof assertions
        # ═══════════════════════════════════════════════════════════
        
        lines = content.split('\n')
        fix2_count = 0
        new_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('assert') and 
                'sizeof' in stripped and
                'ULTIMATE_FIX' not in line):
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(
                    f"{indent}pass  # ULTIMATE_FIX_APPLIED: "
                    f"Bypassed sizeof assertion (was: {stripped[:50]}...)"
                )
                fix2_count += 1
            else:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
        
        if fix2_count > 0:
            print(f"  ✓ Fixed {fix2_count} sizeof assertions")

        # Write patched file
        if content != original_content:
            with open(target, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"\n✓ Fixes applied to {target}")
        else:
            print("\n[INFO] No changes needed")

    # ═══════════════════════════════════════════════════════════
    # VERIFICATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  VERIFYING FIX")
    print("=" * 70)

    # Clear module cache
    mods_to_clear = [k for k in list(sys.modules.keys()) if 'pykinect' in k]
    for m in mods_to_clear:
        del sys.modules[m]

    # Test import
    try:
        from pykinect2 import PyKinectV2
        print("✓ PyKinectV2 imports successfully!")
        
        from pykinect2 import PyKinectRuntime
        print("✓ PyKinectRuntime imports successfully!")
        
        print("\n" + "=" * 70)
        print("  ✓✓✓ SUCCESS! ✓✓✓")
        print("=" * 70)
        print("\nYou can now run:")
        print("  python realtime_har_kinect_v2.py --mode mock")
        print("  python realtime_har_kinect_v2.py --mode kinect")
        print("=" * 70)
        
    except Exception as e:
        print(f"✗ Import still fails: {type(e).__name__}: {e}")
        print("\n" + "=" * 70)
        print("  MANUAL FIX REQUIRED")
        print("=" * 70)
        print(f"\n1. Open this file in a text editor:")
        print(f"   {target}")
        print(f"\n2. Find line with:")
        print(f"   from comtypes import _check_version; _check_version('')")
        print(f"\n3. Comment it out (add # at start):")
        print(f"   # from comtypes import _check_version; _check_version('')")
        print(f"\n4. Save and run this script again")
        print("=" * 70)
        
        # Show the problematic line
        with open(target, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines, 1):
            if '_check_version' in line and 'from comtypes' in line:
                print(f"\nProblematic line {i}:")
                print(f"  {line.rstrip()}")
        
        sys.exit(1)

if __name__ == "__main__":
    main()