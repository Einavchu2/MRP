"""
Run this to verify which simulation.py is loaded and if the fix is active.
Usage: python debug_inv.py
"""
import sys, os
print("="*60)
print("Python:", sys.executable)
print("Working dir:", os.getcwd())
print()

# Check simulation.py on disk
sim_path = os.path.join(os.path.dirname(__file__), "src", "simulation.py")
print("simulation.py path:", sim_path)
with open(sim_path, encoding="utf-8", errors="replace") as f:
    content = f.read()

has_ffill = "ffill()" in content
has_cumulative = "cumulative_po" in content
has_original = "original_inv" in content

print(f"  ffill fix:        {'✅ YES' if has_ffill    else '❌ NO  ← FILE IS OLD'}")
print(f"  original_inv fix: {'✅ YES' if has_original  else '❌ NO  ← FILE IS OLD'}")
print(f"  cumulative_po:    {'✅ YES' if has_cumulative else '❌ NO  ← FILE IS OLD'}")

# Check app.py
app_path = os.path.join(os.path.dirname(__file__), "app.py")
with open(app_path, encoding="utf-8", errors="replace") as f:
    app_content = f.read()
has_orig_app = "original_inv" in app_content
has_cum_app  = "cumulative_po" in app_content
print(f"\napp.py:")
print(f"  original_inv fix: {'✅ YES' if has_orig_app else '❌ NO  ← FILE IS OLD'}")
print(f"  cumulative_po:    {'✅ YES' if has_cum_app  else '❌ NO  ← FILE IS OLD'}")

print()
if has_ffill and has_original and has_cumulative and has_orig_app:
    print("✅ ALL FIXES PRESENT — restart Streamlit (kill python.exe)")
else:
    print("❌ SOME FIXES MISSING — replace files from ZIP again")
print("="*60)
