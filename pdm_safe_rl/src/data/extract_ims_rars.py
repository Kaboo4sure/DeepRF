import os
import subprocess

def find_7z():
    candidates = [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def extract_rar(seven_zip: str, rar_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nExtracting: {rar_path}")
    print(f"To:         {out_dir}")

    # 7z x archive.rar -oOUTDIR -y
    cmd = [seven_zip, "x", rar_path, f"-o{out_dir}", "-y"]
    subprocess.check_call(cmd)

def main():
    # Resolve project root so script works anywhere
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))

    base = os.path.join(
        project_root,
        "src", "data", "data", "raw",
        "ims_bearing", "IMS_Bearing_Data", "4. Bearings", "IMS_extracted"
    )

    if not os.path.exists(base):
        raise FileNotFoundError(f"IMS_extracted not found: {base}")

    seven_zip = find_7z()
    if seven_zip is None:
        raise FileNotFoundError(
            "7z.exe not found. Install 7-Zip or update find_7z() with the correct path."
        )

    rars = [f for f in os.listdir(base) if f.lower().endswith(".rar")]
    if not rars:
        print("No .rar files found. Maybe already extracted?")
        return

    for rar in rars:
        rar_path = os.path.join(base, rar)
        out_dir = os.path.join(base, os.path.splitext(rar)[0])  # 1st_test, 2nd_test, 3rd_test
        if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
            print(f"Skip (already extracted): {out_dir}")
            continue
        extract_rar(seven_zip, rar_path, out_dir)

    print("\nDone extracting all RARs.")

if __name__ == "__main__":
    main()
