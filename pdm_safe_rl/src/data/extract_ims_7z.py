import os
import py7zr

def extract_7z(archive_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Extracting: {archive_path}")
    print(f"To:         {out_dir}")
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(path=out_dir)
    print("Done.")

def main():
    # Resolve project root reliably
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))

    base = os.path.join(
        project_root,
        "src", "data", "data", "raw",
        "ims_bearing", "IMS_Bearing_Data", "4. Bearings"
    )

    archive = os.path.join(base, "IMS.7z")
    out_dir = os.path.join(base, "IMS_extracted")

    if not os.path.exists(archive):
        raise FileNotFoundError(f"7z archive not found: {archive}")

    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        print(f"Already extracted: {out_dir}")
        return

    extract_7z(archive, out_dir)

if __name__ == "__main__":
    main()
