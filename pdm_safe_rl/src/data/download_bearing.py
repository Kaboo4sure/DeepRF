import os
import zipfile
import hashlib
import requests
from tqdm import tqdm

# NASA/PHM dataset public mirror (PCoE/PHM Society)
IMS_BEARING_ZIP_URL = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"

# Where to store in your repo (consistent with your C-MAPSS layout)
RAW_DIR = os.path.join("src", "data", "data", "raw", "ims_bearing")
ZIP_PATH = os.path.join(RAW_DIR, "bearings.zip")
EXTRACT_DIR = os.path.join(RAW_DIR, "extracted")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))

        with open(out_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=os.path.basename(out_path)
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract(zip_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def main():
    print("== NASA IMS Bearing dataset downloader ==")
    print("Target ZIP:", ZIP_PATH)
    print("Extract to:", EXTRACT_DIR)
    print("URL:", IMS_BEARING_ZIP_URL)

    if not os.path.exists(ZIP_PATH):
        print("\nDownloading...")
        download(IMS_BEARING_ZIP_URL, ZIP_PATH)
    else:
        print("\nZIP already exists, skipping download.")

    # Optional integrity printout (useful for reproducibility logs)
    print("\nSHA256:", sha256_file(ZIP_PATH))

    if not os.path.exists(EXTRACT_DIR) or len(os.listdir(EXTRACT_DIR)) == 0:
        print("\nExtracting...")
        extract(ZIP_PATH, EXTRACT_DIR)
    else:
        print("\nExtract directory already populated, skipping extraction.")

    print("\nDone.")


if __name__ == "__main__":
    main()
