import os
import urllib.request
import zipfile

# NASA C-MAPSS is commonly distributed as a zip (often called CMAPSSData.zip
# NOTE: If this URL ever changes, search "NASA C-MAPSS CMAPSSData.zip"
CMAPSS_URL = "https://ti.arc.nasa.gov/c/6/"

def download_file(url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Downloading from: {url}")
    urllib.request.urlretrieve(url, out_path)
    print(f"Saved to: {out_path}")

def unzip(zip_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    print(f"Extracted to: {out_dir}")

def main():
    # Many NASA pages host the file behind a landing page.
    # If the direct download fails, manually download CMAPSSData.zip and place it into data/raw/cmapss/
    zip_path = "data/raw/cmapss/CMAPSSData.zip"
    out_dir = "data/raw/cmapss/CMAPSSData"

    if not os.path.exists(zip_path):
        download_file(CMAPSS_URL, zip_path)

    unzip(zip_path, out_dir)

if __name__ == "__main__":
    main()
