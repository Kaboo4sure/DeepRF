import os
import urllib.request
import zipfile

# NASA IMS Bearing run-to-failure dataset
# If this URL ever changes, search:
# "NASA IMS Bearing run to failure dataset zip"
IMS_BEARING_URL = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"


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
    # Keep folder layout consistent with C-MAPSS
    # If direct download fails, manually download and place zip in data/raw/ims_bearing/
    zip_path = "data/raw/ims_bearing/IMS_Bearing_Data.zip"
    out_dir = "data/raw/ims_bearing/IMS_Bearing_Data"

    if not os.path.exists(zip_path):
        download_file(IMS_BEARING_URL, zip_path)

    unzip(zip_path, out_dir)


if __name__ == "__main__":
    main()
