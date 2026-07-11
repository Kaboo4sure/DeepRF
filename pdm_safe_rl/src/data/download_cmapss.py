import os
import shutil
import urllib.error
import urllib.request
import zipfile

CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"

ZIP_PATH = "data/raw/cmapss/CMAPSSData.zip"
OUT_DIR = "data/raw/cmapss/CMAPSSData"


def download_file(url: str, out_path: str, timeout: int = 120) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Downloading from: {url}")
    print(f"Saving to: {os.path.abspath(out_path)}")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with open(out_path, "wb") as output_file:
                shutil.copyfileobj(response, output_file)

    except (urllib.error.URLError, TimeoutError) as exc:
        if os.path.exists(out_path):
            os.remove(out_path)

        raise RuntimeError(
            "\nC-MAPSS download failed.\n"
            "Download CMAPSSData.zip manually from the official NASA "
            "C-MAPSS Open Data page, then place it at:\n"
            f"{os.path.abspath(out_path)}"
        ) from exc

    print(f"Saved to: {os.path.abspath(out_path)}")


def unzip(zip_path: str, out_dir: str) -> None:
    if not zipfile.is_zipfile(zip_path):
        raise zipfile.BadZipFile(
            f"The downloaded file is not a valid ZIP archive: {zip_path}"
        )

    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(out_dir)

    print(f"Extracted to: {os.path.abspath(out_dir)}")


def main() -> None:
    if not os.path.exists(ZIP_PATH):
        download_file(CMAPSS_URL, ZIP_PATH)
    else:
        print(f"Using existing ZIP: {os.path.abspath(ZIP_PATH)}")

    unzip(ZIP_PATH, OUT_DIR)


if __name__ == "__main__":
    main()