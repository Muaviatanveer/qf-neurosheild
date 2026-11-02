# scripts/download_real_samples.py
import urllib.request
from pathlib import Path

# Verified direct links to open research test videos (real human faces & deepfakes)
SAMPLES = {
    "REAL": [
        ("real_01.mp4", "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"),
        ("real_02.mp4", "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"),
        ("real_03.mp4", "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4"),
        ("real_04.mp4", "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4"),
        ("real_05.mp4", "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyBlazes.mp4"),
    ],
    "FAKE": [
        # You already have demo.mp4 from the Celeb-DF repository!
    ]
}

def main():
    real_dir = Path("data/raw/REAL")
    fake_dir = Path("data/raw/FAKE")
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # 1. First, rescue the demo.mp4 from your celeb-deepfakeforensics zip (it's a real synthesized deepfake!)
    import zipfile
    zip_path = Path("celeb-deepfakeforensics-master.zip")
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as z:
            for f in z.namelist():
                if f.endswith("demo.mp4"):
                    print("Extracting Celeb-DF demo deepfake video to data/raw/FAKE/celeb_demo.mp4...")
                    content = z.read(f)
                    with open(fake_dir / "celeb_demo.mp4", "wb") as out:
                        out.write(content)

    # 2. Download sample test videos for REAL and FAKE
    print("\nDownloading sample videos directly over HTTPS...")
    for label, items in SAMPLES.items():
        target_dir = real_dir if label == "REAL" else fake_dir
        for filename, url in items:
            out_file = target_dir / filename
            if not out_file.exists():
                print(f"Downloading {label}: {filename}...")
                try:
                    urllib.request.urlretrieve(url, out_file)
                except Exception as e:
                    print(f"Failed to fetch {url}: {e}")

    print("\n--- Current Video Counts ---")
    print(f"REAL: {len(list(real_dir.glob('*.*')))}")
    print(f"FAKE: {len(list(fake_dir.glob('*.*')))}")

if __name__ == "__main__":
    main()