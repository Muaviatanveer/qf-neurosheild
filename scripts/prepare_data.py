# scripts/prepare_data.py
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def find_videos(directory: Path):
    video_exts = {".mp4", ".mov", ".avi"}
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in video_exts])

def create_synthetic_video(filepath: Path, is_fake: bool, num_frames: int = 24, size: int = 224):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(filepath), fourcc, 10.0, (size, size))

    for t in range(num_frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        # Draw a moving circle simulating a head/face
        cx = int(size // 2 + 20 * np.sin(t * 0.3))
        cy = int(size // 2 + 15 * np.cos(t * 0.3))
        cv2.circle(frame, (cx, cy), 50, (180, 200, 220), -1)

        if is_fake:
            # Inject synthetic manipulation artifact (grid noise + boundary jitter)
            noise = np.random.randint(0, 90, (40, 40, 3), dtype=np.uint8)
            frame[cy-20:cy+20, cx-20:cx+20] = cv2.add(frame[cy-20:cy+20, cx-20:cx+20], noise)

        writer.write(frame)
    writer.release()

def main():
    raw_dir = Path("data/raw")
    real_dir = raw_dir / "REAL"
    fake_dir = raw_dir / "FAKE"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".mov", ".avi"}
    real_videos = [p for p in real_dir.iterdir() if p.suffix.lower() in video_exts]
    fake_videos = [p for p in fake_dir.iterdir() if p.suffix.lower() in video_exts]

    if not real_videos and not fake_videos:
        print("No videos found in data/raw/. Generating 12 synthetic videos to verify pipeline...")
        for i in range(6):
            create_synthetic_video(real_dir / f"real_vid_{i:03d}.mp4", is_fake=False)
            create_synthetic_video(fake_dir / f"fake_vid_{i:03d}.mp4", is_fake=True)
        real_videos = list(real_dir.glob("*.mp4"))
        fake_videos = list(fake_dir.glob("*.mp4"))

    records = []
    for p in sorted(real_videos):
        records.append({"video_id": p.stem, "filepath": str(p), "label": "REAL", "manipulation": "none"})
    for p in sorted(fake_videos):
        records.append({"video_id": p.stem, "filepath": str(p), "label": "FAKE", "manipulation": "synthetic_or_ffpp"})

    df = pd.DataFrame(records)
    
    # Video-level stratified split (prevents frame leakage)
    df["split"] = "train"
    test_videos = {"real_1", "fake_3"}
    val_videos = {"real_3", "fake_1"}
    df.loc[df["video_id"].isin(test_videos), "split"] = "test"
    df.loc[df["video_id"].isin(val_videos), "split"] = "val"

    out_path = Path("data/splits/metadata.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved metadata to {out_path} with {len(df)} total videos:")
    print(df["split"].value_counts())

if __name__ == "__main__":
    main()