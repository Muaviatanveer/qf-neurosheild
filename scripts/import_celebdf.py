# scripts/import_celebdf.py
import shutil
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="Ingest Celeb-DF into QF-NeuroShield")
    parser.add_argument("--source-dir", type=str, required=True, 
                        help="Path to the unzipped Celeb-DF folder (containing Celeb-real, Celeb-synthesis, etc.)")
    parser.add_argument("--max-videos-per-class", type=int, default=50,
                        help="Limit videos per class for fast Mac training (set to 0 for ALL videos)")
    parser.add_argument("--use-symlinks", action="store_true",
                        help="Use symlinks instead of copying files to save hard drive space")
    args = parser.parse_args()

    source = Path(args.source_dir).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source folder not found: {source}")

    raw_dir = Path("data/raw")
    real_target = raw_dir / "REAL"
    fake_target = raw_dir / "FAKE"
    real_target.mkdir(parents=True, exist_ok=True)
    fake_target.mkdir(parents=True, exist_ok=True)

    # 1. Clean existing synthetic or old files
    for old_file in real_target.glob("*.mp4"):
        old_file.unlink()
    for old_file in fake_target.glob("*.mp4"):
        old_file.unlink()

    # 2. Gather source video paths
    real_sources = list((source / "Celeb-real").glob("*.mp4")) + list((source / "YouTube-real").glob("*.mp4"))
    fake_sources = list((source / "Celeb-synthesis").glob("*.mp4"))

    print(f"Found {len(real_sources)} REAL videos and {len(fake_sources)} FAKE videos in source.")

    if args.max_videos_per_class > 0:
        real_sources = real_sources[:args.max_videos_per_class]
        fake_sources = fake_sources[:args.max_videos_per_class]
        print(f"Sampling {len(real_sources)} REAL and {len(fake_sources)} FAKE videos for this run.")

    # 3. Read official test list if available
    test_list_file = source / "List_of_testing_videos.txt"
    official_test_filenames = set()
    if test_list_file.exists():
        with open(test_list_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    official_test_filenames.add(Path(parts[1]).name)
        print(f"Loaded {len(official_test_filenames)} official test video filenames from List_of_testing_videos.txt")

    records = []

    def link_or_copy(src: Path, dst: Path):
        if args.use_symlinks:
            dst.symlink_to(src)
        else:
            shutil.copy2(src, dst)

    # Ingest REAL
    for src_path in real_sources:
        dst_path = real_target / src_path.name
        link_or_copy(src_path, dst_path)
        is_test = src_path.name in official_test_filenames
        records.append({
            "video_id": src_path.stem,
            "filepath": str(dst_path),
            "label": "REAL",
            "manipulation": "none",
            "split": "test" if is_test else "train"
        })

    # Ingest FAKE
    for src_path in fake_sources:
        dst_path = fake_target / src_path.name
        link_or_copy(src_path, dst_path)
        is_test = src_path.name in official_test_filenames
        records.append({
            "video_id": src_path.stem,
            "filepath": str(dst_path),
            "label": "FAKE",
            "manipulation": "celeb_synthesis",
            "split": "test" if is_test else "train"
        })

    df = pd.DataFrame(records)

    # Carve out validation set from train
    train_mask = df["split"] == "train"
    train_indices = df[train_mask].index
    
    # If the official test set didn't catch enough or none were sampled, stratify standard splits
    if (df["split"] == "test").sum() == 0:
        tr_idx, te_idx = train_test_split(df.index, test_size=0.2, stratify=df["label"], random_state=42)
        df.loc[te_idx, "split"] = "test"
        tr_idx, val_idx = train_test_split(tr_idx, test_size=0.2, stratify=df.loc[tr_idx, "label"], random_state=42)
        df.loc[val_idx, "split"] = "val"
    else:
        # Carve 15% of train for val
        tr_idx, val_idx = train_test_split(train_indices, test_size=0.15, stratify=df.loc[train_indices, "label"], random_state=42)
        df.loc[val_idx, "split"] = "val"

    metadata_path = Path("data/splits/metadata.csv")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_path, index=False)

    print("\n✅ Ingestion Complete!")
    print(f"Saved dataset metadata to {metadata_path}")
    print(df["split"].value_counts())

if __name__ == "__main__":
    main()