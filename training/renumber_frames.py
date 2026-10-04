import argparse
import os
from pathlib import Path

DESCRIPTION = "Renumber all JPEG files of a folder to frame_XXXX.jpg (sorted by current name)."
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDER = PROJECT_ROOT / "data" / "dataset" / "images" / "train"


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--folder", default=str(DEFAULT_FOLDER))
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    files = sorted(name for name in os.listdir(args.folder) if name.lower().endswith(".jpg"))
    print(f"Found {len(files)} files, renumbering starting from frame_{args.start:04d}...")

    temporary_names = []
    for index, file_name in enumerate(files):
        temporary_name = f"temp_{index:04d}{os.path.splitext(file_name)[1]}"
        os.rename(os.path.join(args.folder, file_name), os.path.join(args.folder, temporary_name))
        temporary_names.append(temporary_name)

    for index, temporary_name in enumerate(temporary_names):
        new_name = f"frame_{args.start + index:04d}{os.path.splitext(temporary_name)[1]}"
        os.rename(os.path.join(args.folder, temporary_name), os.path.join(args.folder, new_name))

    print("Renumbering done!")


if __name__ == "__main__":
    main()
