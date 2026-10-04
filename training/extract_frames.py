import argparse
import os
from pathlib import Path

import cv2

DESCRIPTION = "Extract every n-th frame of a video as JPEG images for labelling."
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset" / "images" / "add"


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("video_path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--step", type=int, default=6, help="keep one of every n frames")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    capture = cv2.VideoCapture(args.video_path)
    frame_count = 0
    saved_count = 0
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        if frame_count % args.step == 0:
            cv2.imwrite(os.path.join(args.output_dir, f"frame_{saved_count:04d}.jpg"), frame)
            saved_count += 1
        frame_count += 1
    capture.release()
    print(f"Done! Extracted {saved_count} images, saved to {args.output_dir}")


if __name__ == "__main__":
    main()
