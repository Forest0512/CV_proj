import argparse
import json
import os
from pathlib import Path

DESCRIPTION = "Convert a Label Studio JSON export into YOLO pose label files."
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_PATH = PROJECT_ROOT / "data" / "labels" / "result4.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset" / "labels" / "train"
KEYPOINT_ORDER = ["head_left", "head_right", "tail"]
CLASS_ID = 0


def parse_args():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--export", default=str(DEFAULT_EXPORT_PATH), help="Label Studio JSON export")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="folder for the YOLO .txt labels")
    parser.add_argument(
        "--kpt-dim",
        type=int,
        choices=(2, 3),
        default=2,
        help="2 = x,y per keypoint (kpt_shape [3, 2]); 3 = x,y,visibility (kpt_shape [3, 3])",
    )
    return parser.parse_args()


def image_name_without_upload_prefix(image_path):
    base_name = os.path.basename(image_path)
    return base_name.split("-", 1)[1] if "-" in base_name else base_name


def keypoint_values(item, kpt_dim):
    keypoints = {
        keypoint["keypointlabels"][0]: (keypoint["x"] / 100.0, keypoint["y"] / 100.0)
        for keypoint in item.get("kp-label", [])
    }
    values = []
    for name in KEYPOINT_ORDER:
        if name in keypoints:
            x, y = keypoints[name]
            values.extend([f"{x:.6f}", f"{y:.6f}"] + (["2"] if kpt_dim == 3 else []))
        else:
            values.extend(["0.000000", "0.000000"] + (["0"] if kpt_dim == 3 else []))
    return values


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.export, "r", encoding="utf-8") as export_file:
        data = json.load(export_file)

    for item in data:
        labels = item.get("label", [])
        if not labels:
            continue
        box = labels[0]
        center_x = (box["x"] + box["width"] / 2.0) / 100.0
        center_y = (box["y"] + box["height"] / 2.0) / 100.0
        width = box["width"] / 100.0
        height = box["height"] / 100.0
        line = (
            f"{CLASS_ID} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} "
            + " ".join(keypoint_values(item, args.kpt_dim))
            + "\n"
        )
        label_name = os.path.splitext(image_name_without_upload_prefix(item["image"]))[0] + ".txt"
        with open(os.path.join(args.output_dir, label_name), "w", encoding="utf-8") as label_file:
            label_file.write(line)

    print(
        f"Conversion done! TXT files written to '{args.output_dir}', "
        f"each label has {5 + len(KEYPOINT_ORDER) * args.kpt_dim} columns."
    )


if __name__ == "__main__":
    main()
