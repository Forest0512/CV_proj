import json
import os

# ==================== Settings ====================
JSON_FILE_PATH = "result4.json"  # Label Studio JSON path
OUTPUT_DIR = r"dataset/labels/train"  # Output TXT folder

# ⚠️ Set this according to the kpt_shape in your car_dataset.yaml:
# If kpt_shape: [3, 2] -> set to 2 (produces 11 columns)
# If kpt_shape: [3, 3] -> set to 3 (produces 13 columns, incl. visibility)
KPT_DIM = 2

# Keypoint order (must exactly match the order used for training)
KP_ORDER = ["head_left", "head_right", "tail"]
# ================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    # 1. Safely parse the file name (strip the Label Studio hash prefix)
    raw_img_path = item["image"]
    base_img_name = os.path.basename(raw_img_path)

    if "-" in base_img_name:
        # Split only on the first '-', keeping any '-' in the original file name
        real_img_name = base_img_name.split("-", 1)[1]
    else:
        real_img_name = base_img_name

    txt_name = os.path.splitext(real_img_name)[0] + ".txt"
    txt_path = os.path.join(OUTPUT_DIR, txt_name)

    # 2. Parse the bounding box
    labels = item.get("label", [])
    if not labels:
        continue

    bbox = labels[0]
    bx = (bbox["x"] + bbox["width"] / 2.0) / 100.0
    by = (bbox["y"] + bbox["height"] / 2.0) / 100.0
    bw = bbox["width"] / 100.0
    bh = bbox["height"] / 100.0
    class_id = 0

    # 3. Parse the keypoints
    kp_dict = {}
    kp_labels = item.get("kp-label", [])
    for kp in kp_labels:
        kp_name = kp["keypointlabels"][0]
        kx = kp["x"] / 100.0
        ky = kp["y"] / 100.0
        kp_dict[kp_name] = (kx, ky)

    # 4. Assemble keypoint values according to the configured dimension
    kps_values = []
    for kp_name in KP_ORDER:
        if kp_name in kp_dict:
            kx, ky = kp_dict[kp_name]
            if KPT_DIM == 3:
                # x, y, visibility (2 means visible and labeled)
                kps_values.extend([f"{kx:.6f}", f"{ky:.6f}", "2"])
            else:
                # Only x, y
                kps_values.extend([f"{kx:.6f}", f"{ky:.6f}"])
        else:
            # Pad with 0 if not labeled
            if KPT_DIM == 3:
                kps_values.extend(["0.000000", "0.000000", "0"])
            else:
                kps_values.extend(["0.000000", "0.000000"])

    # 5. Combine into a single line and write to TXT
    line = (
        f"{class_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f} "
        + " ".join(kps_values)
        + "\n"
    )

    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(line)

print(
    f"Conversion done! TXT files written to '{OUTPUT_DIR}', each label has {5 + len(KP_ORDER)*KPT_DIM} columns."
)