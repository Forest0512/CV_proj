import json
import os

# ==================== 設定區 ====================
JSON_FILE_PATH = "result4.json"  # Label Studio JSON 路徑
OUTPUT_DIR = r"dataset/labels/train"  # 輸出 TXT 資料夾

# ⚠️ 請根據你的 car_dataset.yaml 的 kpt_shape 設定：
# 若 kpt_shape: [3, 2] -> 設為 2 (產生 11 欄)
# 若 kpt_shape: [3, 3] -> 設為 3 (產生 13 欄，含 visibility)
KPT_DIM = 2

# 關鍵點順序 (必須與模型訓練順序嚴格一致)
KP_ORDER = ["head_left", "head_right", "tail"]
# ================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    # 1. 安全解析檔名 (去除 Label Studio 的 Hash 前綴)
    raw_img_path = item["image"]
    base_img_name = os.path.basename(raw_img_path)

    if "-" in base_img_name:
        # 只切第一個 '-'，保留原檔名可能包含的 '-'
        real_img_name = base_img_name.split("-", 1)[1]
    else:
        real_img_name = base_img_name

    txt_name = os.path.splitext(real_img_name)[0] + ".txt"
    txt_path = os.path.join(OUTPUT_DIR, txt_name)

    # 2. 解析 Bounding Box
    labels = item.get("label", [])
    if not labels:
        continue

    bbox = labels[0]
    bx = (bbox["x"] + bbox["width"] / 2.0) / 100.0
    by = (bbox["y"] + bbox["height"] / 2.0) / 100.0
    bw = bbox["width"] / 100.0
    bh = bbox["height"] / 100.0
    class_id = 0

    # 3. 解析 Keypoints
    kp_dict = {}
    kp_labels = item.get("kp-label", [])
    for kp in kp_labels:
        kp_name = kp["keypointlabels"][0]
        kx = kp["x"] / 100.0
        ky = kp["y"] / 100.0
        kp_dict[kp_name] = (kx, ky)

    # 4. 依照指定維度組裝 Keypoint 數字
    kps_values = []
    for kp_name in KP_ORDER:
        if kp_name in kp_dict:
            kx, ky = kp_dict[kp_name]
            if KPT_DIM == 3:
                # x, y, visibility (2 代表可見且標註)
                kps_values.extend([f"{kx:.6f}", f"{ky:.6f}", "2"])
            else:
                # 只有 x, y
                kps_values.extend([f"{kx:.6f}", f"{ky:.6f}"])
        else:
            # 未標註時補 0
            if KPT_DIM == 3:
                kps_values.extend(["0.000000", "0.000000", "0"])
            else:
                kps_values.extend(["0.000000", "0.000000"])

    # 5. 組合成單行資料並寫入 TXT
    line = (
        f"{class_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f} "
        + " ".join(kps_values)
        + "\n"
    )

    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(line)

print(
    f"轉換完成！已生成 TXT 至 '{OUTPUT_DIR}'，每個標註包含 {5 + len(KP_ORDER)*KPT_DIM} 個欄位。"
)