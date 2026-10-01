import os

folder_path = r"C:/Users/Aphrodite/Desktop/Master_Class/CV_proj/dataset/images/train"

# 2. 設定自訂開始數字 (例如：0 或 1)
start_num = 0

# 3. 取得資料夾內所有 jpg 檔案並按名稱排序
files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
files.sort()

print(f"找到 {len(files)} 個檔案，將從 frame_{start_num:04d} 開始重新排序...")

# 第一階段：改名為臨時名稱（避免直接覆蓋衝突）
temp_files = []
for i, filename in enumerate(files):
    ext = os.path.splitext(filename)[1]
    temp_name = f"temp_{i:04d}{ext}"
    old_path = os.path.join(folder_path, filename)
    temp_path = os.path.join(folder_path, temp_name)
    os.rename(old_path, temp_path)
    temp_files.append(temp_name)

# 第二階段：依自訂起始數字重新命名
for i, temp_name in enumerate(temp_files):
    ext = os.path.splitext(temp_name)[1]
    current_num = start_num + i
    new_name = f"frame_{current_num:04d}{ext}"
    temp_path = os.path.join(folder_path, temp_name)
    new_path = os.path.join(folder_path, new_name)
    os.rename(temp_path, new_path)

print("重排完成！")