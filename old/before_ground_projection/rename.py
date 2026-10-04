import os

folder_path = r"C:/Users/Aphrodite/Desktop/Master_Class/CV_proj/dataset/images/train"

# 2. Set the custom start number (e.g. 0 or 1)
start_num = 0

# 3. Get all jpg files in the folder and sort them by name
files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
files.sort()

print(f"Found {len(files)} files, renumbering starting from frame_{start_num:04d}...")

# Stage 1: rename to temporary names (avoids overwrite conflicts)
temp_files = []
for i, filename in enumerate(files):
    ext = os.path.splitext(filename)[1]
    temp_name = f"temp_{i:04d}{ext}"
    old_path = os.path.join(folder_path, filename)
    temp_path = os.path.join(folder_path, temp_name)
    os.rename(old_path, temp_path)
    temp_files.append(temp_name)

# Stage 2: rename using the custom start number
for i, temp_name in enumerate(temp_files):
    ext = os.path.splitext(temp_name)[1]
    current_num = start_num + i
    new_name = f"frame_{current_num:04d}{ext}"
    temp_path = os.path.join(folder_path, temp_name)
    new_path = os.path.join(folder_path, new_name)
    os.rename(temp_path, new_path)

print("Renumbering done!")