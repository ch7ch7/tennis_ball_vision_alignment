# generate_labels.py
import os
import re
import pandas as pd
from config import PROCESSED_IMG_DIR, CSV_FILE

# 新命名規則：_IA_{init_angle}_R_{roll}_A_{angle}.副檔名
FILENAME_PATTERN = re.compile(
    r"_IA_(-?\d+)_R_(-?\d+)_A_(-?\d+)\.(png|jpg|jpeg)$",
    re.IGNORECASE
)

def generate_labels_and_get_roll_range():
    filenames = sorted(os.listdir(PROCESSED_IMG_DIR))
    data = []

    for fname in filenames:
        match = FILENAME_PATTERN.search(fname)
        if not match:
            print(f"⚠️ 無法解析檔名：{fname}，已略過")
            continue

        init_angle = float(match.group(1))
        roll_units = float(match.group(2))
        angle = float(match.group(3))

        data.append({
            "filename": fname,
            "init_angle": init_angle,
            "angle": angle,
            "roll": roll_units
        })

    if not data:
        raise ValueError("❌ 沒有任何有效檔名可產生標註資料，請確認 images_processed/ 內容")

    df = pd.DataFrame(data)
    df.to_csv(CSV_FILE, index=False)
    roll_min = df['roll'].min()
    roll_max = df['roll'].max()
    print(f"✅ 產生訓練用標註檔：{CSV_FILE}（共 {len(df)} 筆）")
    print(f"🔍 roll 區間：{roll_min} ~ {roll_max}")
    return roll_min, roll_max

if __name__ == '__main__':
    generate_labels_and_get_roll_range()
