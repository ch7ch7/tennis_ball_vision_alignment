# generate_blur_images.py
import os
import cv2
from tqdm import tqdm
from config import PROCESSED_IMG_DIR, BLUR_DIR, IMG_SIZE

def make_one_blur(img, img_size=IMG_SIZE):
    """
    使用與 dataset.py 相同的模糊流程：
    - GaussianBlur
    - 縮小一半再放大回來（保留輪廓感）
    這裡輸入為 uint8 (0-255) single-channel，輸出也是 uint8 single-channel。
    """
    # 確保是灰階 single-channel np.ndarray
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 先 resize 到 IMG_SIZE（若已是則不變）
    img_resized = cv2.resize(img, img_size)
    # OpenCV 可以直接對 uint8 做 GaussianBlur
    blur = cv2.GaussianBlur(img_resized, (9, 9), 3)
    # 縮小一半再放大（保持輪廓但模糊）
    small = cv2.resize(blur, (img_size[0] // 2, img_size[1] // 2), interpolation=cv2.INTER_AREA)
    blur_big = cv2.resize(small, img_size, interpolation=cv2.INTER_LINEAR)
    # 回傳 uint8
    return blur_big

def generate_all(force_regen=False):
    os.makedirs(BLUR_DIR, exist_ok=True)
    fnames = sorted([f for f in os.listdir(PROCESSED_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not fnames:
        print(f"⚠️ 找不到檔案在 {PROCESSED_IMG_DIR}")
        return

    for fname in tqdm(fnames, desc="generate blur images"):
        src = os.path.join(PROCESSED_IMG_DIR, fname)
        dst = os.path.join(BLUR_DIR, fname)
        if os.path.exists(dst) and not force_regen:
            # 已存在且不強制重新產生 → 跳過
            continue
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ 讀取失敗：{src}，已略過")
            continue
        blur_img = make_one_blur(img)
        # 寫成灰階 uint8
        cv2.imwrite(dst, blur_img)

    print(f"✅ 模糊影像已輸出至：{BLUR_DIR}")

if __name__ == "__main__":
    generate_all()
