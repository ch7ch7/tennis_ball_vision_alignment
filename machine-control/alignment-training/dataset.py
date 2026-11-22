import os
import cv2
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from config import IMG_SIZE

class TennisDataset(Dataset):
    def __init__(self, image_dir, labels_df, angle_min, angle_max, roll_min, roll_max, blur_dir=None):
        """
        Dataset：讀取原圖 (processed)、模糊圖 (blur) 與 Label。
        若提供 blur_dir，則從磁碟讀取預先生成的模糊圖；
        否則退回複製原圖作為佔位符（不建議用於正式訓練）。
        """
        self.image_dir = image_dir
        self.blur_dir = blur_dir
        self.labels = labels_df
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.roll_min = roll_min
        self.roll_max = roll_max

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        # 1. 讀取原圖
        img_path = os.path.join(self.image_dir, row['filename'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"❌ 圖片讀取失敗：{img_path}")
        
        # 2. 轉為 0.0 ~ 1.0 的浮點數
        img = cv2.resize(img, IMG_SIZE) / 255.0
        
        # 3. 轉為 Tensor
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        
        # 從磁碟讀取預先生成的模糊圖
        if self.blur_dir is not None:
            blur_path = os.path.join(self.blur_dir, row['filename'])
            blur = cv2.imread(blur_path, cv2.IMREAD_GRAYSCALE)
            if blur is None:
                raise FileNotFoundError(f"❌ 模糊圖讀取失敗：{blur_path}")
            blur = cv2.resize(blur, IMG_SIZE) / 255.0
            blur_tensor = torch.tensor(blur, dtype=torch.float32).unsqueeze(0)
        else:
            # 未提供 blur_dir 時以原圖佔位（僅供除錯，不應用於正式訓練）
            blur_tensor = img_tensor.clone()

        # 4. Label 處理
        init_angle_deg = row['init_angle']
        init_angle_rad = math.radians(init_angle_deg)
        init_sin = math.sin(init_angle_rad)
        init_cos = math.cos(init_angle_rad)

        angle_deg = row['angle']
        angle_rad = math.radians(angle_deg)
        sin = math.sin(angle_rad)
        cos = math.cos(angle_rad)

        roll = row['roll']
        norm_roll = 2 * (roll - self.roll_min) / (self.roll_max - self.roll_min) - 1

        label = torch.tensor([init_sin, init_cos, sin, cos, norm_roll], dtype=torch.float32)

        return (img_tensor, blur_tensor), label