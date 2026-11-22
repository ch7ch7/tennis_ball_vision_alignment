# config.py
from pathlib import Path
from datetime import datetime
# --- 資料與路徑 ---
time_now = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
LOG_DIR    = Path("logs")
MODEL_DIR  = Path("saved_models")
DATA_DIR   = Path("dataset")
LOG_PATH = LOG_DIR / f"log_{time_now}"
BEST_MODEL_PATH = MODEL_DIR / f"model_{time_now}_best.pth"
RAW_IMAGES_DIR  = DATA_DIR / "raw_images"
MASK_DIR   = DATA_DIR / "masks"
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"

# --- 訓練參數 ---
IMG_SIZE      = 256
BATCH_SIZE    = 32
EPOCHS        = 1600
LR            = 1e-4
SEED          = 42
NUM_WORKERS   = 0      # IDE/Windows 友善
BASE_CHANNELS = 32
VAL_RATIO     = 0.3    # 驗證集比例

# --- 學習率調整器 (LR Scheduler) ---
LR_FACTOR     = 0.5    # 每次調整學習率的倍率 (0.5 = 減半)
LR_PATIENCE   = 20      # 若 val_loss 無改善，等待多少 epoch 才調整

# --- 設備選擇 --- 
# 指定訓練使用的設備，例如 "cuda:0", "cuda:1", "cpu"
DEVICE        = "cuda:0"
