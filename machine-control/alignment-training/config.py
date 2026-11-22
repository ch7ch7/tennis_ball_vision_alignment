# config.py
import os
from datetime import datetime

# ========================
# 🗂️ 資料與模型儲存設定
# ========================
DATA_DIR = "data"
RAW_IMG_DIR = os.path.join(DATA_DIR, "images_raw")              
PROCESSED_IMG_DIR = os.path.join(DATA_DIR, "images_processed")  
CSV_FILE = os.path.join(DATA_DIR, "train_labels.csv")           
BLUR_DIR = os.path.join(DATA_DIR, "images_blur")                

MODEL_DIR = "saved_models"   
LOG_DIR = "logs"             
PLOT_DIR = "plots"           

ROLL_MIN, ROLL_MAX = 700, 5600

# ========================
# 🖥️ 硬體加速設定
# ========================
USE_GPU = True          
GPU_ID = 1              
NUM_WORKERS = 8         
PIN_MEMORY = True       

# ========================
# 🧾 資料處理設定
# ========================
TRAIN_USE_WEIGHTED_SAMPLER = True    
VAL_USE_WEIGHTED_SAMPLER = False     

# ========================
# 🌪️ 數據增強設定 (Data Augmentation)
# ========================
TRAIN_AUGMENT = True       
NOISE_PROB = 0.02          
# ✨ 修改：移除固定大小，改為隨機範圍
NOISE_BLOCK_MIN = 3        # 最小區塊 (像素)
NOISE_BLOCK_MAX = 8       # 最大區塊 (像素)

# ========================
# 🧮 資料集分割設定
# ========================
VAL_SPLIT = 0.3      
RANDOM_SEED = 42     

# ========================
# 🎯 模型與訓練超參數
# ========================
BATCH_SIZE = 64         
IMG_SIZE = (224, 224)    
EPOCHS = 500
LEARNING_RATE = 1e-4

# ========================
# 🕐 模型與記錄命名
# ========================
EXPERIMENT_NAME = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
MODEL_PATH = os.path.join(MODEL_DIR, f"model_{EXPERIMENT_NAME}.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, f"model_{EXPERIMENT_NAME}_best.pth")
LOG_PATH = os.path.join(LOG_DIR, f"log_{EXPERIMENT_NAME}")
LOSS_PLOT_PATH = os.path.join(PLOT_DIR, f"loss_{EXPERIMENT_NAME}.png")