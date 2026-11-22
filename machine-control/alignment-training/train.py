import argparse
import os
import random
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import TennisDataset
from model import MultiBranchEfficientNet
# 引入 Config 設定
from config import (
    CSV_FILE, PROCESSED_IMG_DIR, BLUR_DIR, BATCH_SIZE, LEARNING_RATE, EPOCHS,
    LOG_PATH, VAL_SPLIT, BEST_MODEL_PATH, TRAIN_USE_WEIGHTED_SAMPLER,
    VAL_USE_WEIGHTED_SAMPLER, TRAIN_AUGMENT, NOISE_PROB,
    USE_GPU, GPU_ID, NUM_WORKERS, PIN_MEMORY,
    NOISE_BLOCK_MIN, NOISE_BLOCK_MAX, IMG_SIZE
)
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from generate_labels import generate_labels_and_get_roll_range
from tqdm import tqdm
import generate_blur_images

# ✨ 新增：用於隨機數與 GPU 影像處理
import torch.nn.functional as F
# torchvision.transforms.functional 已不再用於模糊生成（改為從磁碟讀取）

# ✨ 使用新的 torch.amp 介面
from torch.amp import autocast, GradScaler

# ========================
# ⚖️ 建立雙變數 Weighted Sampler
# ========================
def make_weighted_sampler(df, init_bin_size=10, roll_bin_size=200, debug=False, save_dir="plots"):
    """建立同時平衡 init_angle + roll 的 Sampler"""
    df['init_bin'] = (df['init_angle'] // init_bin_size) * init_bin_size

    min_roll = int(df['roll'].min())
    max_roll = int(df['roll'].max()) + roll_bin_size
    bins = range(min_roll, max_roll + 1, roll_bin_size)
    df['roll_bin'] = pd.cut(df['roll'], bins=bins, right=True, include_lowest=True)

    def format_interval(interval):
        left = int(interval.left)
        right = int(interval.right)
        return f"[{left}, {right}]"

    df['roll_bin'] = df['roll_bin'].apply(format_interval)
    df['combo_bin'] = df['init_bin'].astype(str) + "_" + df['roll_bin'].astype(str)

    counts = df['combo_bin'].value_counts().sort_index()
    weights = 1.0 / counts[df['combo_bin']].values

    if debug:
        os.makedirs(save_dir, exist_ok=True)
        count_df = counts.reset_index()
        count_df.columns = ["combo_bin", "count"]
        csv_path = os.path.join(save_dir, "combo_bin_distribution.csv")
        count_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"📑 已輸出原始 combo_bin 分布表格: {csv_path}")

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# ========================
# 🚀 主程式
# ========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="訓練雙分支網球姿態模型")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0:
        parser.error("epochs 必須至少為 1，batch-size 至少為 2，workers 不得為負數")

    # ✨ 開啟 cuDNN Benchmark
    use_cuda = USE_GPU and not args.cpu and torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    if not os.path.isdir(PROCESSED_IMG_DIR):
        parser.error(f"找不到訓練影像目錄：{PROCESSED_IMG_DIR}")

    print("🔧 檢查並生成模糊影像（若尚未存在）...")
    generate_blur_images.generate_all(force_regen=False)

    print("生成 labels...")
    generate_labels_and_get_roll_range()

    labels = pd.read_csv(CSV_FILE)
    train_df, val_df = train_test_split(labels, test_size=VAL_SPLIT, random_state=0)
    if len(train_df) < args.batch_size:
        parser.error(f"訓練集只有 {len(train_df)} 筆，請減少 --batch-size 或加入更多樣本")
    angle_min, angle_max = train_df['angle'].min(), train_df['angle'].max()
    roll_min, roll_max = train_df['roll'].min(), train_df['roll'].max()
    if roll_min == roll_max:
        parser.error("訓練集至少需要兩種滾動量，才能正規化 R")

    print(f"📥 載入資料集 (GPU Augment: {TRAIN_AUGMENT}, Noise Prob: {NOISE_PROB})...")
    
    # Dataset 初始化（傳入預先生成的模糊圖目錄，避免在 GPU 上重複運算）
    train_set = TennisDataset(PROCESSED_IMG_DIR, train_df, angle_min, angle_max, roll_min, roll_max, blur_dir=BLUR_DIR)
    val_set = TennisDataset(PROCESSED_IMG_DIR, val_df, angle_min, angle_max, roll_min, roll_max, blur_dir=BLUR_DIR)

    # DataLoader
    if TRAIN_USE_WEIGHTED_SAMPLER:
        print("⚖️ 訓練集使用 Weighted Sampler")
        train_sampler = make_weighted_sampler(train_df, init_bin_size=10, roll_bin_size=200, debug=True)
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, sampler=train_sampler, drop_last=True,
            num_workers=args.workers, pin_memory=PIN_MEMORY and use_cuda
        )
    else:
        print("➡️ 訓練集使用隨機抽樣")
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.workers, pin_memory=PIN_MEMORY and use_cuda
        )
    
    if VAL_USE_WEIGHTED_SAMPLER:
        print("⚖️ 驗證集使用 Weighted Sampler")
        val_sampler = make_weighted_sampler(val_df, init_bin_size=10, roll_bin_size=200, debug=True, save_dir="plots/val_sampler")
        val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler,
                                num_workers=args.workers, pin_memory=PIN_MEMORY and use_cuda)
    else:
        print("➡️ 驗證集按既定順序載入 (shuffle=False)")
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=PIN_MEMORY and use_cuda)

    # 硬體選擇
    if use_cuda:
        device_str = f"cuda:{GPU_ID}"
        try:
            gpu_name = torch.cuda.get_device_name(GPU_ID)
            print(f"✅ 使用 GPU 加速: {gpu_name} ({device_str})")
        except:
            print(f"⚠️ 指定的 GPU ID {GPU_ID} 不存在，切換回 cuda:0")
            device_str = "cuda:0"
    else:
        device_str = "cpu"
        print("⚠️ 使用 CPU 進行訓練 (未檢測到 GPU 或已在 config 禁用)")
    
    device = torch.device(device_str)

    model = MultiBranchEfficientNet().to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    scaler = GradScaler(device.type, enabled=use_cuda)
    
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
    writer = SummaryWriter(LOG_PATH)
    loss_fn = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-8)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # ========================
        # 🟢 訓練階段 (Train)
        # ========================
        model.train()
        total_loss, total_init_sin, total_init_cos = 0, 0, 0
        total_sin, total_cos, total_roll = 0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for (x, x_blur), y in train_bar:
            x, x_blur, y = x.to(device), x_blur.to(device), y.to(device)

            # 🔥 關鍵修改：GPU 隨機區塊雜訊 (機率 50%)
            # 這樣模型既能學到抗噪能力，又不會忘記乾淨圖片的特徵
            if TRAIN_AUGMENT and random.random() < 0.5:
                # 1. 隨機決定這次的區塊大小
                current_block_size = random.randint(NOISE_BLOCK_MIN, NOISE_BLOCK_MAX)
                
                # 2. 生成遮罩
                H, W = x.shape[2], x.shape[3]
                h_small = H // current_block_size
                w_small = W // current_block_size
                
                small_mask = torch.rand(x.shape[0], 1, h_small, w_small, device=device) < NOISE_PROB
                large_mask = F.interpolate(small_mask.float(), size=(H, W), mode='nearest')
                
                # 3. 對原圖 x 填入白色噪點
                x.masked_fill_(large_mask > 0.5, 1.0)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=use_cuda):
                pred = model(x, x_blur)

                pred_isin, pred_icos, pred_sin, pred_cos, pred_roll = pred[:,0], pred[:,1], pred[:,2], pred[:,3], pred[:,4]
                true_isin, true_icos, true_sin, true_cos, true_roll = y[:,0], y[:,1], y[:,2], y[:,3], y[:,4]

                l_isin = loss_fn(pred_isin, true_isin)
                l_icos = loss_fn(pred_icos, true_icos)
                l_sin = loss_fn(pred_sin, true_sin)
                l_cos = loss_fn(pred_cos, true_cos)
                l_roll = loss_fn(pred_roll, true_roll)

                loss = l_isin + l_icos + l_sin + l_cos + l_roll

            # Scaler Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_init_sin += l_isin.item()
            total_init_cos += l_icos.item()
            total_sin += l_sin.item()
            total_cos += l_cos.item()
            total_roll += l_roll.item()

            train_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        # ========================
        # 🔵 驗證階段 (Validation)
        # ========================
        model.eval()
        val_loss, val_isin, val_icos, val_sin, val_cos, val_roll = 0,0,0,0,0,0
        total_mae_init_angle = 0.0 
        total_mae_curr_angle = 0.0
        total_mae_roll = 0.0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)
        with torch.no_grad():
            for (x, x_blur), y in val_bar:
                x, x_blur, y = x.to(device), x_blur.to(device), y.to(device)
                
                pred = model(x, x_blur)

                pred_isin, pred_icos, pred_sin, pred_cos, pred_roll = pred[:,0], pred[:,1], pred[:,2], pred[:,3], pred[:,4]
                true_isin, true_icos, true_sin, true_cos, true_roll = y[:,0], y[:,1], y[:,2], y[:,3], y[:,4]

                l_isin = loss_fn(pred_isin, true_isin)
                l_icos = loss_fn(pred_icos, true_icos)
                l_sin = loss_fn(pred_sin, true_sin)
                l_cos = loss_fn(pred_cos, true_cos)
                l_roll = loss_fn(pred_roll, true_roll)

                l_total = l_isin + l_icos + l_sin + l_cos + l_roll
                
                val_loss += l_total.item()
                val_isin += l_isin.item()
                val_icos += l_icos.item()
                val_sin += l_sin.item()
                val_cos += l_cos.item()
                val_roll += l_roll.item()

                # Metrics
                pred_init_deg = torch.rad2deg(torch.atan2(pred_isin, pred_icos))
                true_init_deg = torch.rad2deg(torch.atan2(true_isin, true_icos))
                diff_init = torch.abs(pred_init_deg - true_init_deg)
                diff_init = torch.min(diff_init, 360.0 - diff_init)
                total_mae_init_angle += diff_init.mean().item()

                pred_curr_deg = torch.rad2deg(torch.atan2(pred_sin, pred_cos))
                true_curr_deg = torch.rad2deg(torch.atan2(true_sin, true_cos))
                diff_curr = torch.abs(pred_curr_deg - true_curr_deg)
                diff_curr = torch.min(diff_curr, 360.0 - diff_curr)
                total_mae_curr_angle += diff_curr.mean().item()

                diff_roll = torch.abs(pred_roll - true_roll)
                total_mae_roll += diff_roll.mean().item()

                val_bar.set_postfix({"Batch Loss": f"{l_total.item():.4f}"})

        # --- 平均與記錄 ---
        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        avg_train_isin, avg_train_icos = total_init_sin/len(train_loader), total_init_cos/len(train_loader)
        avg_train_sin, avg_train_cos, avg_train_roll = total_sin/len(train_loader), total_cos/len(train_loader), total_roll/len(train_loader)
        
        avg_val_isin, avg_val_icos = val_isin/len(val_loader), val_icos/len(val_loader)
        avg_val_sin, avg_val_cos, avg_val_roll = val_sin/len(val_loader), val_cos/len(val_loader), val_roll/len(val_loader)

        avg_mae_init = total_mae_init_angle / len(val_loader)
        avg_mae_curr = total_mae_curr_angle / len(val_loader)
        avg_mae_roll = total_mae_roll / len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  🟢 Train Loss = {avg_train:.4f}")
        print(f"  🔵 Val   Loss = {avg_val:.4f}")
        print(f"  📏 Metrics -> Init_Ang: {avg_mae_init:.2f}°, Curr_Ang: {avg_mae_curr:.2f}°, Roll: {avg_mae_roll:.2f}")

        # TensorBoard
        writer.add_scalars("Loss/Total", {"train": avg_train, "val": avg_val}, epoch+1)
        writer.add_scalars("Loss/Init_sin", {"train": avg_train_isin, "val": avg_val_isin}, epoch+1)
        writer.add_scalars("Loss/Init_cos", {"train": avg_train_icos, "val": avg_val_icos}, epoch+1)
        writer.add_scalars("Loss/Sin", {"train": avg_train_sin, "val": avg_val_sin}, epoch+1)
        writer.add_scalars("Loss/Cos", {"train": avg_train_cos, "val": avg_val_cos}, epoch+1)
        writer.add_scalars("Loss/Roll (-1~1)", {"train": avg_train_roll, "val": avg_val_roll}, epoch+1)

        writer.add_scalars("Metric/Angle_Error_Deg", {
            "Init_Angle": avg_mae_init,
            "Current_Angle": avg_mae_curr
        }, epoch+1)
        
        writer.add_scalar("Metric/Roll_Error_Unit", avg_mae_roll, epoch+1)

        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  ⚡ Learning Rate = {current_lr}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✅ Best model saved to {BEST_MODEL_PATH}")

    writer.close()
