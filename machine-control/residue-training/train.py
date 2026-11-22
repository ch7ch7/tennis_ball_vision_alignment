# train.py
import random
import numpy as np
import torch
import torch.nn as nn
import cv2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim
from pathlib import Path
import socket
import subprocess
import sys
import webbrowser
import time

from dataset import SimpleSegDataset, list_images
from model import UNet, DiceLoss
from config import (
    TRAIN_IMAGES_DIR, MASK_DIR, LOG_DIR, MODEL_DIR,
    IMG_SIZE, BATCH_SIZE, EPOCHS, LR, SEED, NUM_WORKERS,
    BASE_CHANNELS, VAL_RATIO, LOG_PATH, BEST_MODEL_PATH,
    LR_FACTOR, LR_PATIENCE,
    DEVICE
)

from torch.optim.lr_scheduler import ReduceLROnPlateau

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def find_free_port(start_port: int = 6006) -> int:
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # 關鍵：精準檢查本機端 (127.0.0.1)，這樣才能抓到 TensorBoard 的佔用狀態
                s.bind(('127.0.0.1', port))
                return port
            except socket.error:
                # 如果 6006 被佔用，就 +1 變成 6007 繼續試，依此類推
                port += 1
    raise IOError("找不到任何可用的連接埠")

def launch_tensorboard(log_dir: Path):
    try:
        port = find_free_port()
        print(f"找到可用連接埠: {port}，準備啟動 TensorBoard...")
        proc = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", str(log_dir), "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"TensorBoard 伺服器已在背景啟動，程序 ID: {proc.pid}")
        time.sleep(5)
        url = f"http://localhost:{port}"
        print(f"正在開啟瀏覽器前往: {url}")
        webbrowser.open(url)
        return proc
    except Exception as e:
        print(f"啟動 TensorBoard 時發生錯誤: {e}")
        return None

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def count_mask_stats(paths, mask_dir: Path, thr: int = 127):
    present = 0; positive = 0; empty = 0; missing = 0
    for p in paths:
        p = Path(p)
        mp = mask_dir / (p.stem + ".png")
        if not mp.exists():
            missing += 1
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            missing += 1
            continue
        present += 1
        if (m > thr).any():
            positive += 1
        else:
            empty += 1
    return present, positive, empty, missing

def print_stats(title, paths, mask_dir: Path):
    present, positive, empty, missing = count_mask_stats(paths, mask_dir)
    total = len(paths)
    print(f"[{title}] images={total} | masks: present={present}, positive={positive}, empty={empty}, missing={missing}")

def print_pos_neg_ratio(title, paths, mask_dir: Path):
    _, positive, empty, _ = count_mask_stats(paths, mask_dir)
    total = len(paths)
    print(f"[{title}] pos={positive}  neg={empty}  ratio={positive}/{total} ({positive/total:.2%} pos)")

# --- 訓練邏輯：換回 BCE 並套用高權重 ---
def train_epoch(model, loader, opt, bce, dice, device):
    model.train()
    total = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        opt.zero_grad()
        logits = model(imgs)
        loss = 1.0 * bce(logits, masks) + 0.5 * dice(logits, masks)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / max(1, len(loader))

def val_epoch(model, loader, bce, dice, device):
    model.eval()
    total_loss, total_tp, total_tn, total_fp, total_fn = 0.0, 0, 0, 0, 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = bce(logits, masks) + dice(logits, masks)
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).int()
            masks_int = masks.int()
            total_tp += (preds & masks_int).sum().item()
            total_tn += ((1 - preds) & (1 - masks_int)).sum().item()
            total_fp += (preds & (1 - masks_int)).sum().item()
            total_fn += ((1 - preds) & masks_int).sum().item()
    return {"loss": total_loss / max(1, len(loader)), "tp": total_tp, "tn": total_tn, "fp": total_fp, "fn": total_fn}

def main():
    set_seed(SEED)
    images = list_images(TRAIN_IMAGES_DIR)
    if not images:
        raise SystemExit(f"No images found in {TRAIN_IMAGES_DIR}")

    # 分別切分有膠與無膠影像，再合併兩類樣本。
    def is_positive_check(stem, mask_dir, thr=127):
        mp = mask_dir / f"{stem}.png"
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        return m is not None and (m > thr).any()

    pos_list, neg_list = [], []
    for p in images:
        stem = Path(p).stem
        (pos_list if is_positive_check(stem, MASK_DIR) else neg_list).append(p)

    random.shuffle(pos_list); random.shuffle(neg_list)
    def split(lst, ratio):
        nval = max(1, int(len(lst) * ratio))
        return lst[:nval], lst[nval:]

    val_pos, train_pos = split(pos_list, VAL_RATIO)
    val_neg, train_neg = split(neg_list, VAL_RATIO)
    val_paths = val_pos + val_neg
    train_paths = train_pos + train_neg
    random.shuffle(val_paths); random.shuffle(train_paths)

    # 輸出資料比例
    print_stats("TRAIN(stratified)", train_paths, MASK_DIR)
    print_pos_neg_ratio("TRAIN(stratified)", train_paths, MASK_DIR)
    print_stats("VAL(stratified)", val_paths, MASK_DIR)
    print_pos_neg_ratio("VAL(stratified)", val_paths, MASK_DIR)

    train_ds = SimpleSegDataset(train_paths, MASK_DIR, size=IMG_SIZE, augment=True)
    val_ds = SimpleSegDataset(val_paths, MASK_DIR, size=IMG_SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = UNet(in_ch=3, out_ch=1, base=BASE_CHANNELS, dropout=0.3).to(device)

    # 提高殘膠像素在二元交叉熵中的權重。
    pos_weight = torch.tensor([3.0], device=device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_dice = DiceLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE)
    writer = SummaryWriter(log_dir=str(LOG_PATH))
    launch_tensorboard(LOG_PATH)

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion_bce, criterion_dice, device)
        v = val_epoch(model, val_loader, criterion_bce, criterion_dice, device)
        
        val_loss = v["loss"]
        eps = 1e-6
        recall = v["tp"] / (v["tp"] + v["fn"] + eps)
        f1 = 2 * (v["tp"] / (v["tp"] + v["fp"] + eps) * recall) / (v["tp"] / (v["tp"] + v["fp"] + eps) + recall + eps)

        # 寫入 TensorBoard：將兩條 Loss 併在同一圖中
        writer.add_scalars("Loss/Combined", {"train": tr_loss, "val": val_loss}, epoch)
        writer.add_scalar("Val/Recall", recall, epoch)
        writer.add_scalar("Val/F1-Score", f1, epoch)
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[{epoch}/{EPOCHS}] Train/Loss={tr_loss:.4f} | Val/Loss={val_loss:.4f} | Recall={recall:.4f} | F1={f1:.4f} | LR={current_lr}")

        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch}, BEST_MODEL_PATH)
            print(f">>> Saved Best Model (Loss: {best_val:.4f})")

    writer.close()

if __name__ == "__main__":
    main()
