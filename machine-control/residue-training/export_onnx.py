# export_onnx.py
import torch
from pathlib import Path

from model import UNet
from config import IMG_SIZE, BASE_CHANNELS, BEST_MODEL_PATH

def main():
    # --- 設定 ---
    # 確保這個路徑指向您訓練好的最佳模型
    ckpt_path = BEST_MODEL_PATH 
    if not ckpt_path.exists():
        # 如果設定的路徑不存在，嘗試尋找最新的模型
        model_dir = Path("saved_models")
        try:
            ckpt_path = sorted(model_dir.glob("model_*_best.pth"), key=lambda p: p.stat().st_mtime)[-1]
        except IndexError:
            raise SystemExit(f"找不到任何模型檔案於 {model_dir}")

    output_onnx_path = Path("model.onnx")

    print(f"正在從 {ckpt_path} 載入模型...")

    # --- 載入模型 ---
    device = torch.device("cpu") # 匯出時使用 CPU 即可
    model = UNet(in_ch=3, out_ch=1, base=BASE_CHANNELS)

    # 載入模型權重
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    model.to(device)
    model.eval() # 設定為評估模式

    # --- 建立一個符合模型輸入尺寸的假資料 ---
    # 模型輸入 shape: [batch_size, channels, height, width]
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    # --- 匯出為 ONNX ---
    print(f"正在將模型匯出至 {output_onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        dynamo=False,
        opset_version=11, # 一個常用的穩定版本
        do_constant_folding=True,
        input_names=['input'],   # 輸入節點的名稱
        output_names=['output'], # 輸出節點的名稱
        dynamic_axes={ # 如果希望批次大小或圖片尺寸是動態的，可在此設定
            'input' : {0 : 'batch_size'},
            'output' : {0 : 'batch_size'}
        }
    )
    print("匯出成功！")

if __name__ == "__main__":
    main()
