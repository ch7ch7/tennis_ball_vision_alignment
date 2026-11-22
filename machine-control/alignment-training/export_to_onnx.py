import torch
import glob
import os
from model import MultiBranchEfficientNet

# ========== 找最新的 *_best.pth ==========
def find_latest_best_model(model_dir="saved_models"):
    model_paths = glob.glob(os.path.join(model_dir, "*_best.pth"))
    if not model_paths:
        raise FileNotFoundError("❌ 沒找到 *_best.pth 模型，請先訓練模型")
    return max(model_paths, key=os.path.getmtime)

# ========== 主程式 ==========
if __name__ == "__main__":
    pth_path = find_latest_best_model()
    onnx_path = "model.onnx"

    print(f"✅ 使用模型: {pth_path}")

    device = torch.device("cpu")
    model = MultiBranchEfficientNet().to(device)

    # 載入最新權重
    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 測試輸入
    dummy_x1 = torch.randn(1, 1, 224, 224)
    dummy_x2 = torch.randn(1, 1, 224, 224)

    # 匯出 ONNX
    torch.onnx.export(
        model,
        (dummy_x1, dummy_x2),
        onnx_path,
        input_names=["input_a", "input_b"],
        output_names=["output"],
        dynamic_axes={
            "input_a": {0: "batch"},
            "input_b": {0: "batch"},
            "output": {0: "batch"}
        },
        opset_version=12,   # ✅ 建議換成 11 或 12 更穩定
        dynamo=False,       # 維持原版使用的 ONNX 匯出器與動態 batch 設定
        training=torch.onnx.TrainingMode.EVAL,  # ✅ 強制 eval 模式
        do_constant_folding=True                # ✅ 常量折疊，移除多餘 Dropout
    )

    print(f"🎉 已成功匯出 {onnx_path}，包含 {pth_path} 的權重")
